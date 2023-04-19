use std::{sync::atomic::{AtomicU32, Ordering, compiler_fence}, marker::PhantomData};

use hashbrown::{HashMap, HashSet};
use itertools::Itertools;
use macroquad::prelude::*;
use rayon::prelude::*;

use crate::{fluid::{*, self}, shape::*};
use crate::kernels::*;

#[derive(Default)]
struct GridCell
{
    /// Set of particles which currently lie in this grid cell.
    pub particles: HashMap<ParticleId, FluidParticle>,
    /// Set of boundary indices which intersect this grid cell. Used to accelerate collisions.
    pub boundaries: Vec<usize>,
    // Set of particle sinks which intersect this grid cell. Used to accelerate collisions. 
    pub sinks: Vec<usize>
}

impl GridCell {
    pub fn is_empty(&self) -> bool {
        return self.particles.is_empty() && self.boundaries.is_empty() && self.sinks.is_empty()
    }
}

struct SpatialHash {
    grid: HashMap<IVec2, GridCell>,
    particle_cells: HashMap<ParticleId, IVec2>,
    grid_step: f32,
    inv_grid_step: f32,
}
impl SpatialHash {
    pub fn new(grid_step: f32) -> Self {
        Self { 
            grid: HashMap::new(),
            particle_cells: HashMap::new(),
            grid_step,
            inv_grid_step: 1. / grid_step
        }
    }

    pub fn grid_step(&self) -> f32 {
        self.grid_step
    }

    pub fn inv_grid_step(&self) -> f32 {
        self.inv_grid_step
    }

    pub fn grid_index(&self, position: Vec2) -> IVec2 {
        (position * self.inv_grid_step).floor().as_ivec2()
    }

    pub fn get_cell(&self, i: IVec2) -> Option<&GridCell> {
        self.grid.get(&i)
    }

    pub fn get_cell_mut(&mut self, i: IVec2) -> Option<&mut GridCell> {
        self.grid.get_mut(&i)
    }

    pub fn particle_cells_par(&self) -> impl ParallelIterator<Item = (&FluidParticle, &GridCell)> {
        self.particle_cells.par_iter().map(|(id, i)| {
            let grid = &self.grid[i];
            (&grid.particles[id], grid)
        })
    }

    pub fn get_boundary_cells<'a, S: Shape>(&self, shape: &'a S) -> impl Iterator<Item = IVec2> + 'a {
        let (min, max) = {
            let aabb = shape.aabb();
            (self.grid_index(aabb.0 - self.grid_step), self.grid_index(aabb.1 + self.grid_step))
        };

        let h = self.grid_step;
        (min.x ..= max.x).cartesian_product(min.y ..= max.y)
            .filter(move |&(x, y)| {
                let v = ivec2(x, y).as_vec2() * h;
                shape.boundary_intersects_aabb(v, v + h) 
            })
            .flat_map(|(x, y)| (x-1 ..= x+1).cartesian_product(y-1 ..= y+1))
            .map(|(x,y)| ivec2(x,y))
            .unique()
    }

    fn register_sink<S : Shape>(&mut self, id: usize, shape: &S) {
        for i in self.get_boundary_cells(shape) {
            self.grid.entry(i).or_default().sinks.push(id);
        }
    }

    fn register_boundary<S: Shape>(&mut self, id: usize, shape: &S) {
        for i in self.get_boundary_cells(shape) {
            self.grid.entry(i).or_default().boundaries.push(id);
        }
    }

    pub fn add_particle(&mut self, p: FluidParticle) {
        let i = self.grid_index(p.x);
        self.grid.entry(i).or_default().particles.insert(p.id, p);
        self.particle_cells.insert(p.id, i);
    }

    pub fn remove_particle(&mut self, id: ParticleId) -> Option<FluidParticle> {
        let i = self.particle_cells.remove(&id)?;
        let cell = self.grid.get_mut(&i)?;
        cell.particles.remove(&id)
    }

    pub fn move_particle(&mut self, id : ParticleId, new_pos: Vec2) -> Result<(), ()> {
        let i_new = self.grid_index(new_pos);
        let i = self.particle_cells.get_mut(&id).ok_or(())?;
        if *i != i_new {
            let old_cell = self.grid.get_mut(i).ok_or(())?;
            let p = old_cell.particles.remove(&id).ok_or(())?;
            self.grid.entry(i_new).or_default().particles.insert(id, p);
            *i = i_new;
        }
        Ok(())
    }

    pub fn remove_empty_cells(&mut self) {
        self.grid.drain_filter(|i, cell| cell.is_empty());
    }
}


/// Trait holding the different smoothing kernels which will be used by the fluid simulation.
pub trait SimKernels {
    type DensityKernel : SmoothingKernel;
    type PressureKernel : SmoothingKernel;
    type ViscosityKernel : SmoothingKernel;
    type InterfaceKernel : SmoothingKernel;
    type HeatKernel: SmoothingKernel;
}

/// Default kernels for the fluid simulation.
pub struct DefaultSimKernels;
impl SimKernels for DefaultSimKernels {
    type DensityKernel = WSpiky;
    type PressureKernel = WSpiky;
    type ViscosityKernel = WViscosity;
    type InterfaceKernel = WViscosity;
    type HeatKernel = WViscosity;
}

pub struct FluidSim<K: SimKernels = DefaultSimKernels>
{   
    grid: HashMap<IVec2, GridCell>,
    boundaries: Vec<Box<dyn Shape>>,
    sinks: Vec<Box<dyn Shape>>,
    fluid_types: Vec<FluidType>,
    particle_positions: HashMap<ParticleId, IVec2>,
    h: f32,
    inv_grid_step: f32,
    particle_id_counter: AtomicU32,
    pub gravity: Vec2,
    pub heat_diffusion: f32,

    k_phantom: PhantomData<fn() -> K>
}

impl<K: SimKernels> FluidSim<K>
{
    pub fn new(h: f32, gravity: Vec2) -> Self {
        FluidSim { 
            grid: HashMap::new(), 
            boundaries: Vec::new(), 
            sinks: Vec::new(),
            fluid_types: Vec::new(),
            particle_positions: HashMap::new(), 
            h, 
            inv_grid_step: h.recip(),
            particle_id_counter: AtomicU32::new(0),
            gravity,
            heat_diffusion: 0.001,
            k_phantom: PhantomData
        }
    }

    pub fn fluid_type(&self, id: FluidId) -> &FluidType {
        self.fluid_types.get(id as usize).unwrap()
    }

    pub fn add_fluid_type(&mut self, fluid_type: FluidType) -> FluidId {
        self.fluid_types.push(fluid_type);
        (self.fluid_types.len() - 1) as FluidId
    }

    fn grid_index_no_borrow(pos: Vec2, inv_grid_step: f32) -> IVec2 {
        (pos * inv_grid_step).floor().as_ivec2()
    }

    pub fn grid_index(&self, pos: Vec2) -> IVec2 {
        (pos * self.inv_grid_step).floor().as_ivec2()
    }

    pub fn boundaries(&self) -> impl Iterator<Item = &Box<dyn Shape>> {
        self.boundaries.iter()
    }

    pub fn num_particles(&self) -> usize {
        self.particle_positions.len()
    }

    pub fn add_boundaries(&mut self, boundaries: impl IntoIterator<Item = Box<dyn Shape>>) {
        for b in boundaries {
            for i in self.get_boundary_cells(&b) {
                let cell = self.grid.entry(i).or_default();
                cell.boundaries.push(self.boundaries.len());
            }
            self.boundaries.push(b);
        }
    }

    pub fn add_sinks(&mut self, sinks: impl IntoIterator<Item = Box<dyn Shape>>) {
        for s in sinks {
            for i in self.get_boundary_cells(&s) {
                let cell = self.grid.entry(i).or_default();
                cell.sinks.push(self.sinks.len());
            }
            self.sinks.push(s);
        }
    }

    pub fn spawn_particle(&mut self, mut particle: FluidParticle) -> ParticleId {
        particle.id = self.particle_id_counter.fetch_add(1, Ordering::Relaxed);

        let i = self.grid_index(particle.x);
        let cell = self.grid.entry(i).or_default();

        self.particle_positions.insert(particle.id, i);
        *cell.particles.insert_unique_unchecked(particle.id, particle).0
    }
    
    pub fn delete_particle(&mut self, particle_id: ParticleId) -> Option<FluidParticle> {
        let i = self.particle_positions.remove(&particle_id)?;
        let cell = self.grid.get_mut(&i)?;
        let particle = cell.particles.remove(&particle_id);

        // TODO: Cache grid cells for some time, instead of pre-emptively destroying them!
        if cell.is_empty() {
            self.grid.remove(&i);
        }

        particle
    }

    pub fn clear_forces(&mut self) {
        for cell in self.grid.values_mut() {
            for p in cell.particles.values_mut() {
                p.f = Vec2::ZERO;
            }
        }
    }

    pub fn particles(&self) -> impl Iterator<Item = &FluidParticle> {
        self.grid.values().flat_map(|c| c.particles.values())
    }

    pub fn particles_mut(&mut self) -> impl Iterator<Item = &mut FluidParticle> {
        self.grid.values_mut().flat_map(|c| c.particles.values_mut())
    }

    pub fn range_query(&self, pos: Vec2, r: f32) -> impl Iterator<Item = &FluidParticle> {
        let low = self.grid_index(pos - r);
        let high = self.grid_index(pos + r);
        let r2 = r * r;

        (low.x ..= high.x)
            .cartesian_product(low.y ..= high.y)
            .filter_map(move |i| self.grid.get(&ivec2(i.0, i.1)))
            .flat_map(|c| c.particles.values())
            .filter(move |p| (p.x - pos).length_squared() < r2)
    }

    pub fn update_densities(&mut self, dt: f32) {
        self.particle_positions.par_iter().for_each(|(id, i)| {
            let pi = self.grid[i].particles[id];
            let mut rho: f32 = 0.;
            for pj in self.range_query(pi.x, self.h) {
                rho += pj.m * K::DensityKernel::value_unchecked(pj.x, (pi.x - pj.x).length(), self.h);
            }
            // SAFETY: This breaks Rust's invariants, but is safe? due to the way particles are updated
            compiler_fence(Ordering::SeqCst);
            unsafe {
                let ptr = pi as *const FluidParticle as *mut FluidParticle;
                (*ptr).rho = rho;
            }
        });
    }

    pub fn apply_fluid_forces(&mut self, dt: f32) {
        for pi in self.particles() {
            let mut pressure_force = Vec2::ZERO;
            let mut viscosity_force = Vec2::ZERO;
            
            let mut interface_normal = Vec2::ZERO;
            let mut surface_normal = Vec2::ZERO;

            let mut interface_curvature = 0.;// pi.m * pi.ci / pi.rho * K::InterfaceKernel::laplacian_unchecked(Vec2::ZERO, 0., self.h);
            let mut surface_curvature = 0.; //pi.m * pi.cs / pi.rho * K::InterfaceKernel::laplacian_unchecked(Vec2::ZERO, 0., self.h);

            let mut heat_diff : f32 = 0.;
            let pi_p = pi.k * (pi.rho - pi.rho0);

            for pj in self.range_query(pi.x, self.h) {
                if pi.id == pj.id { continue; }

                let pj_p = pj.k * (pj.rho - pj.rho0);
                let r = pi.x - pj.x;
                let l = r.length();

                pressure_force -= pj.m * (pi_p + pj_p) / (2. * pj.rho) * K::PressureKernel::gradient_unchecked(r, l, self.h); 
                viscosity_force += (pi.mu + pj.mu) / 2. * pj.m * (pj.v - pi.v) / pj.rho * K::ViscosityKernel::laplacian_unchecked(r, l, self.h);

                let interface_gradient = K::InterfaceKernel::gradient_unchecked(r, l, self.h);
                interface_normal += pj.m * pj.ci / pj.rho * interface_gradient;
                surface_normal += pj.m * pj.cs / pj.rho * interface_gradient;

                let interface_laplacian = K::InterfaceKernel::laplacian_unchecked(r, l, self.h);
                interface_curvature += pj.m * pj.ci / pj.rho * interface_laplacian;
                surface_curvature += pj.m * pj.cs / pj.rho * interface_laplacian;

                heat_diff += pj.m * (pj.t - pi.t) / pj.rho * K::HeatKernel::laplacian_unchecked(r, l, self.h);
            }

            interface_normal = if interface_normal.length_squared() >= 0.001 {
                interface_normal.normalize()
            } else {
                Vec2::ZERO
            };
            surface_normal = if surface_normal.length_squared() >= 0.001 {
                surface_normal.normalize()
            } else {
                Vec2::ZERO
            };

            let interface_force = pi.sigma_i * interface_curvature * interface_normal;
            let surface_force = pi.sigma_s * surface_curvature * surface_normal;

            // SAFETY: This breaks Rust's invariants, but is safe due to the way particles are updated
            compiler_fence(Ordering::SeqCst);
            unsafe {
                let ptr = pi as *const FluidParticle as *mut FluidParticle;
                (*ptr).f += pressure_force + viscosity_force + interface_force + surface_force;
                (*ptr).next_t = pi.t + dt * self.heat_diffusion * heat_diff;
            }
        }
    }

    pub fn integrate_and_collide(&mut self, dt: f32) {
        let mut deletion_queue : Vec<ParticleId> = Vec::new();

        for (&id, i) in self.particle_positions.iter_mut() {
            let grid = &mut self.grid;
            let cell = grid.get_mut(i).unwrap();
            let p = cell.particles.get_mut(&id).unwrap();

            let a = p.f / p.rho + self.gravity;
            p.v += a * dt;
            p.x += p.v * dt;

            p.t = p.next_t;
            p.rho0 = p.alpha / p.t;

            // Model the particle as a perfect sphere for boundary collisions
            let particle_radius =  (p.m / p.rho).sqrt() / 2.;

            // Temporary (read: permanent) fix to make borrow checker happy
            for &bi in &cell.boundaries {
                if let Some(c) = self.boundaries[bi].check_circle_collisions(p.x, particle_radius) {
                    // Push particle out of boundary, and perform elastic collision
                    p.x += c.normal * c.penetration;
                    p.v -= 2. * c.normal * p.v.dot(c.normal);
                }
            }

            // Check if particle should be deleted due to touching a sink
            for &si in &cell.sinks {
                if self.sinks[si].check_circle_collisions(p.x, particle_radius).is_some() {
                    deletion_queue.push(id);
                }
            }

            let i_new = Self::grid_index_no_borrow(p.x, self.inv_grid_step);
            let id = p.id;

            if i_new != *i {
                let p = cell.particles.remove(&id).unwrap();
                if cell.is_empty() {
                   grid.remove(i);
                }
                let new_cell = grid.entry(i_new).or_default();
                // SAFETY: Particle IDs guaranteed unique
                new_cell.particles.insert_unique_unchecked(id, p);
                *i = i_new;
            }
        }

        for id in deletion_queue {
            self.delete_particle(id);
        }
    }

    pub fn tick(&mut self, dt: f32) {
        self.clear_forces();
        self.update_densities(dt);
        self.apply_fluid_forces(dt);
        self.integrate_and_collide(dt);
    }

    pub fn draw_grid_cells(&self, thickness: f32, outline: Option<Color>, fill: Option<Color>) {
        for i in self.grid.keys() {
            let pos = i.as_vec2() * self.h;
            if let Some(c) = fill {
                draw_rectangle(pos.x, pos.y, self.h, self.h, c);
            }
            if let Some(c) = outline {
                draw_rectangle_lines(pos.x, pos.y, self.h, self.h, thickness, c);
            }
        }
    }
}