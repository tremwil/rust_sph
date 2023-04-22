use std::{sync::{atomic::{AtomicU32, Ordering, compiler_fence}, Arc}, marker::PhantomData, cell};

use hashbrown::{HashMap, HashSet};
use itertools::Itertools;
use macroquad::prelude::*;
use rayon::prelude::*;

use crate::{fluid::{*, self}, shape::*};
use crate::kernels::*;
use crate::interacting_body::*;

/// Spatial hash grid cell.
#[derive(Default, Clone)]
pub struct GridCell
{
    /// Set of particles which currently lie in this grid cell.
    pub particles: HashMap<ParticleId, FluidParticle>,
    /// Boundaries: 
    pub boundaries: Vec<usize>,
    /// Interacting elements, stored here to accelerate collisions
    pub interacting_bodies: Vec<usize>, 
}

impl GridCell {
    /// Check if the grid cell is empty, i.e. does not contain any particles or boundary/sink references. 
    pub fn is_empty(&self) -> bool {
        return self.particles.is_empty() && self.boundaries.is_empty() && self.interacting_bodies.is_empty()
    }
}

#[derive(Clone)]
/// Spatial hash data structure, meant to accelerate range queries. 
pub struct SpatialHash {
    cells: HashMap<IVec2, GridCell>,
    particle_cells: HashMap<ParticleId, IVec2>,
    step: f32,
    inv_step: f32,
}
impl SpatialHash {
    pub fn new(grid_step: f32) -> Self {
        Self { 
            cells: HashMap::new(),
            particle_cells: HashMap::new(),
            step: grid_step,
            inv_step: 1. / grid_step
        }
    }

    /// Necessary workaround to compute indicies when inside an iterator that already has a &mut self reference
    pub fn index_no_borrow(position: Vec2, inv_step: f32) -> IVec2 {
        (position * inv_step).floor().as_ivec2()
    }

    /// Compute the index of the grid cell a particular position lies in. 
    pub fn index(&self, position: Vec2) -> IVec2 {
        (position * self.inv_step).floor().as_ivec2()
    }

    /// Parallel iterator over particles and the grid cells containing them. 
    pub fn particles_par(&self) -> impl ParallelIterator<Item = (&FluidParticle, &GridCell)> {
        self.particle_cells.par_iter().map(|(id, i)| {
            let grid = &self.cells[i];
            (&grid.particles[id], grid)
        })
    }

    pub fn get_boundary_cells<'a, S: Shape>(&self, shape: &'a S) -> impl Iterator<Item = IVec2> + 'a {
        let (min, max) = {
            let aabb = shape.aabb();
            (self.index(aabb.0 - self.step), self.index(aabb.1 + self.step))
        };

        let h = self.step;
        (min.x ..= max.x).cartesian_product(min.y ..= max.y)
            .filter(move |&(x, y)| {
                let v = ivec2(x, y).as_vec2() * h;
                shape.intersects_aabb(v, v + h) 
            })
            .flat_map(|(x, y)| (x-1 ..= x+1).cartesian_product(y-1 ..= y+1))
            .map(|(x,y)| ivec2(x,y))
            .unique()
    }

    fn register_boundary<S: Shape>(&mut self, id: usize, shape: &S) {
        for i in self.get_boundary_cells(shape) {
            self.cells.entry(i).or_default().boundaries.push(id);
        }
    }

    fn register_interacting_body<S: Shape>(&mut self, id: usize, shape: &S) {
        for i in self.get_boundary_cells(shape) {
            self.cells.entry(i).or_default().interacting_bodies.push(id);
        }
    }

    /// ### SAFETY:
    /// Particle with same ID is not currently present in the spatial hash grid. 
    pub fn add_particle(&mut self, p: FluidParticle) {
        let i = self.index(p.x);
        let id = p.id;
        self.cells.entry(i).or_default().particles.insert_unique_unchecked(id, p);
        self.particle_cells.insert_unique_unchecked(id, i);
    }

    /// Remove particle from the spatial hash grid, returning it if it was present. 
    pub fn remove_particle(&mut self, id: ParticleId) -> Option<FluidParticle> {
        let i = self.particle_cells.remove(&id)?;
        let cell = self.cells.get_mut(&i)?;
        cell.particles.remove(&id)
    }

    /// Update particle properties, including position information, and return a bool specifying if
    /// the particle should be destroyed or not. 
    pub fn move_and_filter_particles(&mut self, mut f: impl FnMut(&mut FluidParticle, &Vec<usize>, &Vec<usize>) -> bool) {
        self.particle_cells.drain_filter(|id, i| {
            let c = self.cells.get_mut(i).unwrap();
            let p = c.particles.get_mut(id).unwrap();

            if f(p, &c.boundaries, &c.interacting_bodies) {
                c.particles.remove(id);
                return true;
            }

            let i_new = Self::index_no_borrow(p.x, self.inv_step);
            if *i != i_new {
                let p = c.particles.remove(id).unwrap();
                self.cells.entry(i_new).or_default().particles.insert_unique_unchecked(*id, p);
                *i = i_new;
            }
            false
        });
    }

    pub fn move_particle(&mut self, id : ParticleId, new_pos: Vec2) -> Result<(), ()> {
        let i_new = self.index(new_pos);
        let i = self.particle_cells.get_mut(&id).ok_or(())?;
        if *i != i_new {
            let old_cell = self.cells.get_mut(i).ok_or(())?;
            let p = old_cell.particles.remove(&id).ok_or(())?;
            self.cells.entry(i_new).or_default().particles.insert_unique_unchecked(id, p);
            *i = i_new;
        }
        Ok(())
    }

    pub fn remove_empty_cells(&mut self) {
        self.cells.drain_filter(|_, cell| cell.is_empty());
    }

    pub fn range_query(&self, pos: Vec2, r: f32) -> impl Iterator<Item = &FluidParticle> {
        let low = self.index(pos - r);
        let high = self.index(pos + r);
        let r2 = r * r;

        (low.x ..= high.x)
            .cartesian_product(low.y ..= high.y)
            .filter_map(move |i| self.cells.get(&ivec2(i.0, i.1)))
            .flat_map(|c| c.particles.values())
            .filter(move |p| (p.x - pos).length_squared() < r2)
    }

    pub fn particle(&self, id: ParticleId) -> Option<&FluidParticle> {
        let pos = self.particle_cells.get(&id)?;
        self.cells.get(pos)?.particles.get(&id)
    }

    pub fn particle_mut(&mut self, id: ParticleId) -> Option<&mut FluidParticle> {
        let pos = self.particle_cells.get_mut(&id)?;
        self.cells.get_mut(pos)?.particles.get_mut(&id)
    }
}


/// Trait holding the different smoothing kernels which will be used by the fluid simulation.
pub trait SimKernels {
    /// Kernel used to particle densities.
    type DensityKernel : SmoothingKernel;
    /// Kernel used to compute pressure forces.
    type PressureKernel : SmoothingKernel;
    /// Kernel used to compute viscosity forces.
    type ViscosityKernel : SmoothingKernel;
    /// Kernel used to compute fluid-fluid interface forces 
    /// (interface tension, surface tension and fluid-air boundary field).
    type InterfaceKernel : SmoothingKernel;
    /// Kernel used to compute heat diffusion. 
    type HeatKernel: SmoothingKernel;
}

/// Default kernels for the fluid simulation.
pub struct DefaultSimKernels;
impl SimKernels for DefaultSimKernels {
    type DensityKernel = WSpiky;
    type PressureKernel = WSpiky;
    type ViscosityKernel = WViscosity;
    type InterfaceKernel = WPoly6;
    type HeatKernel = WViscosity;
}

#[derive(Clone)]
pub struct AirGenerationParams {
    pub generation_threshold: f32,
    pub spawn_position_offset: f32,

    pub deletion_threshold_p: f32,
    pub deletion_threshold_s: f32,
    pub deletion_threshold_rho: f32,

    pub air_buyoyancy: f32, // Artificial buoyancy for air particles
    pub air_mass: f32, // Mass of spawned air particles
    pub air_temp: f32, // Temperature of spawned air
    pub air_id: FluidId // Fluid ID of air particles
}

#[derive(Clone)]
pub struct FluidSim<K: SimKernels = DefaultSimKernels>
{   
    grid: SpatialHash,
    boundaries: Vec<Box<dyn Shape>>,
    interacting_bodies: Vec<Box<dyn InteractingStaticBody>>,
    fluid_types: Vec<FluidType>,
    h: f32,
    particle_id_counter: u32,

    pub gravity: Vec2,
    pub heat_diffusion: f32,
    pub air_gen_params: Option<AirGenerationParams>,
    pub sim_limits: Option<AABB>,

    k_phantom: PhantomData<fn() -> K>
}

impl<K: SimKernels> FluidSim<K>
{
    pub fn new(h: f32, gravity: Vec2, heat_diffusion: f32, air_gen_params: Option<AirGenerationParams>) -> Self {
        FluidSim { 
            grid: SpatialHash::new(h),
            boundaries: Vec::new(), 
            interacting_bodies : Vec::new(),
            fluid_types: Vec::new(),
            h, 
            particle_id_counter: 0,
            gravity,
            heat_diffusion,
            air_gen_params,
            k_phantom: PhantomData,
            sim_limits: None,
        }
    }

    pub fn with_sim_limits(mut self, sim_limits: AABB) -> Self {
        self.sim_limits = Some(sim_limits);
        self
    }

    pub fn h(&self) -> f32 {
        self.h
    }

    pub fn grid(&self) -> &SpatialHash {
        &self.grid
    }

    pub fn grid_mut(&mut self) -> &mut SpatialHash {
        &mut self.grid
    }

    pub fn fluid_type(&self, id: FluidId) -> &FluidType {
        self.fluid_types.get(id as usize).unwrap()
    }

    pub fn add_fluid_type(&mut self, fluid_type: FluidType) -> FluidId {
        self.fluid_types.push(fluid_type);
        (self.fluid_types.len() - 1) as FluidId
    }

    pub fn boundaries(&self) -> impl Iterator<Item = &Box<dyn Shape>> {
        self.boundaries.iter()
    }

    pub fn interacting_bodies(&self) -> impl Iterator<Item = &Box<dyn InteractingStaticBody>> {
        self.interacting_bodies.iter()
    }

    pub fn num_particles(&self) -> usize {
        self.grid.particle_cells.len()
    }

    pub fn add_boundaries(&mut self, boundaries: impl IntoIterator<Item = Box<dyn Shape>>) {
        for b in boundaries {
            self.grid.register_boundary(self.boundaries.len(), &b);
            self.boundaries.push(b);
        }
    }

    pub fn add_interacting_body(&mut self, body: impl InteractingStaticBody + 'static) -> usize {
        self.grid.register_interacting_body(self.interacting_bodies.len(), &body);
        self.interacting_bodies.push(Box::new(body));
        self.interacting_bodies.len() - 1
    }

    pub fn add_interacting_bodies(&mut self, bodies: impl IntoIterator<Item = Box<dyn InteractingStaticBody>>) {
        for b in bodies {
            self.grid.register_interacting_body(self.interacting_bodies.len(), &b);
            self.interacting_bodies.push(b);
        }
    }

    pub fn spawn_particle(&mut self, mut particle: FluidParticle) -> ParticleId {
        self.particle_id_counter += 1;
        particle.id = self.particle_id_counter;
        self.grid.add_particle(particle);
        self.particle_id_counter
    }
    
    pub fn delete_particle(&mut self, particle_id: ParticleId) -> Option<FluidParticle> {
        self.grid.remove_particle(particle_id)
    }

    pub fn clear_forces(&mut self) {
        for cell in self.grid.cells.values_mut() {
            for p in cell.particles.values_mut() {
                p.f = Vec2::ZERO;
            }
        }
    }

    pub fn particles(&self) -> impl Iterator<Item = &FluidParticle> {
        self.grid.cells.values().flat_map(|c| c.particles.values())
    }

    pub fn particles_mut(&mut self) -> impl Iterator<Item = &mut FluidParticle> {
        self.grid.cells.values_mut().flat_map(|c| c.particles.values_mut())
    }

    pub fn range_query(&self, pos: Vec2, r: f32) -> impl Iterator<Item = &FluidParticle> {
        self.grid.range_query(pos, r)
    }

    pub fn update_densities(&mut self, _: f32) {
        let grid = &self.grid;
        grid.particles_par().for_each(|(pi, _)| {
            let mut rho: f32 = 0.;
            for pj in grid.range_query(pi.x, self.h) {
                let r = pi.x - pj.x;
                let l = if K::DensityKernel::VALUE_NEEDS_SQRT { r.length() } else { 0. };
                rho += pj.m * K::DensityKernel::value_unchecked(r, l, self.h);
            }
            // SAFETY: This breaks Rust's invariants, but is safe? due to the way particles are updated
            compiler_fence(Ordering::Acquire);
            unsafe {
                let ptr = pi as *const FluidParticle as *mut FluidParticle;
                (*ptr).rho = rho;
            }
            compiler_fence(Ordering::Release);
        });
    }

    pub fn apply_fluid_forces(&mut self, dt: f32) {
        let grid = &self.grid;
        let fluid_types = &self.fluid_types;
        grid.particles_par().for_each(|(pi, _)| {
            let mut pressure_force = Vec2::ZERO;
            let mut viscosity_force = Vec2::ZERO;

            let ft = unsafe {
                fluid_types.get(pi.fluid_id as usize).unwrap_unchecked()
            };
            
            let mut ci_grad = Vec2::ZERO;
            let mut cs_grad = Vec2::ZERO;
            let mut cp_grad = Vec2::ZERO;

            let mut interface_curvature = 0.;// pi.m * pi.ci / pi.rho * K::InterfaceKernel::laplacian_unchecked(Vec2::ZERO, 0., self.h);
            let mut surface_curvature = 0.; //pi.m * pi.cs / pi.rho * K::InterfaceKernel::laplacian_unchecked(Vec2::ZERO, 0., self.h);

            let mut heat_diff : f32 = 0.;
            let pi_p = pi.k * (pi.rho - pi.rho0);

            for pj in grid.range_query(pi.x, self.h) {
                if pi.id == pj.id { continue; }

                let pj_p = pj.k * (pj.rho - pj.rho0);
                let r = pi.x - pj.x;
                let l = r.length();

                pressure_force -= pj.m * (pi_p + pj_p) / (2. * pj.rho) * K::PressureKernel::gradient_unchecked(r, l, self.h); 
                viscosity_force += (pi.mu + pj.mu) / 2. * pj.m * (pj.v - pi.v) / pj.rho * K::ViscosityKernel::laplacian_unchecked(r, l, self.h);

                let interface_gradient = K::InterfaceKernel::gradient_unchecked(r, l, self.h);
                ci_grad += pj.m * pj.ci / pj.rho * interface_gradient;
                cs_grad += pj.m * pj.cs / pj.rho * interface_gradient;
                cp_grad += pj.m / pj.rho * interface_gradient;

                let interface_laplacian = K::InterfaceKernel::laplacian_unchecked(r, l, self.h);
                interface_curvature += pj.m * pj.ci / pj.rho * interface_laplacian;
                surface_curvature += pj.m * pj.cs / pj.rho * interface_laplacian;

                heat_diff += pj.m * (pj.t - pi.t) / pj.rho * K::HeatKernel::laplacian_unchecked(r, l, self.h);
            }

            let interface_normal = if ci_grad.length_squared() >= 0.1 {
                ci_grad.normalize()
            } else {
                Vec2::ZERO
            };
            let surface_normal = if cs_grad.length_squared() >= 0.1 {
                cs_grad.normalize()
            } else {
                Vec2::ZERO
            };

            let interface_force = ft.sigma_i * interface_curvature * interface_normal;
            let surface_force = ft.sigma_s * surface_curvature * surface_normal;

            // SAFETY: This breaks Rust's invariants, but is safe? due to the way particles are updated
            compiler_fence(Ordering::Acquire);
            unsafe {
                let ptr = pi as *const FluidParticle as *mut FluidParticle;
                (*ptr).f += pressure_force + viscosity_force + interface_force + surface_force;
                (*ptr).ci_grad = ci_grad;
                (*ptr).cs_grad = cs_grad;
                (*ptr).cp_grad = cp_grad;
                (*ptr).next_t = pi.t + dt * self.heat_diffusion * heat_diff;
            }
            compiler_fence(Ordering::Release);
        });
    }

    pub fn tick(&mut self, dt: f32) {
        self.clear_forces();
        self.update_densities(dt);
        self.apply_fluid_forces(dt);

        let limits = self.sim_limits.clone();
        let opt_air_gen = self.air_gen_params.to_owned();
        let mut air_to_spawn : Vec<(Vec2, Vec2)> = Vec::new();

        self.grid.move_and_filter_particles(|p, boundaries, bodies| {
            // Simple sympletic Euler integration scheme
            let a = p.f / p.rho + self.gravity;
            p.v += a * dt;
            p.x += p.v * dt;

            p.t = p.next_t;
            p.rho0 = p.alpha / p.t;

            // Delete particles that escape simulation limits, if any are set
            if let Some(l) = &limits  {
                if !l.aa.cmple(p.x).all() || !l.bb.cmpge(p.x).all() {
                    return true;
                }
            }

            // Model the particle as a perfect sphere for boundary collisions
            let particle_radius =  (p.m / p.rho).sqrt() / 2.;

            // Interact with static bodies in the simulation
            if bodies.iter().any(|&bi| self.interacting_bodies[bi].interact(p, particle_radius, dt)) {
                return true;
            }

            // Solve collisions
            let mut collided = false;
            for &bi in boundaries.iter() {
                if let Some(c) = self.boundaries[bi].check_circle_collisions(p.x, particle_radius) {
                    // Push particle out of boundary, and perform elastic collision
                    collided = true;
                    p.x += c.normal * c.penetration;
                    p.v -= 2. * c.normal * p.v.dot(c.normal);
                }
            }

            // Handle air particle generation & destruction
            if let Some(air_gen) = &opt_air_gen {
                let cp_len = p.cp_grad.length_squared();

                // If this is an air particle...
                if p.fluid_id == air_gen.air_id {
                    // If this is an air particle that is far away from a surface
                    // and from other air particles, or very low-density, delete it.
                    if (cp_len > air_gen.deletion_threshold_p && p.cs_grad.length_squared() < air_gen.deletion_threshold_s) ||
                        p.rho < air_gen.deletion_threshold_rho {
                        return true;
                    }

                    // Apply artificial buoyancy
                    p.f -= air_gen.air_buyoyancy * (p.rho - p.rho0) * self.gravity;
                    return false;
                } 

                // If didn't collide yet, run collision detection again to make sure > h from a boundary
                collided = collided || boundaries.iter().any(|&b| {
                    self.boundaries[b].check_circle_collisions(p.x, self.h).is_some()
                });

                // If this particle is at a liquid-air boundary and facing downward, spawn an air particle. 
                if !collided && cp_len > air_gen.generation_threshold && p.cp_grad.dot(self.gravity) < 0. {
                    air_to_spawn.push((p.x - air_gen.spawn_position_offset * p.cp_grad, p.v));
                }
            }
            false
        });

        if let Some(air_gen) = &opt_air_gen {
            let air_fluid = self.fluid_type(air_gen.air_id).clone();
            for &(x, v) in air_to_spawn.iter() {
                let rho = air_fluid.alpha / air_gen.air_temp;
                self.spawn_particle(FluidParticle { 
                    m: air_gen.air_mass, 
                    x: x, 
                    v: v, 
                    f: Vec2::ZERO, 
                    ci_grad: Vec2::ZERO,
                    cp_grad: Vec2::ZERO, 
                    cs_grad: Vec2::ZERO, 
                    alpha: air_fluid.alpha, 
                    rho0: rho, 
                    rho: rho, 
                    mu: air_fluid.mu, 
                    k: air_fluid.k, 
                    ci: air_fluid.ci, 
                    cs: air_fluid.cs, 
                    t: air_gen.air_temp, 
                    next_t: air_gen.air_temp, 
                    id: 0, 
                    fluid_id: air_gen.air_id 
                });
            }
        }

        self.grid.remove_empty_cells();
    }

    pub fn draw_grid_cells(&self, thickness: f32, outline: Option<Color>, fill: Option<Color>) {
        for i in self.grid.cells.keys() {
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