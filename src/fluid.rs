use macroquad::prelude::*;

pub type ParticleId = u32;
pub type FluidId = u32;

/// Fluid simulation data stored per-particle. 
/// Some things (like the stiffness, color and interface/surface coefs) are properties of the particle type,
/// but copied here for cache performance. 
#[derive(Default, Clone, Debug)]
pub struct FluidParticle
{
    pub m: f32, // Mass (kg)
    pub x: Vec2, // Position (m)
    //pub xp: Vec2, // Last position (m)
    pub v: Vec2, // Velocity (m/s)
    pub f: Vec2, // Body forces (N/m^2)
    pub alpha: f32, // Gas constant (rest density = alpha / temp) (K*kg/m^2)
    pub rho0: f32, // Rest density (computed from gas constant) (kg/m^2)
    pub rho: f32, // Actual density (kg/m^2)
    pub mu: f32, // Viscosity (Ns/m)
    pub k: f32, // Stiffness (Nm/kg)
    pub ci: f32, // Color for interface tension (-0.5 for polar particle, 0.5 for non-polar, 0 for air)
    pub cs: f32, // Color for surface tension (0 for air particle, 1 for liquid particle)
    pub sigma_i: f32, // interface tension coefficient
    pub sigma_s: f32, // surface tension coefficient
    pub t: f32, // Temperature (K)
    pub next_t: f32, // Next temperature (K)
    pub id: ParticleId, // Particle ID, assigned uniquely by simulation
    pub fluid_id: FluidId // Fluid type ID, used for rendering
}

/// Constant properties of a fluid. 
#[derive(Default, Clone, Debug)]
pub struct FluidType
{
    pub alpha: f32,
    pub mu: f32,
    pub k: f32,
    pub ci: f32,
    pub cs: f32,
    pub sigma_i: f32,
    pub sigma_s: f32,
    pub color: Color
}

impl FluidType {
    pub const WATER: Self = Self {
        alpha: 1000. * 273.15,
        mu: 1.,
        k: 50.,
        ci: -0.5,
        cs: 1.,
        sigma_i: 0.1,
        sigma_s: 0.01,
        color: BLUE
    };

    pub const OIL: Self = Self {
        alpha: 500. * 273.15,
        mu: 1.,
        k: 50.,
        ci: 0.5,
        cs: 1.,
        sigma_i: 0.1,
        sigma_s: 0.01,
        color: YELLOW
    };

    pub const ETHANOL: Self = Self {
        alpha: 789. * 273.15,
        mu: 2.,
        k: 10.,
        ci: -0.5,
        cs: 1.,
        sigma_i: 0.3,
        sigma_s: 0.3,
        color: SKYBLUE
    };

    pub const AIR: Self = Self {
        alpha: 100. * 273.15,
        mu: 0.5,
        k: 5.,
        ci: 0.,
        cs: 0.,
        sigma_i: 0.,
        sigma_s: 0.,
        color: Color::new(0.5, 0.5, 0.5, 0.5)
    };
}

#[derive(Default, Clone, Debug)]
pub struct FluidInitParams
{
    pub v: Vec2,
    pub t: f32,
    pub fluid_id: u32
}

impl FluidParticle
{
    pub fn fill_rect(dist: f32, min: Vec2, max: Vec2, consts: &FluidType, params: &FluidInitParams) -> Vec<Self> {        
        let mass = 0.95 * consts.alpha * dist * dist / params.t;
        let sz = ((max - min) / dist).floor().as_ivec2();
        let num = sz.x * sz.y;

        let mut particles : Vec<Self> = Vec::with_capacity(num as usize);
        for i in 0..sz.x {
            for j in 0..sz.y {
                let x = min + dist * IVec2::new(i, j).as_vec2();
                particles.push(FluidParticle { 
                    m: mass, 
                    x,
                    v: params.v,
                    f: Vec2::ZERO, 
                    alpha: consts.alpha, 
                    rho0: consts.alpha / params.t,
                    rho: consts.alpha / params.t, 
                    mu: consts.mu, 
                    k: consts.k, 
                    ci: consts.ci, 
                    cs: consts.cs, 
                    sigma_i: consts.sigma_i,
                    sigma_s: consts.sigma_s,
                    t: params.t, 
                    next_t: params.t,
                    id: 0,
                    fluid_id: params.fluid_id 
                });
            }
        }
        particles
    }
}