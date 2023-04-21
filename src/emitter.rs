
use macroquad::prelude::*;
use crate::{fluid_sim::*, fluid::*, shape::*};

#[derive(Clone)]
pub struct FluidEmitter {
    pub shape: Box<dyn Shape>,
    pub rate: f32,
    pub spawn_mass: f32,
    pub spawn_velocity: Vec2,
    pub spawn_temp: f32,
    pub spawn_fluid_id: u32,
}

impl FluidEmitter {
    pub fn tick<K: SimKernels>(&self, sim: &mut FluidSim<K>, dt: f32) {
        let expected_spawns = self.rate * dt;
        let to_spawn = expected_spawns.floor() as i32 + {
            (rand::gen_range(0., 1.) < expected_spawns.fract()) as i32
        };
        let consts = sim.fluid_type(self.spawn_fluid_id).clone();
        for _ in 0..to_spawn {
            sim.spawn_particle(FluidParticle { 
                m: self.spawn_mass, 
                x: self.shape.random_point(),
                v: self.spawn_velocity,
                f: Vec2::ZERO, 
                ci_grad: Vec2::ZERO,
                cp_grad: Vec2::ZERO,
                cs_grad: Vec2::ZERO,
                alpha: consts.alpha, 
                rho0: consts.alpha / self.spawn_temp,
                rho: consts.alpha / self.spawn_temp, 
                mu: consts.mu, 
                k: consts.k, 
                ci: consts.ci, 
                cs: consts.cs, 
                // sigma_i: consts.sigma_i,
                // sigma_s: consts.sigma_s,
                t: self.spawn_temp,
                next_t: self.spawn_temp,
                id: 0,
                fluid_id: self.spawn_fluid_id 
            });
        }
    }
}