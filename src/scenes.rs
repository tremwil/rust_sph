use itertools::Itertools;

use crate::{prelude::*, shape::Circle};

pub fn dam_break_air() -> GenericScene {
    GenericScene {
        name: "Dam Break With Air, 3x slow-mo".to_owned(),
        fps_limit: Some(60.),
        timestep: 1./3. * 1./60.,
        substeps: 7,
        custom_sim_setup: |sim| {
            for p in FluidParticle::fill_rect_id(
                0.003, 
                1., 
                vec2(-0.495, -0.295), 
                vec2(-0.39, 0.1), 
                sim,
                FluidInitParams { fluid_id: DEFAULT_WATER_ID, ..ROOM_TEMP_FLUID } 
            ) { 
                sim.spawn_particle(p);
            };
        },
        ..GenericScene::default_air()
    }
}

fn immiscible_shared(sim: &mut FluidSim, water: FluidId, oil: FluidId, ethanol: FluidId) {
    sim.add_boundaries([
        ConvexPolygon::new([
            vec2(-0.3, 0.0),
            vec2(-0.15, 0.0),
            vec2(-0.3, 0.1)
        ]).into_shape_box(),
        ConvexPolygon::new([
            vec2(0.3, 0.0),
            vec2(0.3, 0.1),
            vec2(0.15, 0.0),
        ]).into_shape_box()
    ]);
    for p in FluidParticle::fill_rect_id(
        0.005, 
        1., 
        vec2(-0.295, -0.295), 
        vec2(-0.015, -0.15), 
        sim,
        FluidInitParams { fluid_id: ethanol, ..ROOM_TEMP_FLUID } 
    ) { 
        sim.spawn_particle(p);
    };
    for p in FluidParticle::fill_rect_id(
        0.005, 
        1., 
        vec2(0.015, -0.295), 
        vec2(0.295, -0.15), 
        sim,
        FluidInitParams { fluid_id: ethanol, ..ROOM_TEMP_FLUID } 
    ) { 
        sim.spawn_particle(p);
    };
    for p in FluidParticle::fill_rect_id(
        0.005, 
        1., 
        vec2(-0.295, 0.1), 
        vec2(-0.15, 0.295), 
        sim,
        FluidInitParams { fluid_id: water, ..ROOM_TEMP_FLUID } 
    ) { 
        sim.spawn_particle(p);
    };
    for p in FluidParticle::fill_rect_id(
        0.005, 
        1., 
        vec2(0.15, 0.1), 
        vec2(0.295, 0.295), 
        sim,
        FluidInitParams { fluid_id: oil, ..ROOM_TEMP_FLUID } 
    ) { 
        sim.spawn_particle(p);
    };
}

pub fn immiscible_same_density() -> GenericScene {
    GenericScene {
        name: "Oil+Water+EtOH (same densities)".to_owned(),
        fps_limit: Some(60.),
        timestep: 0.003,
        substeps: 6,
        boundaries: AABB { aa: vec2(-0.3, -0.3), bb: vec2(0.3, 0.3) }
            .sides().map(|((s, e), n)| HalfPlane { 
                start: s,
                end: e,
                normal: -n
            }.into_shape_box())
            .collect_vec(),
        fluid_sim: FluidSim::new(0.02, vec2(0., -9.8), 0.001, Some(AirGenerationParams {
            generation_threshold: 3000.,
            spawn_position_offset: 0.00006,
            deletion_threshold_rho: 5.,
            air_mass: 0.005 * 0.005 * 100., 
            ..<GenericScene>::default_air().fluid_sim.air_gen_params.unwrap()
        })),
        custom_sim_setup: |sim| {
            sim.add_boundaries([
                AABB { aa: vec2(-0.01, -0.3), bb: vec2(0.01, 0.3) }.into_shape_box()
            ]);
            let water = sim.add_fluid_type(FluidType {
                alpha: 1000. * 273.15,
                sigma_s: 0.1,
                sigma_i: 0.5,      
                ..FluidType::WATER
            });
            let oil = sim.add_fluid_type(FluidType {
                alpha: 1000. * 273.15,
                sigma_s: 0.1,
                sigma_i: 0.5,
                ..FluidType::OIL
            });
            let ethanol = sim.add_fluid_type(FluidType {
                alpha: 1000. * 273.15,
                sigma_s: 0.1,
                sigma_i: 0.5,
                ..FluidType::ETHANOL
            });
            immiscible_shared(sim, water, oil, ethanol);
        },
        ..GenericScene::default_air()
    }
}

pub fn immiscible_different_densities() -> GenericScene {
    GenericScene {
        name: "Oil+Water+EtOH (realistic densities)".to_owned(),
        custom_sim_setup: |sim| {
            sim.add_boundaries([
                AABB { aa: vec2(-0.01, 0.), bb: vec2(0.01, 0.3) }.into_shape_box()
            ]);
            let water = sim.add_fluid_type(FluidType {
                alpha: 1000. * 273.15,
                sigma_s: 0.2,
                sigma_i: 0.2,      
                ..FluidType::WATER
            });
            let oil = sim.add_fluid_type(FluidType {
                alpha: 900. * 273.15,
                sigma_s: 0.1,
                sigma_i: 0.2,
                ..FluidType::OIL
            });
            let ethanol = sim.add_fluid_type(FluidType {
                alpha: 789. * 273.15,
                sigma_s: 0.1,
                sigma_i: 0.2,
                ..FluidType::ETHANOL
            });
            immiscible_shared(sim, water, oil, ethanol);
        },
        ..immiscible_same_density()
    }
}

pub fn two_fluid_fountain() -> GenericScene {
    let emitter = FluidEmitter {
        shape: Circle { center: vec2(0.0, 0.22), radius: 0.02 }.into_shape_box(),
        rate: 3000.,
        spawn_mass: 0.003 * 0.003 * 1000. * 273.15 / 293.,
        spawn_velocity: Vec2::ZERO,
        spawn_temp: 293.,
        spawn_fluid_id: 3,
    };
    GenericScene {
        name: "Water+EtOH Fountain (1/8 speed)".to_owned(),        
        fps_limit: Some(60.),
        timestep: 1./8. * 1./60.,
        substeps: 2,        
        fluid_sim: FluidSim::new(0.012, vec2(0., -9.8), 0.001, Some(AirGenerationParams {
            generation_threshold: 10000.,
            deletion_threshold_rho: 10.,
            air_mass: 0.003 * 0.003 * 100. * 273.15 / 293., 
            ..<GenericScene>::default_air().fluid_sim.air_gen_params.unwrap()
        })),
        mouse_interaction: MouseInteraction {
            effect_radius: (0.05, 0.003 .. 0.2),  
            force_coef: (2.5, 0. .. 10.),
            ..MouseInteraction::default()
        },
        emitters: vec![emitter],
        boundaries: AABB { aa: vec2(-0.3, -0.3), bb: vec2(0.3, 0.3) }
            .sides().map(|((s, e), n)| HalfPlane { 
                start: s,
                end: e,
                normal: -n
            }.into_shape_box())
            .collect_vec(),
        custom_sim_setup: |sim| {
            sim.add_boundaries([
                Circle { center: vec2(0., 0.07), radius: 0.07 }.into_shape_box()
            ]);
            sim.add_interacting_body(Sink {
                shape: Circle { center: vec2(0., -0.1), radius: 0.03 }.into_shape_box()
            });
            let water = sim.add_fluid_type(FluidType {
                alpha: 1000. * 273.15,
                sigma_s: 0.2,
                sigma_i: 0.2,      
                ..FluidType::WATER
            });
            let ethanol = sim.add_fluid_type(FluidType {
                alpha: 789. * 273.15,
                sigma_s: 0.1,
                sigma_i: 0.2,
                ..FluidType::ETHANOL
            });
            for p in FluidParticle::fill_rect_id(
                0.003, 
                1., 
                vec2(-0.295, -0.295), 
                vec2(0.295, -0.2), 
                sim,
                FluidInitParams { fluid_id: ethanol, ..ROOM_TEMP_FLUID } 
            ) { 
                sim.spawn_particle(p);
            };            
            for p in FluidParticle::fill_rect_id(
                0.003, 
                1., 
                vec2(-0.02, 0.2), 
                vec2(0.023, 0.243), 
                sim,
                FluidInitParams { fluid_id: water, ..ROOM_TEMP_FLUID } 
            ) { 
                sim.spawn_particle(p);
            };
        },
        ..GenericScene::default_air()
    }
}

pub fn high_viscosity_liquid() -> GenericScene {
    GenericScene {
        name: "High-Viscosity Material".to_owned(),
        fps_limit: Some(60.),
        timestep: 1. / 120.,
        substeps: 12,
        boundaries: AABB { aa: vec2(-0.3, -0.3), bb: vec2(0.3, 0.3) }
            .sides().map(|((s, e), n)| HalfPlane { 
                start: s,
                end: e,
                normal: -n
            }.into_shape_box())
            .collect_vec(),
        fluid_sim: FluidSim::new(0.02, vec2(0., -9.8), 0.001, None),
        custom_sim_setup: |sim| {
            let molten_rock = sim.add_fluid_type(FluidType {
                alpha: 2000. * 273.15,
                mu: 50.,
                k: 200.,
                ci: 0.,
                sigma_s: 0.,
                sigma_i: 0.,
                color: ORANGE,      
                ..FluidType::WATER
            });
            for p in FluidParticle::fill_rect_id(
                0.005, 
                1., 
                vec2(-0.295, 0.), 
                vec2(0., 0.2), 
                sim,
                FluidInitParams { fluid_id: molten_rock, ..ROOM_TEMP_FLUID } 
            ) { 
                sim.spawn_particle(p);
            };
            for p in FluidParticle::fill_rect_id(
                0.005, 
                1., 
                vec2(-0.295, -0.295), 
                vec2(0.295, -0.15), 
                sim,
                FluidInitParams { fluid_id: DEFAULT_WATER_ID, ..ROOM_TEMP_FLUID } 
            ) { 
                sim.spawn_particle(p);
            };
        },
        ..GenericScene::default_air()
    }
}

pub fn diffusion() -> GenericScene {
    GenericScene {
        name: "Heat Diffusion".to_owned(),
        fps_limit: Some(60.),
        timestep: 1. / 120.,
        substeps: 12,
        boundaries: AABB { aa: vec2(-0.3, -0.3), bb: vec2(0.3, 0.3) }
            .sides().map(|((s, e), n)| HalfPlane { 
                start: s,
                end: e,
                normal: -n
            }.into_shape_box())
            .collect_vec(),
        fluid_sim: FluidSim::new(0.02, vec2(0., -9.8), 0.001, Some(AirGenerationParams {
            generation_threshold: 3000.,
            spawn_position_offset: 0.00006,
            deletion_threshold_rho: 5.,
            air_mass: 0.005 * 0.005 * 100., 
            ..<GenericScene>::default_air().fluid_sim.air_gen_params.unwrap()
        })),
        custom_sim_setup: |sim| {
            let water = sim.add_fluid_type(FluidType {
                alpha: 1000. * 273.15,
                sigma_s: 0.1,
                sigma_i: 0.5,      
                ..FluidType::WATER
            });
            for p in FluidParticle::fill_rect_id(
                0.005, 
                1., 
                vec2(-0.295, -0.295), 
                vec2(0.295, -0.15), 
                sim,
                FluidInitParams { fluid_id: water, ..ROOM_TEMP_FLUID } 
            ) { 
                sim.spawn_particle(p);
            };
            sim.add_interacting_body(TempSource {
                shape: AABB { aa: vec2(-0.3, -0.3), bb: vec2(-0.15, -0.15) },
                temp: 500.,
                diffusion_coef: 0.01
            });
            sim.add_interacting_body(TempSource {
                shape: AABB { aa: vec2(0.15, -0.3), bb: vec2(0.3, -0.15) },
                temp: 200.,
                diffusion_coef: 0.05
            });
        },
        ..GenericScene::default_air()
    }
}

pub fn lava_lamp() -> GenericScene {
    GenericScene {
        name: "Lava Lamp".to_owned(),
        fps_limit: Some(60.),
        timestep: 1. / 400.,
        substeps: 6,
        boundaries: AABB { aa: vec2(-0.15, -0.3), bb: vec2(0.15, 0.3) }
            .sides().map(|((s, e), n)| HalfPlane { 
                start: s,
                end: e,
                normal: -n
            }.into_shape_box())
            .collect_vec(),
        fluid_sim: FluidSim::new(0.02, vec2(0., -9.8), 0.0001, None),
        custom_sim_setup: |sim| {
            let heavy = sim.add_fluid_type(FluidType {
                alpha: 1000. * 10.,
                sigma_s: 0.1,
                sigma_i: 0.5,
                mu: 1.,
                color: PURPLE,      
                ..FluidType::WATER
            });
            let light = sim.add_fluid_type(FluidType {
                alpha: 500. * 10.,
                sigma_s: 0.1,
                sigma_i: 0.5, 
                mu: 0.9,
                color: GREEN,     
                ..FluidType::WATER
            });
            for p in FluidParticle::fill_rect_id(
                0.005, 
                1.5, 
                vec2(-0.145, -0.295), 
                vec2(0.145, -0.15), 
                sim,
                FluidInitParams { fluid_id: heavy, t: 10., ..ROOM_TEMP_FLUID } 
            ) { 
                sim.spawn_particle(p);
            };
            for p in FluidParticle::fill_rect_id(
                0.005, 
                1., 
                vec2(-0.145, -0.15), 
                vec2(0.145, 0.295), 
                sim,
                FluidInitParams { fluid_id: light, t: 10., ..ROOM_TEMP_FLUID } 
            ) { 
                sim.spawn_particle(p);
            };
            sim.add_interacting_body(TempSource {
                shape: AABB { aa: vec2(-0.1, -0.4), bb: vec2(0.1, -0.25) },
                temp: 30.,
                diffusion_coef: 0.5,
            });
            sim.add_interacting_body(TempSource {
                shape: AABB { aa: vec2(-0.15, 0.2), bb: vec2(0.15, 0.3) },
                temp: 8.,
                diffusion_coef: 0.2
            });
            // sim.add_interacting_body(TempSource {
            //     shape: Circle { center: vec2(-0.15, 0.3), radius: 0.1 },
            //     temp: 250.,
            //     diffusion_coef: 0.1
            // });
            // sim.add_interacting_body(TempSource {
            //     shape: Circle { center: vec2(0., 0.3), radius: 0.15 },
            //     temp: 250.,
            //     diffusion_coef: 0.05
            // });
        },
        ..GenericScene::default_no_air()
    }
}

pub fn stress_test_1() -> GenericScene {
    GenericScene {
        name: "Fluid Impact Stress Test".to_owned(),
        timestep: 0.0005,
        substeps: 1,
        custom_sim_setup: |sim| {
            for p in FluidParticle::fill_rect_id(
                0.003, 
                1., 
                vec2(-0.495, -0.295), 
                vec2(-0.05, 0.1), 
                sim,
                FluidInitParams { fluid_id: DEFAULT_WATER_ID, v: vec2(1., 0.), ..ROOM_TEMP_FLUID } 
            ) { 
                sim.spawn_particle(p);
            };
            for p in FluidParticle::fill_rect_id(
                0.003, 
                1., 
                vec2(0.05, -0.295), 
                vec2(0.495, 0.1), 
                sim,
                FluidInitParams { fluid_id: DEFAULT_OIL_ID, v: vec2(-1., 0.), ..ROOM_TEMP_FLUID } 
            ) { 
                sim.spawn_particle(p);
            };
        },
        ..GenericScene::default_air()
    }
}

pub fn stress_test_2() -> GenericScene {
    GenericScene {
        name: "100K Stress Test".to_owned(),
        timestep: 0.0005,
        substeps: 1,
        fluid_sim: FluidSim::new(0.006, vec2(0., -9.8), 0.001, Some(AirGenerationParams {
            generation_threshold: 15000.,
            spawn_position_offset: 0.00002,
            deletion_threshold_rho: 5.,
            air_mass: 0.0015 * 0.0015 * 100., 
            ..<GenericScene>::default_air().fluid_sim.air_gen_params.unwrap()
        })),
        custom_sim_setup: |sim| {
            sim.add_boundaries([
                Circle { center: vec2(0.25, 0.), radius: 0.1 }.into_shape_box()
            ]);
            for p in FluidParticle::fill_rect_id(
                0.0015, 
                1., 
                vec2(-0.495, -0.295), 
                vec2(-0.05, 0.215), 
                sim,
                FluidInitParams { fluid_id: DEFAULT_WATER_ID, v: vec2(2., 0.), ..ROOM_TEMP_FLUID } 
            ) { 
                sim.spawn_particle(p);
            };
        },
        ..GenericScene::default_air()
    }
}