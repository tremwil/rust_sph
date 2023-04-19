use core::time;

use emitter::FluidEmitter;
use macroquad::color::hsl_to_rgb;
use shape::{HalfPlane, Shape, Circle};
use macroquad::prelude::*;
use macroquad::ui::{hash, root_ui, widgets};

mod kernels;
mod shape;
mod fluid;
mod fluid_sim;
mod instanced_circle_draw;
mod emitter;
mod scene;

use fluid::*;
use fluid_sim::FluidSim;

use instanced_circle_draw::InstancedCircleDrawStage;

#[macroquad::main("SPH")]
async fn main() {
    let mut sim = <FluidSim>::new(0.02, vec2(0., -9.8));

    let boundaries: [Box<dyn Shape>; 4] = [
        Box::new(HalfPlane { 
            start: Vec2::new(-0.35, 0.32), 
            end: Vec2::new(0.35, 0.32),
            normal: Vec2::new(0., -1.),
        }),
        Box::new(HalfPlane { 
            start: Vec2::new(-0.35, -0.05), 
            end: Vec2::new(-0.35, 0.32),
            normal: Vec2::new(1., 0.),
        }),
        Box::new(HalfPlane { 
            start: Vec2::new(0.35, -0.05), 
            end: Vec2::new(0.35, 0.32),
            normal: Vec2::new(-1., 0.),
        }),
        Box::new(HalfPlane { 
            start: Vec2::new(-0.35, -0.05), 
            end: Vec2::new(0.35, -0.05),
            normal: Vec2::new(0., 1.),
        }),
    ];
    sim.add_boundaries(boundaries);

    let sinks: [Box<dyn Shape>; 1] = [        
        Box::new(Circle {
            center: vec2(-0.2, 0.1),
            radius: 0.1
        })
    ];
    //sim.add_sinks(sinks);

    let water = sim.add_fluid_type(FluidType::WATER);
    let air = sim.add_fluid_type(FluidType::AIR);
    let oil = sim.add_fluid_type(FluidType::OIL);

    let water_source = FluidEmitter {
        shape: Box::new(Circle {
            center: vec2(-0.2, 0.1),
            radius: 0.1
        }),
        rate: 2000.,
        spawn_mass: 0.005 * 0.005 * FluidType::WATER.alpha / 273.15,
        spawn_temp: 300.,
        spawn_velocity: vec2(0., 0.),
        spawn_fluid_id: water
    };

    let (aa, bb) = water_source.shape.aabb();
    let water_particles = FluidParticle::fill_rect(0.005, aa, bb, 
        &FluidType::WATER, &FluidInitParams { v: Vec2::ZERO, t: 300., fluid_id: water });

    let oil_particles = FluidParticle::fill_rect(0.005, Vec2::new(0.,0.01), Vec2::new(0.3,0.20), 
        &FluidType::OIL, &FluidInitParams { v: Vec2::ZERO, t: 250., fluid_id: oil });

    for p in water_particles.into_iter().chain(oil_particles.into_iter()) {
        sim.spawn_particle(p);
    }

    let gl = unsafe { get_internal_gl() };
    let mut circle_drawer = InstancedCircleDrawStage::new(gl.quad_context, 1 << 20);
    drop(gl);

    request_new_screen_size(1600., 900.);

    let mut inv_timestep : f32 = 1000.;
    let mut substeps: f32 = 1.;
    let mut is_paused: bool = true;
    let mut draw_velocities: bool = false;
    let mut draw_grid_cells: bool = false;
    let mut draw_temperatures : bool = false;

    let mut screen_center = vec2(0., 0.15);
    let mut zoom : f32 = 1500.;

    let mut last_mouse_pos: Option<Vec2> = None;

    loop {
        clear_background(WHITE);
        
        let sw = screen_width();
        let sh = screen_height();

        let display_rect = Rect {
            x: screen_center.x - 0.5 * sw / zoom,
            y: screen_center.y + 0.5 * sh / zoom,
            w: sw / zoom,
            h: -sh / zoom
        };

        let camera = Camera2D::from_display_rect(display_rect);
        set_camera(&camera);

        if !is_paused {
            let s = substeps.floor() as u32;
            let timestep = 1. / inv_timestep / s as f32;

            for _ in 0..s {
                sim.tick(timestep);
                //water_source.tick(&mut sim, timestep);
            }
        }

        if draw_grid_cells {
            sim.draw_grid_cells(0.001, None, Some(PINK))
        }

        water_source.shape.draw(0.001, Some(DARKBLUE), Some(SKYBLUE), None);

        circle_drawer.clear();

        const MIN_TEMP: f32 = 250.0;
        const MAX_TEMP: f32 = 300.0;

        let mut min_temp: f32 = f32::INFINITY;
        let mut max_temp: f32 = f32::NEG_INFINITY;
        for p in sim.particles() {
            let r =  (p.m / p.rho).sqrt() / 2.;
            min_temp = min_temp.min(p.t);
            max_temp = max_temp.max(p.t);
            let color = if draw_temperatures {
                let h = 0.333 - ((p.t - MIN_TEMP) / (MAX_TEMP - MIN_TEMP)).clamp(0., 1.) * 0.333;
                hsl_to_rgb(h, 1., 0.5)
            } else {
                sim.fluid_type(p.fluid_id).color
            };
            circle_drawer.push_circle(p.x, r, color);
        }

        circle_drawer.draw(&camera, &mut unsafe { get_internal_gl() });

        if draw_velocities {
            for p in sim.particles() {
                let vel_end = p.x + p.v * 0.01;
                draw_line(p.x.x, p.x.y, vel_end.x, vel_end.y, 0.001, RED);
            }
        }

        for b in sim.boundaries() {
            b.draw(0.001, Some(DARKGRAY), None, Some(RED));
        }

        widgets::Window::new(hash!(), vec2(10., 40.), vec2(500., 300.))
        .label("Simulation Control")
        .movable(false)
        .ui(&mut *root_ui(), |ui| {            
            ui.label(None, &format!("FPS: {:.0}", 1. / get_frame_time()));
            ui.label(None, &format!("# Particles: {}", sim.num_particles()));
            ui.label(None, &format!("Min temp: {:.3} K", min_temp));
            ui.label(None, &format!("Max temp: {:.3} K", max_temp));
            
            ui.separator();
            ui.slider(hash!(), "1/timestep", 100. .. 5000., &mut inv_timestep);
            ui.slider(hash!(), "substeps", 1. .. 20., &mut substeps);

            ui.checkbox(hash!(), "Draw velocities", &mut draw_velocities);
            ui.checkbox(hash!(), "Draw grid cells", &mut draw_grid_cells);
            ui.checkbox(hash!(), "Draw temperatures", &mut draw_temperatures);
            if ui.button(None, if is_paused { "Resume "} else { "Pause "}) {
                is_paused = !is_paused;
            }
        });

        if !root_ui().is_mouse_captured() && is_mouse_button_pressed(MouseButton::Left) {
            last_mouse_pos = Some(mouse_position().into());
        }
        else if is_mouse_button_down(MouseButton::Left) {
            let cpos : Vec2 = mouse_position().into();
            if let Some(lpos) = last_mouse_pos {
                let delta = camera.screen_to_world(cpos) - camera.screen_to_world(lpos);
                screen_center -= delta;
                last_mouse_pos = Some(cpos);
            }
        }
        else if is_mouse_button_released(MouseButton::Left) {
            last_mouse_pos = None;
        }

        if mouse_wheel().1 > 0. {
            zoom *= 1.1;
        }
        else if mouse_wheel().1 < 0. {
            zoom *= 0.9;   
        }

        let (mx, my) = mouse_position();
        let mouse_coords = camera.screen_to_world(vec2(mx, my));
        set_default_camera();
        draw_text(&format!("({:.3}, {:.3})", mouse_coords.x, mouse_coords.y), mx, my, 10., BLACK);

        next_frame().await
    }
}