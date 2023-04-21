use std::{time::Duration, ops::Range, f32::consts::PI};

use hashbrown::{HashMap, HashSet};
use macroquad::{prelude::*, color::hsl_to_rgb, ui::{hash, root_ui, widgets}};
use itertools::Itertools;
use crate::{
    fluid::*,
    emitter::*,
    fluid_sim::*, 
    interacting_body::*, 
    instanced_circle_draw::*,
    shape::{*, self}
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParticleColoringMode {
    Type,
    InterfaceGradient,
    SurfaceGradient,
    Density,
    Compression,
    Temperature
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SceneControlCmd {
    None,
    Reset,
    LoadPrevious,
    LoadNext,
    Quit
}

#[derive(Clone)]
pub struct SceneRenderProperties {
    pub screen_size: Vec2,
    pub screen_center: Vec2,
    pub zoom: f32,

    pub background_color: Color,

    pub line_thickness: f32,
    pub boundary_outline: Option<Color>,
    pub boundary_fill: Option<Color>,
    
    pub color_mode: ParticleColoringMode,

    pub render_normals: bool,
    pub normals_color: Color,
    
    pub render_grid_cells: bool,
    pub grid_cell_color: Color,

    pub render_sinks: bool,

    pub render_velocities: bool,
    pub velocity_scaling: f32,

    pub render_temp_sources: bool,
    pub temp_source_alpha: f32,

    pub dynamic_temp_color: bool,
    pub temp_color_min: f32,
    pub temp_color_max: f32,

    pub gradient_hue_low: f32,
    pub gradient_hue_high: f32,

    pub selected_particle_color: Color,
}

impl SceneRenderProperties {
    pub fn default() -> Self {
        Self { 
            screen_size: vec2(1600., 900.),
            screen_center: vec2(-0.1, 0.0),
            zoom: 1000., 
            background_color: WHITE,
            line_thickness: 2., 
            boundary_outline: Some(DARKGRAY), 
            boundary_fill: Some(GRAY), 
            normals_color: RED, 
            render_velocities: false,
            render_normals: false, 
            render_temp_sources: true, 
            render_sinks: true, 
            velocity_scaling: 0.005, 
            color_mode: ParticleColoringMode::Type, 
            gradient_hue_low: 0.7, 
            gradient_hue_high: 0.,
            temp_source_alpha: 0.2,
            temp_color_min: 0.,
            temp_color_max: 500.,
            dynamic_temp_color: true,
            render_grid_cells: false,
            grid_cell_color: Color { a: 0.3, ..PINK },
            selected_particle_color: PURPLE
        }
    }

    pub fn eval_color_gradient(&self, val: f32, min: f32, max: f32, alpha: f32) -> Color {
        let frac = if max - min > 1e-3 {
            (val - min) / (max - min)
        }
        else {
            0.5
        }.clamp(0., 1.);
        let mut pure_color = hsl_to_rgb(self.gradient_hue_low + frac * (self.gradient_hue_high - self.gradient_hue_low), 1., 0.5);
        pure_color.a = alpha;
        pure_color
    }
}

pub trait Scene {
    fn setup(&mut self) {}
    fn destroy(&mut self) {}
    fn do_frame(&mut self,  man_ctx: &SceneControlCtx, circle_renderer: &mut InstancedCircleDrawStage) -> SceneControlCmd;
}

#[derive(Clone, PartialEq, Eq)]
pub enum MouseInteractionMode {
    None,
    Force,
    Temperature,
    Sink,
    Emitter
}

#[derive(Clone)]
pub struct MouseInteraction {
    pub mode: MouseInteractionMode,
    pub effect_radius: (f32, Range<f32>),
    pub force_coef: (f32, Range<f32>),
    pub target_temp: (f32, Range<f32>),
    pub diffusion: (f32, Range<f32>),
    pub emission_rate: (f32, Range<f32>),
    pub target_velocity: Vec2,
    pub last_world_pos: Option<Vec2>,
}

impl MouseInteraction {
    pub fn default() -> Self {
        MouseInteraction { 
            mode: MouseInteractionMode::None, 
            target_velocity: Vec2::ZERO,
            effect_radius: (0.02, 0.003 .. 0.2),  
            force_coef: (1., 1. .. 10.), 
            diffusion: (1., 0. .. 2.),
            target_temp: (293., 0. .. 1000.), 
            emission_rate: (1000., 0. .. 10000.),
            last_world_pos: None,
        }
    }

    pub fn compute_drag_force(&mut self, cam: &Camera2D) {
        let cpos: Vec2 = cam.screen_to_world(mouse_position().into());
        if !is_mouse_button_down(MouseButton::Left) {
            self.target_velocity *= 0.;
        }
        if let Some(lpos) =  self.last_world_pos {
            self.target_velocity = 0.7 * self.target_velocity + 0.3 * (cpos - lpos) * self.force_coef.0;
        }
        self.last_world_pos = Some(cpos);
    }

    pub fn tick<K: SimKernels>(&self, sim: &mut FluidSim<K>, dt: f32) {
        if self.mode == MouseInteractionMode::None || !is_mouse_button_down(MouseButton::Left) {
            return;
        }
        if let Some(pos) = self.last_world_pos {
            let affected = sim.range_query(pos, self.effect_radius.0).map(|p| p.id).collect_vec();

            // Will smooth out force/temp changes at circle boundary
            let smoother = |v: Vec2| {
                let u = (v - pos).length_squared() / self.effect_radius.0.powi(2);
                1. - u.powi(4)
            };

            match self.mode {
                MouseInteractionMode::Force => {
                    for id in affected {
                        let p = sim.grid_mut().particle_mut(id).unwrap();
                        p.v += self.force_coef.0 * (self.target_velocity - p.v * dt) * smoother(p.x);
                    }
                }
                MouseInteractionMode::Temperature => {
                    for id in affected {
                        let p = sim.grid_mut().particle_mut(id).unwrap();
                        p.t += smoother(p.x) * self.diffusion.0 * (self.target_temp.0 - p.t) * dt; 
                    }
                }
                MouseInteractionMode::Sink => {
                    for id in affected {
                        sim.delete_particle(id);
                    }
                }
                MouseInteractionMode::Emitter => {
                    // Find most common particle in region, and spawn more of it
                    let mut parts: HashMap<FluidId, (u32, f32, f32, Vec2)> = HashMap::default();
                    for id in affected {
                        let p = sim.grid().particle(id).unwrap();
                        let data = parts.entry(p.fluid_id).or_default();
                        data.0 += 1;
                        data.1 += p.m;
                        data.2 += p.t;
                        data.3 += p.v;
                    }
                    if let Some((&id, &(num, m, t, v))) = parts.iter().max_by_key(|d| d.0) {
                        // Spawn with properties of most common fluid in region
                        let emitter = FluidEmitter {
                            shape: Box::new(shape::Circle { 
                                center: pos,
                                radius: self.effect_radius.0,
                            }),
                            rate: self.emission_rate.0,
                            spawn_mass: m / num as f32,
                            spawn_velocity: v / num as f32,
                            spawn_temp: t / num as f32,
                            spawn_fluid_id: id
                        };
                        emitter.tick(sim, dt);
                    }
                }
                _ => {}
            };
        }
    }
}

#[derive(Clone)]
pub struct GenericScene<K : SimKernels = DefaultSimKernels> {
    pub name: String,

    pub fluid_types: Vec<FluidType>,
    pub boundaries: Vec<Box<dyn Shape>>,
    pub emitters: Vec<FluidEmitter>,
    pub fluid_sim: FluidSim<K>,
    pub custom_sim_setup: fn(&mut FluidSim<K>) -> (), 
    
    pub fps_limit: Option<f32>,
    pub timestep: f32,
    pub min_log_ts: f32,
    pub max_log_ts: f32,
    pub substeps: u32,
    pub do_step: bool,
    pub is_paused: bool,
    pub last_mouse_pos: Option<Vec2>,

    pub mouse_interaction: MouseInteraction,
    pub render_properties: SceneRenderProperties
}

pub const DEFAULT_AIR_ID: FluidId = 0;
pub const DEFAULT_WATER_ID: FluidId = 1;
pub const DEFAULT_OIL_ID: FluidId = 2;

impl<K: SimKernels> GenericScene<K> {
    pub fn default_air() -> Self {
        Self {
            name: "Default".to_owned(),
            fluid_types: vec![
                FluidType::AIR,
                FluidType::WATER,
                FluidType::OIL,
            ],
            boundaries: AABB { aa: vec2(-0.5, -0.3), bb: vec2(0.5, 0.3) }
                .sides().map(|((s, e), n)| HalfPlane { 
                    start: s,
                    end: e,
                    normal: -n
                 }.into_shape_box())
                 .collect_vec(),
            emitters: Vec::new(),
            fluid_sim: FluidSim::new(0.012, vec2(0., -9.8), 0.001, Some(AirGenerationParams {
                generation_threshold: 6000.,
                deletion_threshold_p: 10.,
                deletion_threshold_s: 0.5,
                deletion_threshold_rho: 10.,
                spawn_position_offset: 0.00003,
                air_mass: 100. * 0.003 * 0.003,
                air_buyoyancy: 1.,
                air_temp: 293.,
                air_id: 0
            })).with_sim_limits(AABB { aa: vec2(-2., -2.), bb: vec2(2., 2.) }),
            custom_sim_setup: |_| {},
            fps_limit: None,
            timestep: 0.001,
            min_log_ts: -5.,
            max_log_ts: -2.,
            substeps: 1,
            do_step: false,
            is_paused: true,
            last_mouse_pos: None,
            mouse_interaction: MouseInteraction::default(),
            render_properties: SceneRenderProperties::default()
        }
    }

    pub fn default_no_air() -> Self {
        Self {
            fluid_sim: FluidSim::new(0.012, vec2(0., -9.8), 0.001, None),
            ..GenericScene::default_air()
        }
    }
}

pub struct SceneControlCtx {
    pub scene_index: usize,
    pub scene_count: usize,
}

impl<K: SimKernels> Scene for GenericScene<K> {
    fn setup(&mut self) {
        request_new_screen_size(self.render_properties.screen_size.x, self.render_properties.screen_size.y);
        for t in self.fluid_types.drain(..) {
            self.fluid_sim.add_fluid_type(t);
        }
        self.fluid_sim.add_boundaries(self.boundaries.drain(..));
        (self.custom_sim_setup)(&mut self.fluid_sim);
    }

    fn do_frame(&mut self, man_ctx: &SceneControlCtx, circle_renderer: &mut InstancedCircleDrawStage) -> SceneControlCmd {
        let frame_start = std::time::Instant::now();

        /* SIMULATION */

        // Timestep and tick particle emitters
        if !self.is_paused || self.do_step {
            self.do_step = false;
            let dt = self.timestep / self.substeps as f32;

            for _ in 0..self.substeps {
                self.fluid_sim.tick(dt);
                self.mouse_interaction.tick(&mut self.fluid_sim, dt);
                for e in self.emitters.iter() {
                    e.tick(&mut self.fluid_sim, dt);
                }
            }
        }
        
        /* RENDER */

        let rend = &mut self.render_properties;
        clear_background(rend.background_color);

        // Setup camera & mouse
        let sw = screen_width();
        let sh = screen_height();

        let display_rect = Rect {
            x: rend.screen_center.x - 0.5 * sw / rend.zoom,
            y: rend.screen_center.y + 0.5 * sh / rend.zoom,
            w: sw / rend.zoom,
            h: -sh / rend.zoom
        };

        let camera = Camera2D::from_display_rect(display_rect);
        set_camera(&camera);
        
        let mouse_pos = mouse_position().into();
        let mouse_pos_world = camera.screen_to_world(mouse_pos);
        self.mouse_interaction.compute_drag_force(&camera);
        
        // Compute desired line thickness (should be invariant of zoom, about 2 pixels)
        rend.line_thickness = 2. / rend.zoom;

        // Render grid cells
        if rend.render_grid_cells {
            self.fluid_sim.draw_grid_cells(0.001, None, Some(rend.grid_cell_color));
        }

        let normal_color = rend.render_normals.then_some(rend.normals_color);

        // Render boundaries 
        for b in self.fluid_sim.boundaries() {
            b.draw(
                rend.line_thickness, 
                rend.boundary_outline, 
                rend.boundary_fill, 
                normal_color
            );
        }

        // Render emitters
        for e in self.emitters.iter() {
            let fill_color = self.fluid_sim.fluid_type(e.spawn_fluid_id).color;
            let outline_color = Color::from_vec(fill_color.to_vec() * 0.7);
            e.shape.draw(
                rend.line_thickness, 
                Some(outline_color), 
                Some(fill_color), 
                normal_color
            )
        }

        // Compute min/max temps and quantity for rendering temperature sources
        let mut min_temp = f32::INFINITY;
        let mut max_temp = f32::NEG_INFINITY;

        let mut min_q = f32::INFINITY;
        let mut max_q = f32::NEG_INFINITY;

        for b in self.fluid_sim.interacting_bodies() {
            if let Some(t) = b.temperature() {
                min_temp = min_temp.min(t);
                max_temp = max_temp.max(t);
            }
        }

        let get_q = |p: &FluidParticle| -> f32 {
            match rend.color_mode {
                ParticleColoringMode::Temperature => p.t,
                ParticleColoringMode::SurfaceGradient => p.cs_grad.length_squared(),
                ParticleColoringMode::InterfaceGradient => p.ci_grad.length_squared(),
                ParticleColoringMode::Density => p.rho,
                ParticleColoringMode::Compression => (1. + p.rho / p.rho0).log10(),
                ParticleColoringMode::Type => 0.
            }
        };

        for p in self.fluid_sim.particles() {
            min_temp = min_temp.min(p.t);
            max_temp = max_temp.max(p.t);
            let q = get_q(p);
            min_q = min_q.min(q);
            max_q = max_q.max(q);
        }

        if !rend.dynamic_temp_color {
            min_temp = rend.temp_color_min;
            max_temp = rend.temp_color_max;
        }
        
        std::mem::swap(&mut min_temp, &mut rend.temp_color_min);
        std::mem::swap(&mut max_temp, &mut rend.temp_color_max);

        // Render interacting bodies
        for b in self.fluid_sim.interacting_bodies() {
            b.render(rend);
        }

        let affected : HashSet<ParticleId> = if self.mouse_interaction.mode == MouseInteractionMode::None {
            HashSet::new()
        } else {
            self.fluid_sim
                .range_query(mouse_pos_world, self.mouse_interaction.effect_radius.0)
                .map(|p| p.id)
                .collect()
        };

        // Render particles 
        circle_renderer.clear();
        for p in self.fluid_sim.particles() {
            let r =  (p.m / p.rho).sqrt() / 2.;
            let color = if affected.contains(&p.id) {
                rend.selected_particle_color
            } else { 
                match rend.color_mode {
                    ParticleColoringMode::Type => self.fluid_sim.fluid_type(p.fluid_id).color,
                    ParticleColoringMode::Temperature => {
                        rend.eval_color_gradient(p.t, min_q, max_q, 1.)
                    },
                    _ => rend.eval_color_gradient(get_q(p), min_q, max_q, 1.)
                }
            };
            circle_renderer.push_circle(p.x, r, color);
        }
        circle_renderer.draw(&camera, &mut unsafe { get_internal_gl() });

        std::mem::swap(&mut min_temp, &mut rend.temp_color_min);
        std::mem::swap(&mut max_temp, &mut rend.temp_color_max);

        // Render particle velocities, if enabled
        if rend.render_velocities {
            for p in self.fluid_sim.particles() {
                let vel_end = p.x + p.v * rend.velocity_scaling;
                draw_line(p.x.x, p.x.y, vel_end.x, vel_end.y, rend.line_thickness, RED);
            }
        }

        /* UI */

        let mut log_timestep = self.timestep.log10();
        let mut substeps = self.substeps as f32;
        let mut scene_command = SceneControlCmd::None;
        
        widgets::Window::new(hash!(), vec2(10., 10.), vec2(350., 850.))
        .label(&format!("Scene {}/{}: {}", man_ctx.scene_index + 1, man_ctx.scene_count, &self.name))
        .ui(&mut *root_ui(), |ui| {   
            if let Some(limit) = self.fps_limit {
                ui.label(None, &format!("FPS: {:.0} (limit {:.0})", 1. / get_frame_time(), limit));
            }      
            else {
                ui.label(None, &format!("FPS: {:.0}", 1. / get_frame_time()));
            }
            ui.label(None, &format!("# particles: {}", self.fluid_sim.num_particles()));
            ui.label(None, &format!("min temp: {:.3} K", min_temp));
            ui.label(None, &format!("max temp: {:.3} K", max_temp));
            
            ui.separator();
            ui.separator();

            ui.label(None, &format!("1/tstep: {:.0} s^-1", 1. / self.timestep));
            ui.label(None, &format!("tstep: {:.7} s", self.timestep));
            ui.label(None, &format!("dt (tstep / substeps): {:.7} s", self.timestep / self.substeps as f32));

            ui.slider(hash!(), "log(tstep)", self.min_log_ts .. self.max_log_ts, &mut log_timestep);
            ui.slider(hash!(), "substeps", 1. .. 20., &mut substeps);

            self.is_paused = self.is_paused ^ (
                ui.button(None, if self.is_paused { "resume (space)"} else { "pause (space)"}) ||
                is_key_pressed(KeyCode::Space)
            );
            if self.is_paused {
                ui.same_line(0.);
                self.do_step = ui.button(None, "step (s)") || is_key_pressed(KeyCode::S);
            }

            ui.separator();
            ui.separator();
            ui.label(None, "RENDERING");

            rend.color_mode = match ui.combo_box(hash!(), "color mode", &[
                "fluid type",
                "interface grad.",
                "surface grad.",
                "density",
                "compression",
                "temperature"
            ], None) {
                1 => ParticleColoringMode::InterfaceGradient,
                2 => ParticleColoringMode::SurfaceGradient,
                3 => ParticleColoringMode::Density,
                4 => ParticleColoringMode::Compression,
                5 => ParticleColoringMode::Temperature,
                _ => ParticleColoringMode::Type,
            };

            ui.checkbox(hash!(), "draw velocities", &mut rend.render_velocities);
            ui.checkbox(hash!(), "draw grid cells", &mut rend.render_grid_cells);
            ui.checkbox(hash!(), "draw sinks", &mut rend.render_sinks);
            ui.checkbox(hash!(), "draw heat sources", &mut rend.render_temp_sources);
            ui.checkbox(hash!(), "draw normals", &mut rend.render_normals);

            ui.separator();
            ui.separator();
            ui.label(None, "GLOBAL SIM. CONSTANTS");

            let mut t = f32::atan2(self.fluid_sim.gravity.y, self.fluid_sim.gravity.x).to_degrees(); 
            let mut r = self.fluid_sim.gravity.length();
            ui.slider(hash!(), "gravity", 0. .. 30., &mut r);
            ui.slider(hash!(), "(angle)", -180. .. 180., &mut t);
            self.fluid_sim.gravity = r * Vec2::from_angle(t.to_radians());

            let mut heat_log = self.fluid_sim.heat_diffusion.log10();
            ui.label(None, &format!("heat diffusion: {}", self.fluid_sim.heat_diffusion));
            ui.slider(hash!(), "log(diff.)", -5. .. 0., &mut heat_log);
            self.fluid_sim.heat_diffusion = (10_f32).powf(heat_log);

            ui.separator();
            ui.separator();
            ui.label(None, "MOUSE INTERACTION (LEFT MOUSE + DRAG)");

            let mi = &mut self.mouse_interaction;
            mi.mode = match ui.combo_box(hash!(), "mode", &[
                "apply force",
                "apply heat/cold",
                "fluid source",
                "fluid sink",
                "none"
            ], None) {
                0 => MouseInteractionMode::Force,
                1 => MouseInteractionMode::Temperature,
                2 => MouseInteractionMode::Emitter,
                3 => MouseInteractionMode::Sink,
                _ => MouseInteractionMode::None,
            };

            if mi.mode != MouseInteractionMode::None {
                ui.slider(hash!(), "radius", mi.effect_radius.1.clone(), &mut mi.effect_radius.0);
            }

            match mi.mode {
                MouseInteractionMode::Force => {
                    ui.slider(hash!(), "multiplier", mi.force_coef.1.clone(), &mut mi.force_coef.0);
                },
                MouseInteractionMode::Emitter => {
                    ui.slider(hash!(), "rate", mi.emission_rate.1.clone(), &mut mi.emission_rate.0);
                }
                MouseInteractionMode::Temperature => {
                    ui.slider(hash!(), "temperature", mi.target_temp.1.clone(), &mut mi.target_temp.0);
                    ui.slider(hash!(), "diffusion", mi.diffusion.1.clone(), &mut mi.diffusion.0);
                }
                _ => {}
            };

            ui.separator();
            ui.separator();
            ui.label(None, "SCENES");

            if ui.button(None, "reset scene (r)") || is_key_pressed(KeyCode::R) {
                scene_command = SceneControlCmd::Reset; 
            }
            if ui.button(None, "previous scene (<)") || is_key_pressed(KeyCode::Left) {
                scene_command = SceneControlCmd::LoadPrevious; 
            }
            if ui.button(None, "next scene (>)") || is_key_pressed(KeyCode::Right) {
                scene_command = SceneControlCmd::LoadNext; 
            }
            if ui.button(None, "quit (esc)") || is_key_pressed(KeyCode::Escape) {
                scene_command = SceneControlCmd::Quit;
            }

            ui.separator();
            ui.separator();
            ui.label(None, "OTHER CONTROLS");
            ui.label(None, "drag view (right mouse + drag)");
            ui.label(None, "zoom (scroolwheel)");
        });

        self.timestep = (10_f32).powf(log_timestep);
        self.substeps = substeps.floor() as u32;

        // Dragging and zooming

        if !root_ui().is_mouse_captured() && is_mouse_button_pressed(MouseButton::Right) {
            self.last_mouse_pos = Some(mouse_pos);
        }
        else if is_mouse_button_down(MouseButton::Right) {
            if let Some(lpos) = self.last_mouse_pos {
                let delta = mouse_pos_world - camera.screen_to_world(lpos);
                rend.screen_center -= delta;
                self.last_mouse_pos = Some(mouse_pos);
            }
        }
        else if is_mouse_button_released(MouseButton::Right) {
            self.last_mouse_pos = None;
        }

        if mouse_wheel().1 > 0. {
            rend.zoom *= 1.1;
        }
        else if mouse_wheel().1 < 0. {
            rend.zoom *= 0.9;   
        }

        set_default_camera();
        draw_text(&format!("({:.3}, {:.3})", mouse_pos_world.x, mouse_pos_world.y), mouse_pos.x, mouse_pos.y, 10., BLACK);

        if let Some(limit) = self.fps_limit {
            let remaining = Duration::from_secs_f32(1./limit).saturating_sub(frame_start.elapsed());
            spin_sleep::sleep(remaining);
        }

        scene_command
    }
}

#[derive(Default)]
pub struct SceneCollection {
    scenes: Vec<Box<dyn Fn() -> Box<dyn Scene>>>
}

impl SceneCollection {
    pub fn add_scene<S: Scene + 'static>(&mut self, generator: impl Fn() -> S + 'static) {
        self.scenes.push(Box::new(move || Box::new(generator())))
    }

    pub fn add_scene_boxed<S: Scene>(&mut self, generator: impl Fn() -> Box<dyn Scene> + 'static) {
        self.scenes.push(Box::new(generator));
    }

    pub async fn mainloop(&mut self) {
        let gl = unsafe { get_internal_gl() };
        let mut circle_drawer = InstancedCircleDrawStage::new(gl.quad_context, 1 << 20);
        drop(gl);

        if self.scenes.is_empty() {
            return;
        }
        
        let mut ctrl = SceneControlCtx { scene_index : 0, scene_count : self.scenes.len() };
        let mut scene = self.scenes[0]();
        scene.setup();

        loop {
            if let Some(to_load) = match scene.do_frame(&ctrl, &mut circle_drawer) {
                SceneControlCmd::None => None, 
                SceneControlCmd::Reset => Some(ctrl.scene_index),
                SceneControlCmd::LoadNext => Some((ctrl.scene_index + 1) % ctrl.scene_count),
                SceneControlCmd::LoadPrevious => Some((ctrl.scene_index + ctrl.scene_count - 1) % ctrl.scene_count),
                SceneControlCmd::Quit => break
            } {
                ctrl.scene_index = to_load;
                scene.destroy();
                scene = self.scenes[to_load]();
                scene.setup();
            }

            next_frame().await;
        }
    }
}