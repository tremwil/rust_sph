mod interacting_body;
mod kernels;
mod shape;
mod fluid;
mod fluid_sim;
mod instanced_circle_draw;
mod emitter;
mod scene;
mod prelude;
mod scenes;

#[macroquad::main("SPH Air-Fluid-Fluid Simulation")]
async fn main() {
    let mut scenes = scene::SceneCollection::default();
    scenes.add_scene(scenes::dam_break_air);
    scenes.add_scene(scenes::immiscible_same_density);
    scenes.add_scene(scenes::immiscible_different_densities);
    scenes.add_scene(scenes::two_fluid_fountain);
    scenes.add_scene(scenes::high_viscosity_liquid);
    scenes.add_scene(scenes::diffusion);
    scenes.add_scene(scenes::lava_lamp);
    scenes.add_scene(scenes::stress_test_1);
    scenes.add_scene(scenes::stress_test_2);
    scenes.mainloop().await;
}