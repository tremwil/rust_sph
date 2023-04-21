use crate::{shape::*, fluid::FluidParticle};
use crate::scene::{SceneRenderProperties, Scene, ParticleColoringMode};

/// Represents an object with a shape that interacts with neighboring particles in some way.
#[dyn_clonable::clonable]
pub trait InteractingStaticBody : Clone + Shape {
    /// Interact with the particle in some way. If `true` is returned, particle will be deleted. 
    /// `r` is the last computed particle radius. 
    fn interact(&mut self, p: &mut FluidParticle, r: f32, dt: f32) -> bool;
    
    fn render(&self, properties: &SceneRenderProperties);
    
    fn temperature(&self) -> Option<f32> { 
        None
    }
}

/// Represents a static temperature source. Will diffuse heat (or cold) to the particles it touches. 
#[derive(Clone)]
pub struct TempSource<S: Shape> {
    pub shape: S,
    pub temp: f32,
    pub diffusion_coef: f32,
}

impl<S: Clone + Shape> ShapeOwner for TempSource<S> {
    type ShapeType = S;
    fn shape(&self) -> &Self::ShapeType {
        &self.shape
    }
}

impl<S: Clone + Shape> InteractingStaticBody for TempSource<S> {
    fn interact(&mut self, p: &mut FluidParticle, r: f32, dt: f32) -> bool {
        if self.shape.check_circle_collisions(p.x, r).is_some() {
            p.t += dt * self.diffusion_coef * (self.temp - p.t); 
        }
        false
    }

    fn render(&self, properties: &SceneRenderProperties) {
        if properties.render_temp_sources {
            self.shape.draw(
                properties.line_thickness,
                properties.boundary_outline,
                Some(properties.eval_color_gradient(
                    self.temp, 
                    properties.temp_color_min, 
                    properties.temp_color_max,
                    properties.temp_source_alpha,
                )), None
            );
        }
    }

    fn temperature(&self) -> Option<f32> {
        Some(self.temp)
    }
}

/// Represents a particle sink. Will delete all particles that enter in contact with it.
#[derive(Clone, Debug)]
pub struct Sink<S: Shape> {
    pub shape: S
}

impl<S: Shape> ShapeOwner for Sink<S> {
    type ShapeType = S;
    fn shape(&self) -> &Self::ShapeType {
        &self.shape
    }
}

impl<S: Clone + Shape> InteractingStaticBody for Sink<S> {
    fn interact(&mut self, p: &mut FluidParticle, r: f32, dt: f32) -> bool {
        self.shape.check_circle_collisions(p.x, r).is_some()
    }

    fn render(&self,properties: &SceneRenderProperties) {
        if properties.render_sinks {
            self.draw(
                properties.line_thickness, 
                properties.boundary_outline, 
                None, 
                None
            )
        }
    }
}