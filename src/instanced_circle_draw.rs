// Rust implementation of this idea:
// https://www.johnaparker.com/blog/circle-graphics#instanced-drawing

use macroquad::prelude::*;
use miniquad::*;

pub struct InstancedCircleDrawStage {
    pub pipeline: Pipeline,
    pub bindings: Bindings,

    max_circle_count: usize,
    num_circles: usize,
    transforms: Vec<Mat4>,
    colors: Vec<Color>
}

impl InstancedCircleDrawStage {
    pub fn new(ctx: &mut Context, max_circle_count: usize) -> InstancedCircleDrawStage {
        #[rustfmt::skip]
        let vertices: [f32; 12] = [
            1.0,  1.0, 0.0,
            1.0, -1.0, 0.0,
           -1.0, -1.0, 0.0,
           -1.0,  1.0, 0.0,
        ];
        let vertex_buffer = Buffer::immutable(ctx, BufferType::VertexBuffer, &vertices);

        let indices: [u16; 6] = [0, 1, 3, 1, 2, 3];
        let index_buffer = Buffer::immutable(ctx, BufferType::IndexBuffer, &indices);

        let transform_buffer = Buffer::stream(ctx, BufferType::VertexBuffer, std::mem::size_of::<Mat4>() * max_circle_count);
        let color_buffer = Buffer::stream(ctx, BufferType::VertexBuffer, std::mem::size_of::<Color>() * max_circle_count);

        let bindings = Bindings {
            vertex_buffers: vec![vertex_buffer, transform_buffer, color_buffer],
            index_buffer,
            images: Vec::new(),
        };

        let shader =
            Shader::new(ctx, shader::VERTEX, shader::FRAGMENT, shader::meta()).unwrap();

        let mut pipeline_params = PipelineParams::default();
        pipeline_params.color_blend = Some(BlendState::new(
            Equation::Add,
            BlendFactor::Value(BlendValue::SourceAlpha),
            BlendFactor::OneMinusValue(BlendValue::SourceAlpha)
        ));

        let pipeline = Pipeline::with_params(
            ctx,
            &[
                BufferLayout::default(),
                BufferLayout {
                    step_func: VertexStep::PerInstance, 
                    step_rate: 1,
                    stride: 0
                },
                BufferLayout {
                    step_func: VertexStep::PerInstance, 
                    step_rate: 1,
                    stride: 0
                }
            ],
            &[
                VertexAttribute::with_buffer("aPos", VertexFormat::Float3, 0),
                VertexAttribute::with_buffer("transform", VertexFormat::Mat4, 1),
                VertexAttribute::with_buffer("color", VertexFormat::Float4, 2)
            ],
            shader,
            pipeline_params
        );

        InstancedCircleDrawStage { 
            pipeline, 
            bindings, 
            max_circle_count,
            num_circles: 0,
            transforms: vec![Mat4::ZERO; max_circle_count],
            colors: vec![BLACK; max_circle_count]
        }
    }

    pub fn clear(&mut self) {
        self.num_circles = 0;
    }

    pub fn push_circle(&mut self, position: Vec2, size: f32, color: Color) -> bool {
        if self.num_circles == self.max_circle_count {
            return false;
        }

        self.transforms[self.num_circles] = Mat4::from_cols_array(&[
            size, 0., 0., 0.,
            0., size, 0., 0.,
            0., 0., size, 0.,
            position.x, position.y, 0., 1.
        ]);
        self.colors[self.num_circles] = color;

        self.num_circles += 1;
        true
    }

    pub fn draw(&mut self, cam: &impl Camera, gl: &mut InternalGlContext) {
            // Ensure that macroquad's shapes are not going to be lost
            gl.flush();

            gl.quad_context.apply_pipeline(&self.pipeline);

            gl.quad_context.begin_default_pass(miniquad::PassAction::Nothing);
            gl.quad_context.apply_bindings(&self.bindings);

            gl.quad_context.apply_uniforms(&shader::Uniforms {
                vp: cam.matrix()
            });
            self.bindings.vertex_buffers[1].update(gl.quad_context, &self.transforms[0..self.num_circles]);
            self.bindings.vertex_buffers[2].update(gl.quad_context, &self.colors[0..self.num_circles]);

            assert!(gl.quad_context.features().instancing, "instancing not supported!");
            gl.quad_context.draw(0, 6, self.num_circles as i32);

            gl.quad_context.end_render_pass();
    }
}

pub mod shader {
    use macroquad::{miniquad::*, prelude::Mat4};

    pub const VERTEX: &str = r#"#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in mat4 transform;
layout (location = 5) in vec4 color;

uniform mat4 vp;
out vec4 circleColor;
out vec2 pos;

void main() {
    gl_Position = vp * transform * vec4(aPos, 1.0);
    circleColor = color;
    pos = aPos.xy;
}    
"#;

    pub const FRAGMENT: &str = r#"#version 330 core
in vec2 pos;
in vec4 circleColor;
out vec4 FragColor;

void main() {
    float rsq = dot(pos, pos);

    if (rsq > 1)
        discard;

    float t = clamp((rsq - 0.5) / 0.5, 0.0, 1.0);
    FragColor =  vec4(1, 1, 1, 1-t) * circleColor;
}
"#;

    pub fn meta() -> ShaderMeta {
        ShaderMeta {
            images: Vec::new(),
            uniforms: UniformBlockLayout {
                uniforms: vec![UniformDesc::new("vp", UniformType::Mat4)],
            },
        }
    }

    #[repr(C)]
    pub struct Uniforms {
        pub vp: Mat4,
    }
}