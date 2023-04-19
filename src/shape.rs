use std::{f32::consts::PI, fmt::Debug};

use itertools::Itertools;
use macroquad::{prelude::*, rand::gen_range};

pub struct Collision {
    pub normal: Vec2,
    pub penetration: f32,
}

pub trait Shape {
    fn aabb(&self) -> (Vec2, Vec2);
    fn check_circle_collisions(&self, p: Vec2, r: f32) -> Option<Collision>;
    fn boundary_intersects_aabb(&self, aa: Vec2, bb: Vec2) -> bool;
    fn draw(&self, thickness: f32, outline: Option<Color>, fill: Option<Color>, normals: Option<Color>);
    fn random_point(&self) -> Vec2;
}

// Allow Box<dyn Shape> objects to impl Shape funcs
impl<S : Shape + ?Sized> Shape for Box<S> {
    fn aabb(&self) -> (Vec2, Vec2) {
        self.as_ref().aabb()
    }

    fn boundary_intersects_aabb(&self, aa: Vec2, bb: Vec2) -> bool {
        self.as_ref().boundary_intersects_aabb(aa, bb)
    }

    fn check_circle_collisions(&self, p: Vec2, r: f32) -> Option<Collision> {
        self.as_ref().check_circle_collisions(p, r)
    }

    fn draw(&self, thickness: f32, outline: Option<Color>, fill: Option<Color>, normals: Option<Color>) {
        self.as_ref().draw(thickness, outline, fill, normals)
    }

    fn random_point(&self) -> Vec2 {
        self.as_ref().random_point()
    }
} 

/// Half-plane boundary object. Assumes elements will not go over start or end for culling purposes,
/// so make the segment large enough!  
#[derive(Clone, Debug)]
pub struct HalfPlane
{
    pub start: Vec2,
    pub end: Vec2,
    pub normal: Vec2,
}

impl Shape for HalfPlane {
    fn aabb(&self) -> (Vec2, Vec2) {
        (Vec2::min(self.start, self.end), Vec2::max(self.start, self.end))
    }

    fn check_circle_collisions(&self, p: Vec2, r: f32) -> Option<Collision> {
        let penetration = r - (p - self.start).dot(self.normal);
        (penetration > 0.).then_some(Collision {
            normal: self.normal,
            penetration
        })
    }

    fn boundary_intersects_aabb(&self, aa: Vec2, bb: Vec2) -> bool {
        let corners = &[aa, bb, vec2(aa.x, bb.y), vec2(bb.x, aa.y)];
        !corners.iter().map(|&c| (c - self.start).dot(self.normal).signum()).all_equal()
    }

    fn draw(&self, thickness: f32, outline: Option<Color>, _: Option<Color>, normals: Option<Color>) {
        if let Some(c) = outline {
            draw_line(self.start.x, self.start.y, self.end.x, self.end.y, thickness, c);
        }
        if let Some(c) = normals {
            let middle = (self.start + self.end) / 2.;
            let normal_point = middle + self.normal * 20. * thickness;
            draw_line(middle.x, middle.y, normal_point.x, normal_point.y, thickness, c);
        }
    }

    fn random_point(&self) -> Vec2 {
        self.start + gen_range::<f32>(0., 1.) * (self.end - self.start)
    }
}

#[derive(Clone, Debug)]
pub struct Circle {
    pub center: Vec2,
    pub radius: f32,
}

impl Shape for Circle {
    fn aabb(&self) -> (Vec2, Vec2) {
        (self.center - self.radius, self.center + self.radius)
    }

    fn check_circle_collisions(&self, p: Vec2, r: f32) -> Option<Collision> {
        let to_p = p - self.center;
        let r2 = to_p.length_squared();
        if r2 < 1e-6 {
            None
        } 
        else if r2 < r * r + 2.*r*self.radius + self.radius * self.radius {
            let len = r2.sqrt();
            Some(Collision {
                normal: to_p / len,
                penetration: r + self.radius - len
            })
        }
        else {
            None
        }
    }

    fn boundary_intersects_aabb(&self, aa: Vec2, bb: Vec2) -> bool {
        let circumcircle_radius = 0.5 * (bb - aa).max_element();
        let distance_to_center_squared = (self.center - 0.5 * (aa + bb)).length_squared();
        
        let closest_point = self.center.clamp(aa, bb);
        let distance_squared = (closest_point - self.center).length_squared();

        distance_squared < self.radius.powi(2) && distance_to_center_squared >= (self.radius - circumcircle_radius).max(0.).powi(2)
    }

    fn draw(&self, thickness: f32, outline: Option<Color>, fill: Option<Color>, _: Option<Color>) {
        if let Some(c) = fill {
            draw_poly(self.center.x, self.center.y, 100, self.radius, 0., c)
        }
        if let Some(c) = outline {
            draw_poly_lines(self.center.x, self.center.y, 100, self.radius, 0., thickness, c);
        }
    }

    fn random_point(&self) -> Vec2 {
        let theta : f32 = gen_range(0., 2.*PI);
        let r : f32 = gen_range(0., self.radius * self.radius).sqrt();
        self.center + r * Vec2::from_angle(theta)
    }
}

#[derive(Clone, Debug)]
pub struct ConvexPolygon {
    pub vertices : Vec<Vec2>,
    pub normals: Vec<Vec2>
}

impl ConvexPolygon {
    /// Create a convex polygon from a list of counterclockwise vertices. 
    pub fn new(vertices: impl IntoIterator<Item = Vec2>) -> ConvexPolygon {
        let vertices = vertices.into_iter().collect_vec();
        let mut normals : Vec<Vec2> = Vec::with_capacity(vertices.len());
        for i in 0..vertices.len() {
            let n = (vertices[(i+1) % vertices.len()] - vertices[i]).perp().normalize();
            normals.push(n);
        }
        ConvexPolygon {
            vertices,
            normals
        }
    }
}

impl Shape for ConvexPolygon {
    fn aabb(&self) -> (Vec2, Vec2) {
        self.vertices.iter().fold(
            (vec2(f32::INFINITY, f32::INFINITY), vec2(f32::NEG_INFINITY, f32::NEG_INFINITY)), 
            |a, &e| (Vec2::min(a.0, e), Vec2::max(a.1, e)) 
        )
    }

    fn check_circle_collisions(&self, p: Vec2, r: f32) -> Option<Collision> {
        let mut least_separation = Collision {
            penetration: f32::INFINITY,
            normal: Vec2::ZERO
        };

        for (v, &n) in self.vertices.iter().zip(self.normals.iter()) {
            let pen = r - (p - *v).dot(n);
            if pen <= 0. { return None; }
            if pen < least_separation.penetration {
                least_separation.penetration = pen;
                least_separation.normal = n;
            }
        }

        Some(least_separation)
    }

    fn boundary_intersects_aabb(&self, aa: Vec2, bb: Vec2) -> bool {
        let corners = &[aa, bb, vec2(aa.x, bb.y), vec2(bb.x, aa.y)];
        self.vertices
            .iter()
            .zip(self.normals.iter())
            .any(|(&v, &n)| !corners.iter().map(|&c| (c - v).dot(n).signum()).all_equal())
    }

    fn draw(&self, thickness: f32, outline: Option<Color>, fill: Option<Color>, normals: Option<Color>) {
        let sides = self.vertices.len();

        if let Some(color) = fill {
            let context = unsafe { get_internal_gl() }.quad_gl;

            let mut vertices = Vec::<Vertex>::with_capacity(sides + 2);
            let mut indices = Vec::<u16>::with_capacity(sides as usize * 3);
        
            let center_point = self.vertices.iter().fold(Vec2::ZERO, |a, &e| a + e) / sides as f32;
            vertices.push(Vertex::new(center_point.x, center_point.y, 0., 0., 0., color));

            for i in 0..sides+1 {
                let v = self.vertices[i];
                let n = self.normals[i];
                let vertex = Vertex::new(v.x, v.y, 0., n.x, n.y, color);
                vertices.push(vertex);
        
                if i != sides {
                    indices.extend_from_slice(&[0, i as u16 + 1, i as u16 + 2]);
                }
            }
        
            context.texture(None);
            context.draw_mode(DrawMode::Triangles);
            context.geometry(&vertices, &indices);
        }
        if let Some(color) = outline {
            for i in 0..sides {
                let v1 = self.vertices[i];
                let v2 = self.vertices[(i + 1) % sides];
                draw_line(v1.x, v1.y, v2.x, v2.y, thickness, color);
            }
        }
        if let Some(color) = normals {
            for i in 0..sides {
                let v1 = self.vertices[i];
                let v2 = self.vertices[(i + 1) % sides];
                let middle = (v1 + v2) / 2.;
                let normal_point = middle + self.normals[i] * 20. * thickness;
                draw_line(middle.x, middle.y, normal_point.x, normal_point.y, thickness, color);
            }
        }
    }

    // TODO: Make this less stupid! Pick triangle using weighted distribution, for example
    fn random_point(&self) -> Vec2 {
        let (aa, bb) = self.aabb();
        loop {
            let p = vec2(gen_range(aa.x, bb.x), gen_range(aa.y, bb.y));

            if self.vertices
                .iter()
                .zip(self.normals.iter())
                .any(|(&v, &n)| (p - v).dot(n) > 0.) {
                continue;
            }

            return p;
        }
    }
}