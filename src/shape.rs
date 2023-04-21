use std::{f32::consts::PI, fmt::Debug};

use itertools::Itertools;
use macroquad::{prelude::*, rand::gen_range};

pub struct Collision {
    pub normal: Vec2,
    pub penetration: f32,
}

#[dyn_clonable::clonable]
pub trait Shape : Clone {
    /// Get a rectangular bounding box for this shape.
    fn aabb(&self) -> (Vec2, Vec2);

    /// Resolve collisions between this shape and a circle of position p and radius r.
    fn check_circle_collisions(&self, p: Vec2, r: f32) -> Option<Collision>;
    
    /// Check if this shape (or its boundary, if infinite) intersects a rectangular bounding box.
    fn intersects_aabb(&self, aa: Vec2, bb: Vec2) -> bool;
    
    /// Draw this shape with the given thickness, outline and fill color. Optionally draw surface normals as well.
    fn draw(&self, thickness: f32, outline: Option<Color>, fill: Option<Color>, normals: Option<Color>);
    
    /// Query a random point inside this shape. 
    fn random_point(&self) -> Vec2;

    fn into_shape_box(self) -> Box<dyn Shape> where Self: Sized + 'static {
        Box::new(self)
    }
}
/// Trait to make it easy to forward the Shape implementation of an object owning a Shape, without dynamic dispatch.]
pub trait ShapeOwner {
    type ShapeType : Shape + ?Sized;
    fn shape(&self) -> &Self::ShapeType;
}

impl<SO: ShapeOwner + Clone> Shape for SO {
    fn aabb(&self) -> (Vec2, Vec2) {
        self.shape().aabb()
    }

    fn intersects_aabb(&self, aa: Vec2, bb: Vec2) -> bool {
        self.shape().intersects_aabb(aa, bb)
    }

    fn check_circle_collisions(&self, p: Vec2, r: f32) -> Option<Collision> {
        self.shape().check_circle_collisions(p, r)
    }

    fn draw(&self, thickness: f32, outline: Option<Color>, fill: Option<Color>, normals: Option<Color>) {
        self.shape().draw(thickness, outline, fill, normals)
    }

    fn random_point(&self) -> Vec2 {
        self.shape().random_point()
    }
}

// Allow Box<dyn Shape> objects to impl Shape funcs
impl<S : Shape + ?Sized> ShapeOwner for Box<S> {
    type ShapeType = S;
    fn shape(&self) -> &Self::ShapeType {
        self.as_ref()
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

    fn intersects_aabb(&self, aa: Vec2, bb: Vec2) -> bool {
        let corners = &[aa, bb, vec2(aa.x, bb.y), vec2(bb.x, aa.y)];
        let it = corners.iter().map(|&c| (c - self.start).dot(self.normal));
        it.clone().any(|s| s.abs() < 1e-6) || !it.map(|s| s.signum()).all_equal()
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
pub struct AABB {
    pub aa: Vec2,
    pub bb: Vec2
}

impl AABB {
    pub const NORMALS : [Vec2; 4] = [
        vec2(0., -1.),
        vec2(1., 0.),
        vec2(0., 1.),
        vec2(-1., 0.)
    ];

    pub fn center(&self) -> Vec2 {
        (self.aa + self.bb) / 2.
    }

    pub fn size(&self) -> Vec2 {
        self.bb - self.aa
    }

    pub fn corners(&self) -> [Vec2; 4] {
        [self.aa, vec2(self.bb.x, self.aa.y), self.bb, vec2(self.aa.x, self.bb.y)]
    }

    pub fn corners_looping(&self) -> [Vec2; 5] {
        [self.aa, vec2(self.bb.x, self.aa.y), self.bb, vec2(self.aa.x, self.bb.y), self.aa]
    }

    pub fn sides(&self) -> impl Iterator<Item = ((Vec2, Vec2), Vec2)> {
        self.corners_looping().into_iter().tuple_windows().zip(Self::NORMALS)
    }
}

impl Shape for AABB {
    fn aabb(&self) -> (Vec2, Vec2) {
        (self.aa, self.bb)
    }
    
    fn intersects_aabb(&self, aa: Vec2, bb: Vec2) -> bool {
        (self.bb.min(bb) - self.aa.max(aa)).cmpge(Vec2::ZERO).all()
    }

    fn check_circle_collisions(&self, p: Vec2, r: f32) -> Option<Collision> {
        let closest_point = p.clamp(self.aa, self.bb);
        let distance_squared = (p - closest_point).length_squared();
        (distance_squared <= r * r).then_some({
            let normal : Vec2;
            let penetration : f32;

            // Center of circle is inside AABB, check every axis
            if p.cmpge(self.aa).all() && p.cmple(self.bb).all() {
                (penetration, normal) = [
                    p.y - self.aa.y + r,
                    self.bb.x - p.x + r,
                    self.bb.y - p.y + r,
                    p.x - self.aa.x + r 
                ].into_iter()
                    .zip(Self::NORMALS)
                    .min_by(|a, b| a.0.total_cmp(&b.0))
                    .unwrap();
            }
            else {
                let dist = distance_squared.sqrt();
                normal = (p - closest_point) / dist;
                penetration = r - dist;
            }
            Collision {
                normal,
                penetration
            }
        })
    }

    fn draw(&self, thickness: f32, outline: Option<Color>, fill: Option<Color>, normals: Option<Color>) {
        let sz = self.size();
        if let Some(c) = fill {
            draw_rectangle(self.aa.x, self.aa.y, sz.x, sz.y, c);
        }
        if let Some(c) = outline {
            draw_rectangle_lines(self.aa.x, self.aa.y, sz.x, sz.y, thickness, c)
        }
        if let Some(c) = normals {
            for ((s, e), n) in self.sides() {
                let middle = (s + e) / 2.;
                let normal_point = middle + n * 20. * thickness;
                draw_line(middle.x, middle.y, normal_point.x, normal_point.y, thickness, c);
            }
        }
    }

    fn random_point(&self) -> Vec2 {
        vec2(
            gen_range(self.aa.x, self.bb.x),
            gen_range(self.aa.y, self.bb.y)
        )
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

    fn intersects_aabb(&self, aa: Vec2, bb: Vec2) -> bool {
        let closest_point = self.center.clamp(aa, bb);
        let distance_squared = (closest_point - self.center).length_squared();
        distance_squared < self.radius.powi(2)
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
            let n = -(vertices[(i+1) % vertices.len()] - vertices[i]).perp().normalize();
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

    fn intersects_aabb(&self, aa: Vec2, bb: Vec2) -> bool {
        let aabb = AABB { aa, bb };
        let aabb_verts = aabb.corners();
        // SAT with polygon
        let sat_poly = self.vertices
            .iter()
            .zip(self.normals.iter())
            .all(|(&v, &n)| aabb_verts.iter().any(|&u| (u - v).dot(n) < 0.));
        // SAT with AABB
        sat_poly && aabb
            .sides()
            .all(|((s, e), n)| self.vertices.iter().any(|&v| (v - s).dot(n) < 0.))
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
                let v = self.vertices[i % self.vertices.len()];
                let n = self.normals[i % self.vertices.len()];
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