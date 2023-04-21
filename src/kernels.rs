use std::f32::consts;
use macroquad::math::Vec2;

use crate::fluid::FluidParticle;

pub trait SmoothableParticle {
    fn mass(&self) -> f32;
    fn position(&self) -> Vec2;
    fn density(&self) -> f32;
}

impl SmoothableParticle for FluidParticle {
    fn density(&self) -> f32 {
        self.rho
    }
    fn mass(&self) -> f32 {
        self.m
    }
    fn position(&self) -> Vec2 {
        self.x
    }
}

/// Trait implemented by types which behave as a vector space over the reals.
pub trait RealVec : 
    Default + Sized + 
    std::ops::Add<Self, Output = Self> + 
    std::ops::Sub<Self, Output = Self> +
    std::ops::Mul<f32, Output = Self> +
    std::ops::Div<f32, Output = Self> {
}

impl<T> RealVec for T where T:
    Default + Sized + 
    std::ops::Add<Self, Output = Self> + 
    std::ops::Sub<Self, Output = Self> +
    std::ops::Mul<f32, Output = Self> +
    std::ops::Div<f32, Output = Self> {
}

pub trait SmoothingKernel {
    const NAME: &'static str;

    const VALUE_NEEDS_SQRT: bool = true;
    const GRADIENT_NEEDS_SQRT: bool = true;
    const LAPLACIAN_NEEDS_SQRT: bool = true; 

    fn value_unchecked(r: Vec2, l: f32, h: f32) -> f32;
    fn gradient_unchecked(r: Vec2, l: f32, h: f32) -> Vec2;
    fn laplacian_unchecked(r: Vec2, l: f32, h: f32) -> f32;

    fn value(r: Vec2, h: f32) -> f32 {
        let r2 = r.length_squared();
        if r2 < h * h {
            Self::value_unchecked(r, if Self::VALUE_NEEDS_SQRT { r2.sqrt() } else { 0. }, h)
        } else { 
            0.
        }
    }

    fn gradient(r: Vec2, h: f32) -> Vec2 {
        let r2 = r.length_squared();
        if r2 < h * h {
            Self::gradient_unchecked(r, if Self::GRADIENT_NEEDS_SQRT { r2.sqrt() } else { 0. }, h)
        } else { 
            Vec2::ZERO
        }
    }

    fn laplacian(r: Vec2, h: f32) -> f32 {
        let r2 = r.length_squared();
        if r2 < h * h {
            Self::laplacian_unchecked(r, if Self::LAPLACIAN_NEEDS_SQRT { r2.sqrt() } else { 0. }, h)
        } else { 
            0.
        }
    }

    fn smooth<'a, V, P: 'a>(
        r: Vec2, 
        h: f32, 
        scalar: impl Fn(&P) -> f32, 
        particles: impl Iterator<Item = &'a P>, 
        smoother: impl Fn(Vec2, f32) -> V) -> V 
        
        where P: SmoothableParticle, V: RealVec {

        let mut sum = V::default();
        for p in particles {
            sum = sum + smoother(p.position() - r, h) * scalar(p) * p.mass() / p.density();
        }
        sum 
    }

    fn smooth_value<'a, P: SmoothableParticle + 'a>(
        r: Vec2, 
        h: f32, 
        scalar: impl Fn(&P) -> f32, 
        particles: impl Iterator<Item = &'a P>) -> f32 {

        Self::smooth(r, h, scalar, particles, Self::value)
    }

    fn smooth_gradient<'a, P: SmoothableParticle + 'a>(
        r: Vec2, 
        h: f32, 
        scalar: impl Fn(&P) -> f32, 
        particles: impl Iterator<Item = &'a P>) -> Vec2 {

        Self::smooth(r, h, scalar, particles, Self::gradient)
    }

    fn smooth_laplacian<'a, P: SmoothableParticle + 'a>(
        r: Vec2, 
        h: f32, 
        scalar: impl Fn(&P) -> f32, 
        particles: impl Iterator<Item = &'a P>) -> f32 {

        Self::smooth(r, h, scalar, particles, Self::laplacian)
    }
}

// Source for 2D constants:
// https://www.diva-portal.org/smash/get/diva2:573583/FULLTEXT01.pdf

/// General-purpose smoothing kernel used for most SPH calculations.
pub struct WPoly6;
impl WPoly6 {
    const A: f32 = 4. / consts::PI;
    const B: f32 = 24. / consts::PI;
    const C: f32 = 24. / consts::PI;
}
impl SmoothingKernel for WPoly6 {
    const NAME: &'static str = "WPoly6(r, h) = 4/(pi h^8) * (h^2 - r^2)^3";

    const VALUE_NEEDS_SQRT: bool = false;
    const GRADIENT_NEEDS_SQRT: bool = false;
    const LAPLACIAN_NEEDS_SQRT: bool = false; 

    fn value_unchecked(r: Vec2, l: f32, h: f32) -> f32 {
        let r2 = r.length_squared();
        let h2 = h * h;
        Self::A / h2.powi(4) * (h2 - r2).powi(3)
    }

    fn gradient_unchecked(r: Vec2, l: f32, h: f32) -> Vec2 {
        let r2 = r.length_squared();
        let h2 = h * h;
        -Self::B / h2.powi(4) * (h2 - r2).powi(2) * r
    }

    fn laplacian_unchecked(r: Vec2, l: f32, h: f32) -> f32 {
        let r2 = r.length_squared();
        let h2 = h * h;
        -Self::C / h2.powi(4) * (h2 - r2) * (3.*h2 - 7.*r2)
    }
}

/// Smoothing kernel with a middle "spike" used for pressure calculations.
pub struct WSpiky;
impl WSpiky {
    const A: f32 = 10. / consts::PI;
    const B: f32 = 30. / consts::PI;
    const C: f32 = 60. / consts::PI;
}

impl SmoothingKernel for WSpiky {
    const NAME: &'static str = "WSpiky(r, h) = 10/(pi h^5) * (h-r)^3";

    fn value_unchecked(r: Vec2, l: f32, h: f32) -> f32 {
        Self::A / h.powi(5) * (h - l).powi(3)
    }

    fn gradient_unchecked(r: Vec2, l: f32, h: f32) -> Vec2 {
        -Self::B / h.powi(5) * (h - l).powi(2) * r / l
    }

    fn laplacian_unchecked(r: Vec2, l: f32, h: f32) -> f32 {
        -Self::C / h.powi(5) * (h - l) * (h - 2.*l) / l
    }
}

/// Smoothing kernel with a well-behaved Laplacian used for viscosity calculations.
pub struct WViscosity;
impl WViscosity {
    const A: f32 = 10. / (3. * consts::PI);
    const B: f32 = 10. / consts::PI;
    const C: f32 = 20. / consts::PI;
}

impl SmoothingKernel for WViscosity {
    const NAME: &'static str = "WViscosity(r, h) = 10/(3pi h^2) (-r3 / (2h^3) + r^2/h^2 + h/(2r) - 1)";

    fn value_unchecked(r: Vec2, l: f32, h: f32) -> f32 {
        let r2 = l * l;
        let h2 = h * h;
        let term = - r2 * l / (2. * h2 * h) + r2 / h2 + h / (2. * l) - 1.;
        Self::A / h2 * term
    }

    fn gradient_unchecked(r: Vec2, l: f32, h: f32) -> Vec2 {
        let r2 = l * l;
        let h2 = h * h;
        let term = -1.5 * l / (h2 * h) + 2. / h2 - h / (2. * r2 * l);
        Self::B / h2 * r * term
    }

    fn laplacian_unchecked(r: Vec2, l: f32, h: f32) -> f32 {
        Self::C / h.powi(5) * (h - l)
    }
}