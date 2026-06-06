//! A minimal complex-number type for quantum amplitudes.
//!
//! [`Complex64`] is deliberately small and `#[repr(C)]` so its memory layout is
//! a predictable `[re, im]` pair. The simulator only needs the handful of
//! operations defined here, so we hand-roll them rather than depend on an
//! external crate.

use std::fmt;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

/// A complex number backed by two [`f64`] components.
///
/// # Examples
///
/// ```
/// use everett::Complex64;
///
/// let z = Complex64::new(3.0, 4.0);
/// assert_eq!(z.norm_sqr(), 25.0);
/// assert_eq!(z.conj(), Complex64::new(3.0, -4.0));
/// ```
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Default)]
pub struct Complex64 {
    /// Real component.
    pub re: f64,
    /// Imaginary component.
    pub im: f64,
}

impl Complex64 {
    /// The additive identity, `0 + 0i`.
    pub const ZERO: Self = Self { re: 0.0, im: 0.0 };
    /// The multiplicative identity, `1 + 0i`.
    pub const ONE: Self = Self { re: 1.0, im: 0.0 };
    /// The imaginary unit, `0 + 1i`.
    pub const I: Self = Self { re: 0.0, im: 1.0 };

    /// Constructs a complex number from real and imaginary parts.
    #[inline]
    #[must_use]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// Constructs a real-valued complex number `re + 0i`.
    #[inline]
    #[must_use]
    pub const fn real(re: f64) -> Self {
        Self { re, im: 0.0 }
    }

    /// Constructs a complex number from polar form, `r * e^{i*theta}`.
    #[inline]
    #[must_use]
    pub fn from_polar(r: f64, theta: f64) -> Self {
        Self::new(r * theta.cos(), r * theta.sin())
    }

    /// Constructs a unit-modulus phase `e^{i*theta}`.
    #[inline]
    #[must_use]
    pub fn expi(theta: f64) -> Self {
        Self::new(theta.cos(), theta.sin())
    }

    /// Returns the complex conjugate, `re - im*i`.
    #[inline]
    #[must_use]
    pub const fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }

    /// Returns the squared magnitude `re^2 + im^2`.
    ///
    /// Prefer this over [`Complex64::norm`] in hot paths: it avoids the square
    /// root, and amplitudes are most often combined as probabilities anyway.
    #[inline]
    #[must_use]
    pub fn norm_sqr(self) -> f64 {
        self.re.mul_add(self.re, self.im * self.im)
    }

    /// Returns the magnitude `sqrt(re^2 + im^2)`.
    #[inline]
    #[must_use]
    pub fn norm(self) -> f64 {
        self.re.hypot(self.im)
    }

    /// Returns the phase angle in radians, in `(-pi, pi]`.
    #[inline]
    #[must_use]
    pub fn arg(self) -> f64 {
        self.im.atan2(self.re)
    }
}

impl Add for Complex64 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl Sub for Complex64 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl Mul for Complex64 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        // (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        Self::new(
            self.re.mul_add(rhs.re, -(self.im * rhs.im)),
            self.re.mul_add(rhs.im, self.im * rhs.re),
        )
    }
}

impl Mul<f64> for Complex64 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f64) -> Self {
        Self::new(self.re * rhs, self.im * rhs)
    }
}

impl Div<f64> for Complex64 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: f64) -> Self {
        Self::new(self.re / rhs, self.im / rhs)
    }
}

impl Neg for Complex64 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.re, -self.im)
    }
}

impl AddAssign for Complex64 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for Complex64 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl MulAssign for Complex64 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl MulAssign<f64> for Complex64 {
    #[inline]
    fn mul_assign(&mut self, rhs: f64) {
        *self = *self * rhs;
    }
}

impl From<f64> for Complex64 {
    #[inline]
    fn from(re: f64) -> Self {
        Self::real(re)
    }
}

impl fmt::Debug for Complex64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // reuse the Display formatting; the debug pair is rarely what you want.
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for Complex64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.im < 0.0 {
            write!(f, "{}-{}i", self.re, -self.im)
        } else {
            write!(f, "{}+{}i", self.re, self.im)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multiplication_matches_definition() {
        let a = Complex64::new(1.0, 2.0);
        let b = Complex64::new(3.0, 4.0);
        // (1+2i)(3+4i) = 3 + 4i + 6i + 8i^2 = -5 + 10i
        assert_eq!(a * b, Complex64::new(-5.0, 10.0));
    }

    #[test]
    fn i_squared_is_negative_one() {
        assert_eq!(Complex64::I * Complex64::I, Complex64::new(-1.0, 0.0));
    }

    #[test]
    fn conj_times_self_is_norm_sqr() {
        let z = Complex64::new(-2.0, 5.0);
        let p = z * z.conj();
        assert!((p.re - z.norm_sqr()).abs() < 1e-15);
        assert!(p.im.abs() < 1e-15);
    }

    #[test]
    fn expi_is_unit_modulus() {
        for k in 0..16 {
            let theta = f64::from(k) * 0.4;
            assert!((Complex64::expi(theta).norm_sqr() - 1.0).abs() < 1e-15);
        }
    }

    #[test]
    fn repr_is_two_contiguous_f64() {
        // layout guarantee the kernel relies on.
        assert_eq!(
            std::mem::size_of::<Complex64>(),
            2 * std::mem::size_of::<f64>()
        );
        assert_eq!(
            std::mem::align_of::<Complex64>(),
            std::mem::align_of::<f64>()
        );
    }

    #[test]
    fn real_has_zero_imaginary_part() {
        let z = Complex64::real(3.0);
        assert_eq!(z, Complex64::new(3.0, 0.0));
    }

    #[test]
    fn from_f64_is_real() {
        let z: Complex64 = 5.0.into();
        assert_eq!(z, Complex64::real(5.0));
    }

    #[test]
    fn from_polar_matches_expi() {
        let z = Complex64::from_polar(2.0, std::f64::consts::FRAC_PI_2);
        assert!((z.re).abs() < 1e-15);
        assert!((z.im - 2.0).abs() < 1e-15);
    }

    #[test]
    fn norm_is_sqrt_norm_sqr() {
        let z = Complex64::new(3.0, 4.0);
        assert!((z.norm() - 5.0).abs() < 1e-15);
    }

    #[test]
    fn arg_of_i_is_half_pi() {
        assert!((Complex64::I.arg() - std::f64::consts::FRAC_PI_2).abs() < 1e-15);
    }

    #[test]
    fn subtraction_is_componentwise() {
        let a = Complex64::new(5.0, 3.0);
        let b = Complex64::new(2.0, 1.0);
        assert_eq!(a - b, Complex64::new(3.0, 2.0));
    }

    #[test]
    fn div_by_f64_scales_components() {
        let z = Complex64::new(4.0, 2.0);
        assert_eq!(z / 2.0, Complex64::new(2.0, 1.0));
    }

    #[test]
    fn neg_flips_both_components() {
        let z = Complex64::new(1.0, -2.0);
        assert_eq!(-z, Complex64::new(-1.0, 2.0));
    }

    #[test]
    fn add_assign_accumulates() {
        let mut z = Complex64::new(1.0, 2.0);
        z += Complex64::new(3.0, 4.0);
        assert_eq!(z, Complex64::new(4.0, 6.0));
    }

    #[test]
    fn sub_assign_decrements() {
        let mut z = Complex64::new(5.0, 6.0);
        z -= Complex64::new(1.0, 2.0);
        assert_eq!(z, Complex64::new(4.0, 4.0));
    }

    #[test]
    fn mul_assign_complex_updates_in_place() {
        let mut z = Complex64::new(1.0, 2.0);
        z *= Complex64::new(3.0, 4.0);
        assert_eq!(z, Complex64::new(1.0, 2.0) * Complex64::new(3.0, 4.0));
    }

    #[test]
    fn mul_assign_f64_scales_in_place() {
        let mut z = Complex64::new(2.0, 3.0);
        z *= 2.0_f64;
        assert_eq!(z, Complex64::new(4.0, 6.0));
    }

    #[test]
    fn display_positive_imaginary() {
        assert_eq!(format!("{}", Complex64::new(1.0, 2.0)), "1+2i");
    }

    #[test]
    fn display_negative_imaginary() {
        assert_eq!(format!("{}", Complex64::new(1.0, -2.0)), "1-2i");
    }

    #[test]
    fn debug_matches_display() {
        let z = Complex64::new(1.0, 2.0);
        assert_eq!(format!("{z:?}"), format!("{z}"));
    }
}
