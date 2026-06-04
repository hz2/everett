//! Gate matrices.
//!
//! A single-qubit gate is a `2x2` unitary; a two-qubit gate is a `4x4` unitary.
//! These types only *store* the matrix — the actual application to a state is
//! the kernel's job (see [`crate::kernel`]). Matrices are kept tiny and stored
//! in row-major order so the kernel can index them directly.

use crate::complex::Complex64;

/// A single-qubit gate: a `2x2` complex matrix in row-major order.
///
/// # Examples
///
/// ```
/// use everett::Gate1;
///
/// // the Hadamard gate is its own inverse, so H * H = I.
/// let h = Gate1::h();
/// assert!(h.compose(&h).is_identity(1e-12));
/// ```
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Gate1 {
    /// Row-major entries: `[m00, m01, m10, m11]`.
    pub m: [Complex64; 4],
}

/// A two-qubit gate: a `4x4` complex matrix in row-major order.
///
/// The basis ordering is `|q_a q_b>` with `q_a` the more-significant bit, so the
/// rows/columns run `|00>, |01>, |10>, |11>`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Gate2 {
    /// Row-major entries, 16 of them.
    pub m: [Complex64; 16],
}

// shorthand constructors for matrix entries.
const fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}
const ZERO: Complex64 = Complex64::ZERO;
const ONE: Complex64 = Complex64::ONE;

impl Gate1 {
    /// Builds a gate from four row-major entries.
    #[inline]
    #[must_use]
    pub const fn new(m00: Complex64, m01: Complex64, m10: Complex64, m11: Complex64) -> Self {
        Self {
            m: [m00, m01, m10, m11],
        }
    }

    /// The identity gate `I`.
    #[must_use]
    pub const fn id() -> Self {
        Self::new(ONE, ZERO, ZERO, ONE)
    }

    /// The Pauli-`X` (bit-flip) gate.
    #[must_use]
    pub const fn x() -> Self {
        Self::new(ZERO, ONE, ONE, ZERO)
    }

    /// The Pauli-`Y` gate.
    #[must_use]
    pub const fn y() -> Self {
        // [[0, -i], [i, 0]]
        Self::new(ZERO, c(0.0, -1.0), c(0.0, 1.0), ZERO)
    }

    /// The Pauli-`Z` (phase-flip) gate.
    #[must_use]
    pub const fn z() -> Self {
        Self::new(ONE, ZERO, ZERO, c(-1.0, 0.0))
    }

    /// The Hadamard gate `H`.
    #[must_use]
    pub fn h() -> Self {
        // (1/sqrt(2)) * [[1, 1], [1, -1]]
        let s = std::f64::consts::FRAC_1_SQRT_2;
        Self::new(c(s, 0.0), c(s, 0.0), c(s, 0.0), c(-s, 0.0))
    }

    /// The phase gate `S = diag(1, i)`.
    #[must_use]
    pub const fn s() -> Self {
        Self::new(ONE, ZERO, ZERO, Complex64::I)
    }

    /// The `T = diag(1, e^{i*pi/4})` gate.
    #[must_use]
    pub fn t() -> Self {
        Self::new(
            ONE,
            ZERO,
            ZERO,
            Complex64::expi(std::f64::consts::FRAC_PI_4),
        )
    }

    /// A rotation about the X axis by `theta`, `R_x(theta) = e^{-i*theta*X/2}`.
    #[must_use]
    pub fn rx(theta: f64) -> Self {
        // [[cos(t/2), -i sin(t/2)], [-i sin(t/2), cos(t/2)]]
        let (s, co) = (theta / 2.0).sin_cos();
        Self::new(c(co, 0.0), c(0.0, -s), c(0.0, -s), c(co, 0.0))
    }

    /// A rotation about the Y axis by `theta`, `R_y(theta) = e^{-i*theta*Y/2}`.
    #[must_use]
    pub fn ry(theta: f64) -> Self {
        // [[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]]
        let (s, co) = (theta / 2.0).sin_cos();
        Self::new(c(co, 0.0), c(-s, 0.0), c(s, 0.0), c(co, 0.0))
    }

    /// A rotation about the Z axis by `theta`, `R_z(theta) = e^{-i*theta*Z/2}`.
    #[must_use]
    pub fn rz(theta: f64) -> Self {
        // diag(e^{-i t/2}, e^{+i t/2})
        Self::new(
            Complex64::expi(-theta / 2.0),
            ZERO,
            ZERO,
            Complex64::expi(theta / 2.0),
        )
    }

    /// A relative phase gate `diag(1, e^{i*lambda})`.
    #[must_use]
    pub fn phase(lambda: f64) -> Self {
        Self::new(ONE, ZERO, ZERO, Complex64::expi(lambda))
    }

    /// Returns the conjugate transpose `U^dagger`, the inverse of a unitary.
    #[must_use]
    pub fn adjoint(&self) -> Self {
        // transpose and conjugate: (U^dagger)_{ij} = conj(U_{ji}).
        Self::new(
            self.m[0].conj(),
            self.m[2].conj(),
            self.m[1].conj(),
            self.m[3].conj(),
        )
    }

    /// Returns `self * other` (matrix product), the gate that applies `other`
    /// then `self`.
    #[must_use]
    pub fn compose(&self, other: &Self) -> Self {
        let a = &self.m;
        let b = &other.m;
        Self::new(
            a[0] * b[0] + a[1] * b[2],
            a[0] * b[1] + a[1] * b[3],
            a[2] * b[0] + a[3] * b[2],
            a[2] * b[1] + a[3] * b[3],
        )
    }

    /// Returns `true` if this gate is the identity up to `eps`.
    #[must_use]
    pub fn is_identity(&self, eps: f64) -> bool {
        let i = Self::id();
        (0..4).all(|k| (self.m[k] - i.m[k]).norm() <= eps)
    }

    /// Returns `true` if this gate is unitary up to `eps`, i.e. `U^dagger U = I`.
    #[must_use]
    pub fn is_unitary(&self, eps: f64) -> bool {
        self.adjoint().compose(self).is_identity(eps)
    }
}

impl Gate2 {
    /// Builds a gate from 16 row-major entries.
    #[inline]
    #[must_use]
    pub const fn new(m: [Complex64; 16]) -> Self {
        Self { m }
    }

    /// The controlled-NOT gate with operand `a` the control, `b` the target.
    ///
    /// Flips `b` when `a` is set: it maps `|10> <-> |11>` and fixes `|00>, |01>`.
    #[must_use]
    pub const fn cnot() -> Self {
        // rows/cols ordered |00>,|01>,|10>,|11>
        Self::new([
            ONE, ZERO, ZERO, ZERO, // |00> -> |00>
            ZERO, ONE, ZERO, ZERO, // |01> -> |01>
            ZERO, ZERO, ZERO, ONE, // |10> -> |11>
            ZERO, ZERO, ONE, ZERO, // |11> -> |10>
        ])
    }

    /// The controlled-Z gate. Symmetric in its operands; phases `|11>` by -1.
    #[must_use]
    pub const fn cz() -> Self {
        Self::new([
            ONE,
            ZERO,
            ZERO,
            ZERO, //
            ZERO,
            ONE,
            ZERO,
            ZERO, //
            ZERO,
            ZERO,
            ONE,
            ZERO, //
            ZERO,
            ZERO,
            ZERO,
            c(-1.0, 0.0),
        ])
    }

    /// The SWAP gate. Exchanges the two qubits: `|01> <-> |10>`.
    #[must_use]
    pub const fn swap() -> Self {
        Self::new([
            ONE, ZERO, ZERO, ZERO, //
            ZERO, ZERO, ONE, ZERO, // |01> -> |10>
            ZERO, ONE, ZERO, ZERO, // |10> -> |01>
            ZERO, ZERO, ZERO, ONE,
        ])
    }

    /// Lifts a single-qubit gate `u` into a controlled two-qubit gate, with
    /// operand `a` the control and `b` the target.
    ///
    /// The result is `diag(I_2, u)` in the `|q_a q_b>` basis: the bottom-right
    /// `2x2` block is `u`, applied only when the control is set.
    #[must_use]
    pub fn controlled(u: &Gate1) -> Self {
        Self::new([
            ONE, ZERO, ZERO, ZERO, //
            ZERO, ONE, ZERO, ZERO, //
            ZERO, ZERO, u.m[0], u.m[1], //
            ZERO, ZERO, u.m[2], u.m[3],
        ])
    }

    /// Returns the conjugate transpose `U^dagger`.
    #[must_use]
    pub fn adjoint(&self) -> Self {
        let mut out = [Complex64::ZERO; 16];
        for row in 0..4 {
            for col in 0..4 {
                // (U^dagger)_{row,col} = conj(U_{col,row})
                out[row * 4 + col] = self.m[col * 4 + row].conj();
            }
        }
        Self::new(out)
    }

    /// Returns `true` if this gate is unitary up to `eps`.
    #[must_use]
    pub fn is_unitary(&self, eps: f64) -> bool {
        let prod = self.adjoint().mul(self);
        for row in 0..4 {
            for col in 0..4 {
                let expected = if row == col { ONE } else { ZERO };
                if (prod.m[row * 4 + col] - expected).norm() > eps {
                    return false;
                }
            }
        }
        true
    }

    // internal 4x4 matrix product, used only to check unitarity.
    fn mul(&self, other: &Self) -> Self {
        let mut out = [Complex64::ZERO; 16];
        for row in 0..4 {
            for col in 0..4 {
                let mut acc = Complex64::ZERO;
                for k in 0..4 {
                    acc += self.m[row * 4 + k] * other.m[k * 4 + col];
                }
                out[row * 4 + col] = acc;
            }
        }
        Self::new(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paulis_are_unitary() {
        for g in [
            Gate1::x(),
            Gate1::y(),
            Gate1::z(),
            Gate1::h(),
            Gate1::s(),
            Gate1::t(),
        ] {
            assert!(g.is_unitary(1e-12));
        }
    }

    #[test]
    fn pauli_x_is_its_own_inverse() {
        assert!(Gate1::x().compose(&Gate1::x()).is_identity(1e-12));
    }

    #[test]
    fn s_squared_is_z() {
        let s2 = Gate1::s().compose(&Gate1::s());
        assert!(s2.compose(&Gate1::z().adjoint()).is_identity(1e-12));
    }

    #[test]
    fn t_to_the_fourth_is_z() {
        let t = Gate1::t();
        let t4 = t.compose(&t).compose(&t).compose(&t);
        assert!(t4.compose(&Gate1::z().adjoint()).is_identity(1e-12));
    }

    #[test]
    fn rz_pi_matches_z_up_to_global_phase() {
        // R_z(pi) = diag(e^{-i pi/2}, e^{i pi/2}) = -i * Z. equal up to phase.
        let rz = Gate1::rz(std::f64::consts::PI);
        let z = Gate1::z();
        // multiply rz by +i and compare to z.
        let adjusted = Gate1::new(
            rz.m[0] * Complex64::I,
            rz.m[1] * Complex64::I,
            rz.m[2] * Complex64::I,
            rz.m[3] * Complex64::I,
        );
        assert!(adjusted.compose(&z.adjoint()).is_identity(1e-12));
    }

    #[test]
    fn two_qubit_gates_are_unitary() {
        for g in [Gate2::cnot(), Gate2::cz(), Gate2::swap()] {
            assert!(g.is_unitary(1e-12));
        }
    }

    #[test]
    fn controlled_x_is_cnot() {
        assert!(Gate2::controlled(&Gate1::x()).is_unitary(1e-12));
        assert_eq!(Gate2::controlled(&Gate1::x()), Gate2::cnot());
    }

    #[test]
    fn controlled_z_matches_cz() {
        assert_eq!(Gate2::controlled(&Gate1::z()), Gate2::cz());
    }
}
