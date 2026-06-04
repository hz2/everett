//! The quantum statevector.

use crate::complex::Complex64;
use crate::error::{Error, Result};

/// The pure state of an `n`-qubit system: a complex vector of `2^n` amplitudes.
///
/// The amplitude at index `j` is the coefficient of the computational basis
/// state `|j>`, with qubit `0` the least-significant bit of `j`. The vector is
/// kept normalized by construction; unitary gates preserve the norm and
/// measurement renormalizes the collapsed state.
///
/// # Examples
///
/// ```
/// use everett::State;
///
/// let s = State::zero(2);
/// assert_eq!(s.num_qubits(), 2);
/// // starts in |00> with probability 1.
/// assert!((s.probability(0b00) - 1.0).abs() < 1e-12);
/// ```
#[derive(Clone, PartialEq, Debug)]
pub struct State {
    // invariant: amps.len() == 1 << n. upheld by every constructor and never
    // violated by a gate (gates permute/mix amplitudes but never resize).
    amps: Vec<Complex64>,
    n: usize,
}

impl State {
    /// Creates the all-zeros state `|0...0>` of `n` qubits.
    ///
    /// # Panics
    ///
    /// Panics if `n` is large enough that `2^n` overflows `usize` (i.e. `n >= 64`
    /// on a 64-bit platform). Such a state could never be allocated anyway.
    #[must_use]
    pub fn zero(n: usize) -> Self {
        assert!(n < usize::BITS as usize, "2^{n} amplitudes overflows usize");
        let mut amps = vec![Complex64::ZERO; 1usize << n];
        amps[0] = Complex64::ONE;
        Self { amps, n }
    }

    /// Builds a state from an explicit amplitude vector.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DimensionMismatch`] if `amps.len()` is not a power of two.
    pub fn from_amplitudes(amps: Vec<Complex64>) -> Result<Self> {
        let len = amps.len();
        if !len.is_power_of_two() {
            return Err(Error::DimensionMismatch {
                len,
                // nearest power of two above, for a useful hint.
                expected: len.next_power_of_two(),
            });
        }
        let n = len.trailing_zeros() as usize;
        Ok(Self { amps, n })
    }

    /// The number of qubits, `n`.
    #[inline]
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.n
    }

    /// The number of amplitudes, `2^n`.
    #[inline]
    #[must_use]
    pub fn dim(&self) -> usize {
        self.amps.len()
    }

    /// The amplitudes as a read-only slice.
    #[inline]
    #[must_use]
    pub fn amplitudes(&self) -> &[Complex64] {
        &self.amps
    }

    /// The amplitudes as a mutable slice. For use by backends and the kernel.
    #[inline]
    #[must_use]
    pub fn amplitudes_mut(&mut self) -> &mut [Complex64] {
        &mut self.amps
    }

    /// The total probability `sum |amp|^2`, which should stay near `1`.
    #[must_use]
    pub fn norm_sqr(&self) -> f64 {
        self.amps.iter().map(|a| a.norm_sqr()).sum()
    }

    /// Rescales the state to unit norm.
    ///
    /// Unitary evolution preserves the norm in exact arithmetic, but rounding
    /// causes slow drift over deep circuits; call this periodically, and always
    /// after a measurement collapse.
    pub fn normalize(&mut self) {
        let norm = self.norm_sqr().sqrt();
        if norm > 0.0 {
            let inv = 1.0 / norm;
            for a in &mut self.amps {
                *a *= inv;
            }
        }
    }

    /// The probability of observing computational basis state `basis`.
    ///
    /// # Panics
    ///
    /// Panics if `basis >= 2^n`.
    #[must_use]
    pub fn probability(&self, basis: usize) -> f64 {
        assert!(basis < self.amps.len(), "basis state {basis} out of range");
        self.amps[basis].norm_sqr()
    }

    /// The overlap `<other|self>`, a complex number.
    ///
    /// # Panics
    ///
    /// Panics if the two states have different dimensions.
    #[must_use]
    pub fn overlap(&self, other: &Self) -> Complex64 {
        assert_eq!(self.n, other.n, "states must have equal qubit counts");
        // <other|self> = sum_j conj(other_j) * self_j
        let mut acc = Complex64::ZERO;
        for (a, b) in self.amps.iter().zip(&other.amps) {
            acc += b.conj() * *a;
        }
        acc
    }

    /// The fidelity `|<other|self>|^2` between two pure states, in `[0, 1]`.
    ///
    /// This is phase-agnostic: two states differing only by a global phase have
    /// fidelity `1`. It is the right tool for comparing simulator outputs.
    ///
    /// # Panics
    ///
    /// Panics if the two states have different dimensions.
    #[must_use]
    pub fn fidelity(&self, other: &Self) -> f64 {
        self.overlap(other).norm_sqr()
    }

    /// The Bloch-sphere vector `(x, y, z)` of a single qubit, tracing out the
    /// rest of the register.
    ///
    /// For a qubit in a pure, unentangled state this lands on the unit sphere;
    /// for an entangled qubit it lies strictly inside it.
    ///
    /// # Panics
    ///
    /// Panics if `qubit >= n`.
    #[must_use]
    pub fn bloch_vector(&self, qubit: usize) -> [f64; 3] {
        assert!(qubit < self.n, "qubit {qubit} out of range");
        let bit = 1usize << qubit;
        // reduced density matrix entries for the target qubit:
        //   rho = [[r00, r01], [conj(r01), r11]]
        // obtained by summing over all other qubits' indices.
        let mut r00 = 0.0;
        let mut r11 = 0.0;
        let mut r01 = Complex64::ZERO;
        for (j, amp) in self.amps.iter().enumerate() {
            if j & bit == 0 {
                r00 += amp.norm_sqr();
                // partner index with this qubit set; pairs (|..0..>, |..1..>).
                let partner = self.amps[j | bit];
                r01 += *amp * partner.conj();
            } else {
                r11 += amp.norm_sqr();
            }
        }
        // x = 2 Re(r01), y = -2 Im(r01)... using <X>,<Y>,<Z> conventions:
        //   <Z> = r00 - r11, <X> = 2 Re(r01), <Y> = 2 Im(r01).
        [2.0 * r01.re, 2.0 * r01.im, r00 - r11]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_state_is_normalized() {
        for n in 0..6 {
            assert!((State::zero(n).norm_sqr() - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn from_amplitudes_rejects_non_power_of_two() {
        let amps = vec![Complex64::ONE; 3];
        assert!(matches!(
            State::from_amplitudes(amps),
            Err(Error::DimensionMismatch { len: 3, .. })
        ));
    }

    #[test]
    fn fidelity_ignores_global_phase() {
        let a = State::zero(2);
        let mut b = a.clone();
        // multiply every amplitude by e^{i*0.7}: same physical state.
        for amp in b.amplitudes_mut() {
            *amp *= Complex64::expi(0.7);
        }
        assert!((a.fidelity(&b) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn bloch_vector_of_zero_state_points_up() {
        let s = State::zero(1);
        let [x, y, z] = s.bloch_vector(0);
        assert!(x.abs() < 1e-12 && y.abs() < 1e-12);
        assert!((z - 1.0).abs() < 1e-12);
    }

    #[test]
    fn normalize_fixes_drift() {
        let mut s =
            State::from_amplitudes(vec![Complex64::new(2.0, 0.0), Complex64::new(0.0, 2.0)])
                .unwrap();
        s.normalize();
        assert!((s.norm_sqr() - 1.0).abs() < 1e-12);
    }
}
