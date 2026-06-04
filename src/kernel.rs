//! The numerical kernel: in-place gate application over amplitude index-groups.
//!
//! A gate on `m` qubits never materializes a `2^n x 2^n` matrix. Instead we
//! iterate over the `2^(n-m)` groups of amplitudes that the gate mixes, and
//! apply the small `2^m x 2^m` matrix to each group. For a single-qubit gate
//! this is the classic "bit-insertion" loop.
//!
//! # The single-qubit index pair
//!
//! A gate on qubit `k` mixes, for each setting of the other `n-1` qubits, the
//! amplitude pair whose indices differ only in bit `k`. Enumerate those pairs
//! with a counter `i` running over `0..2^(n-1)`: insert a `0` bit at position
//! `k` of `i` to get the "bit-`k`-clear" index `i0`, then set bit `k` for its
//! partner `i1`:
//!
//! ```text
//! i0 = ((i >> k) << (k + 1)) | (i & ((1 << k) - 1))
//! i1 = i0 | (1 << k)
//! ```
//!
//! The low `k` bits of `i` pass through unchanged; the rest shift up by one to
//! make room for the inserted `0`. This makes `i -> i0` a bijection from
//! `0..2^(n-1)` onto the bit-`k`-clear indices, so across the loop every
//! amplitude is touched exactly once. Those bounds and the bijection are what
//! the Kani proofs in [`crate::kernel::proofs`] establish.

use crate::complex::Complex64;
use crate::gate::{Gate1, Gate2};

/// Computes the index pair `(i0, i1)` for the `i`-th step of a single-qubit
/// gate on qubit `k`.
///
/// `i0` is the index with bit `k` clear; `i1 = i0 | (1 << k)` is its partner.
/// See the module docs for the derivation.
#[inline]
#[must_use]
pub(crate) fn index_pair(k: usize, i: usize) -> (usize, usize) {
    let low_mask = (1usize << k) - 1;
    let i0 = ((i >> k) << (k + 1)) | (i & low_mask);
    let i1 = i0 | (1usize << k);
    (i0, i1)
}

/// Applies a single-qubit gate to qubit `k` of an `n`-qubit state.
///
/// `amps` must have length `2^n` and `k` must be `< n`.
pub fn apply_1q(amps: &mut [Complex64], k: usize, gate: &Gate1) {
    debug_assert!(amps.len().is_power_of_two());
    debug_assert!((1usize << k) < amps.len());
    let m = &gate.m;
    let pairs = amps.len() / 2;
    for i in 0..pairs {
        let (i0, i1) = index_pair(k, i);
        // a0, a1 are the amplitudes of the basis states differing in bit k.
        let a0 = amps[i0];
        let a1 = amps[i1];
        amps[i0] = m[0] * a0 + m[1] * a1;
        amps[i1] = m[2] * a0 + m[3] * a1;
    }
}

/// Computes the index quadruple for a two-qubit gate on qubits `a` and `b`.
///
/// Returns `[i00, i01, i10, i11]`, the indices whose `(a, b)` bits are
/// `(0,0), (0,1), (1,0), (1,1)` and which agree elsewhere. The `i`-th step
/// inserts `0` bits at both `a` and `b` of `i` (low position first), then sets
/// the two bits to enumerate the group.
#[inline]
#[must_use]
pub(crate) fn index_quad(a: usize, b: usize, i: usize) -> [usize; 4] {
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    // insert a 0 at the low target position.
    let low_mask = (1usize << lo) - 1;
    let t = ((i >> lo) << (lo + 1)) | (i & low_mask);
    // insert a 0 at the high target position (in the already-expanded index).
    let mid_mask = (1usize << hi) - 1;
    let base = ((t >> hi) << (hi + 1)) | (t & mid_mask);
    // base has both target bits clear; set them per operand a, b.
    let bit_a = 1usize << a;
    let bit_b = 1usize << b;
    [base, base | bit_b, base | bit_a, base | bit_a | bit_b]
}

/// Applies a two-qubit gate to qubits `a` (more significant in the gate basis)
/// and `b` of an `n`-qubit state.
///
/// `a` and `b` must be distinct and both `< n`; `amps` must have length `2^n`.
pub fn apply_2q(amps: &mut [Complex64], a: usize, b: usize, gate: &Gate2) {
    debug_assert!(a != b);
    debug_assert!(amps.len().is_power_of_two());
    let m = &gate.m;
    let groups = amps.len() / 4;
    for i in 0..groups {
        let idx = index_quad(a, b, i);
        let v = [amps[idx[0]], amps[idx[1]], amps[idx[2]], amps[idx[3]]];
        for (row, &out) in idx.iter().enumerate() {
            let base = row * 4;
            amps[out] =
                m[base] * v[0] + m[base + 1] * v[1] + m[base + 2] * v[2] + m[base + 3] * v[3];
        }
    }
}

/// Applies a single-qubit gate to `target`, but only on the subspace where
/// every qubit in `controls` is set. This is the general controlled gate.
///
/// `target` must not appear in `controls`; all indices must be `< n`.
pub fn apply_controlled_1q(
    amps: &mut [Complex64],
    controls: &[usize],
    target: usize,
    gate: &Gate1,
) {
    debug_assert!(!controls.contains(&target));
    let m = &gate.m;
    let mut control_mask = 0usize;
    for &c in controls {
        control_mask |= 1usize << c;
    }
    let pairs = amps.len() / 2;
    for i in 0..pairs {
        let (i0, i1) = index_pair(target, i);
        // apply only when all control bits are set in this pair's indices.
        // i0 and i1 agree on every bit except target, so testing i0 suffices.
        if i0 & control_mask == control_mask {
            let a0 = amps[i0];
            let a1 = amps[i1];
            amps[i0] = m[0] * a0 + m[1] * a1;
            amps[i1] = m[2] * a0 + m[3] * a1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_pair_inserts_zero_bit() {
        // k = 1: i0 must always have bit 1 clear; i1 sets it.
        for i in 0..8 {
            let (i0, i1) = index_pair(1, i);
            assert_eq!(i0 & (1 << 1), 0);
            assert_eq!(i1, i0 | (1 << 1));
        }
    }

    #[test]
    fn index_pair_covers_all_indices_once() {
        // for n = 3, k = 1, the pairs must partition 0..8 exactly.
        let mut seen = [false; 8];
        for i in 0..4 {
            let (i0, i1) = index_pair(1, i);
            assert!(!std::mem::replace(&mut seen[i0], true));
            assert!(!std::mem::replace(&mut seen[i1], true));
        }
        assert!(seen.iter().all(|&b| b));
    }

    #[test]
    fn x_gate_flips_single_qubit() {
        // |0> --X--> |1>
        let mut amps = vec![Complex64::ONE, Complex64::ZERO];
        apply_1q(&mut amps, 0, &Gate1::x());
        assert_eq!(amps, vec![Complex64::ZERO, Complex64::ONE]);
    }

    #[test]
    fn hadamard_then_hadamard_is_identity() {
        let mut amps = vec![Complex64::ONE, Complex64::ZERO];
        apply_1q(&mut amps, 0, &Gate1::h());
        apply_1q(&mut amps, 0, &Gate1::h());
        assert!((amps[0] - Complex64::ONE).norm() < 1e-12);
        assert!(amps[1].norm() < 1e-12);
    }

    #[test]
    fn index_quad_partitions_indices() {
        // n = 3, gate on qubits a=0, b=2: the 2 groups must cover 0..8 once.
        let mut seen = [false; 8];
        for i in 0..2 {
            for &idx in &index_quad(0, 2, i) {
                assert!(!std::mem::replace(&mut seen[idx], true));
            }
        }
        assert!(seen.iter().all(|&b| b));
    }

    #[test]
    fn cnot_flips_target_when_control_set() {
        // start in |10> (qubit1 set), control=1, target=0 -> |11>.
        // basis index for |q1 q0> = 2*q1 + q0, so |10> is index 2.
        let mut amps = vec![Complex64::ZERO; 4];
        amps[2] = Complex64::ONE;
        apply_2q(&mut amps, 1, 0, &Gate2::cnot());
        // |11> is index 3.
        assert!((amps[3] - Complex64::ONE).norm() < 1e-12);
    }

    #[test]
    fn controlled_x_matches_cnot() {
        // build a random-ish 2-qubit state and compare the two code paths.
        let mut a = vec![
            Complex64::new(0.5, 0.1),
            Complex64::new(0.2, -0.3),
            Complex64::new(-0.4, 0.2),
            Complex64::new(0.1, 0.6),
        ];
        let mut b = a.clone();
        apply_2q(&mut a, 1, 0, &Gate2::cnot());
        apply_controlled_1q(&mut b, &[1], 0, &Gate1::x());
        for (x, y) in a.iter().zip(&b) {
            assert!((*x - *y).norm() < 1e-12);
        }
    }
}
