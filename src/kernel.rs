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
//! the Kani proofs in the `proofs` module (built only under `cfg(kani)`)
//! establish, which is what justifies the `get_unchecked` accesses in the apply
//! functions below.

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

/// Applies a single-qubit gate to qubit `k` of an `n`-qubit state.
///
/// `amps` must have length `2^n` (a power of two) and `k` must be `< n`.
///
/// On `x86_64` with AVX2 + FMA this dispatches to a vectorized path that
/// processes two amplitude pairs per step; everywhere else (and under Miri) it
/// uses the scalar path. Both compute the same result — the SIMD path is checked
/// against the scalar one by the kernel-vs-naive property test.
pub fn apply_1q(amps: &mut [Complex64], k: usize, gate: &Gate1) {
    debug_assert!(amps.len().is_power_of_two());
    debug_assert!(
        (1usize << k) < amps.len(),
        "qubit index k must be < log2(n)"
    );

    // the SIMD path needs stride >= 2 (k >= 1) so each AVX register holds two
    // complex amplitudes that share one matrix application. k == 0 (adjacent
    // pairs) stays scalar.
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    {
        if k >= 1 && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: guarded by the runtime feature detection just above.
            unsafe { apply_1q_avx2(amps, k, gate) };
            return;
        }
    }

    apply_1q_scalar(amps, k, gate);
}

/// The portable scalar single-qubit kernel. Always correct; the SIMD path is
/// validated against it.
fn apply_1q_scalar(amps: &mut [Complex64], k: usize, gate: &Gate1) {
    let n = amps.len();
    let m = &gate.m;
    let pairs = n / 2;
    for i in 0..pairs {
        let (i0, i1) = index_pair(k, i);
        debug_assert!(i0 < n && i1 < n);
        // SAFETY: with `n` a power of two and `(1 << k) < n`, `index_pair(k, i)`
        // for `i < n/2` yields `i0, i1 < n` and `i0 != i1` (proven in `proofs`
        // for all qubit counts up to the model-checking bound). both amplitudes
        // are read before either is written, so the two `get_unchecked_mut`
        // writes never alias.
        unsafe {
            let a0 = *amps.get_unchecked(i0);
            let a1 = *amps.get_unchecked(i1);
            *amps.get_unchecked_mut(i0) = m[0] * a0 + m[1] * a1;
            *amps.get_unchecked_mut(i1) = m[2] * a0 + m[3] * a1;
        }
    }
}

/// AVX2 + FMA single-qubit kernel. Processes two amplitude pairs per iteration.
///
/// Requires `k >= 1` (stride `>= 2`) so that the bit-`k`-clear indices and their
/// partners each form contiguous runs of two `Complex64` that fit one register.
#[cfg(all(target_arch = "x86_64", not(miri)))]
#[target_feature(enable = "avx2,fma")]
// the m00r/m00i/... broadcasts map directly onto the gate matrix entries; the
// parallel naming is the point, so the similar-names lint is noise here.
#[allow(clippy::similar_names)]
unsafe fn apply_1q_avx2(amps: &mut [Complex64], k: usize, gate: &Gate1) {
    use std::arch::x86_64::{
        _mm256_add_pd, _mm256_fmaddsub_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_permute_pd,
        _mm256_set1_pd, _mm256_storeu_pd,
    };

    let n = amps.len();
    let stride = 1usize << k; // >= 2 and even
    let m = &gate.m;

    // SAFETY (whole body): `Complex64` is `#[repr(C)]` so `[Complex64]` aliases
    // `[f64]` two-to-one; `ptr.add(2 * i)` addresses amplitude `i`. the stride
    // loop visits each pair (i0, i1 = i0 + stride) exactly once with both runs
    // in bounds: the largest access is `i1 + 1 = base + 2*stride - 1 < n`. reads
    // happen before writes and the i0/i1 runs are disjoint, so no aliasing.
    unsafe {
        // broadcast the real and imaginary parts of each matrix entry.
        let m00r = _mm256_set1_pd(m[0].re);
        let m00i = _mm256_set1_pd(m[0].im);
        let m01r = _mm256_set1_pd(m[1].re);
        let m01i = _mm256_set1_pd(m[1].im);
        let m10r = _mm256_set1_pd(m[2].re);
        let m10i = _mm256_set1_pd(m[2].im);
        let m11r = _mm256_set1_pd(m[3].re);
        let m11i = _mm256_set1_pd(m[3].im);

        let ptr = amps.as_mut_ptr().cast::<f64>();
        let mut base = 0;
        while base < n {
            let mut off = 0;
            while off < stride {
                let i0 = base + off;
                let i1 = i0 + stride;
                // each register holds two complex: [re0, im0, re1, im1].
                let v0 = _mm256_loadu_pd(ptr.add(2 * i0));
                let v1 = _mm256_loadu_pd(ptr.add(2 * i1));
                // swap re/im within each complex for the cross terms.
                let v0s = _mm256_permute_pd(v0, 0b0101);
                let v1s = _mm256_permute_pd(v1, 0b0101);
                // fmaddsub(mr, z, mi*swap(z)) = (mr,mi) * z, packed two complex.
                let m00v0 = _mm256_fmaddsub_pd(m00r, v0, _mm256_mul_pd(m00i, v0s));
                let m01v1 = _mm256_fmaddsub_pd(m01r, v1, _mm256_mul_pd(m01i, v1s));
                let m10v0 = _mm256_fmaddsub_pd(m10r, v0, _mm256_mul_pd(m10i, v0s));
                let m11v1 = _mm256_fmaddsub_pd(m11r, v1, _mm256_mul_pd(m11i, v1s));
                let new0 = _mm256_add_pd(m00v0, m01v1);
                let new1 = _mm256_add_pd(m10v0, m11v1);
                _mm256_storeu_pd(ptr.add(2 * i0), new0);
                _mm256_storeu_pd(ptr.add(2 * i1), new1);
                off += 2;
            }
            base += 2 * stride;
        }
    }
}

/// Applies a two-qubit gate to qubits `a` (more significant in the gate basis)
/// and `b` of an `n`-qubit state.
///
/// `a` and `b` must be distinct and both `< n`; `amps` must have length `2^n`.
///
/// CNOT, CZ, and SWAP dispatch to specialized loops that avoid complex
/// multiplies; all other gates use the general dense 4×4 path.
pub fn apply_2q(amps: &mut [Complex64], a: usize, b: usize, gate: &Gate2) {
    let n = amps.len();
    debug_assert!(a != b);
    debug_assert!(n.is_power_of_two());
    debug_assert!((1usize << a) < n && (1usize << b) < n);

    // exact-match dispatch for the three common permutation/diagonal gates.
    // these need no complex multiply, so a specialized loop is most of the win.
    // exact equality is correct because the const matrices use integer values.
    if *gate == Gate2::cnot() {
        apply_2q_cnot(amps, a, b);
        return;
    }
    if *gate == Gate2::cz() {
        apply_2q_cz(amps, a, b);
        return;
    }
    if *gate == Gate2::swap() {
        apply_2q_swap(amps, a, b);
        return;
    }

    let mat = &gate.m;
    let groups = n / 4;
    for i in 0..groups {
        let idx = index_quad(a, b, i);
        debug_assert!(idx.iter().all(|&j| j < n));
        // SAFETY: with `n` a power of two and `a, b < log2(n)` distinct,
        // `index_quad(a, b, i)` for `i < n/4` yields four distinct indices all
        // `< n` (proven in `proofs`). the four amplitudes are read into `amp`
        // before any write, so the writes never alias the reads.
        let amp = unsafe {
            [
                *amps.get_unchecked(idx[0]),
                *amps.get_unchecked(idx[1]),
                *amps.get_unchecked(idx[2]),
                *amps.get_unchecked(idx[3]),
            ]
        };
        for (row, &out) in idx.iter().enumerate() {
            let base = row * 4;
            // dense 4x4 matrix-vector product for this amplitude group.
            let new = mat[base] * amp[0]
                + mat[base + 1] * amp[1]
                + mat[base + 2] * amp[2]
                + mat[base + 3] * amp[3];
            // SAFETY: `out` is one of the four indices proven `< n` above.
            unsafe {
                *amps.get_unchecked_mut(out) = new;
            }
        }
    }
}

/// CNOT: for each group swap the `i10` and `i11` amplitudes (flip target when
/// control qubit `a` is set). no complex arithmetic needed.
fn apply_2q_cnot(amps: &mut [Complex64], a: usize, b: usize) {
    let groups = amps.len() / 4;
    for i in 0..groups {
        let idx = index_quad(a, b, i);
        debug_assert!(idx.iter().all(|&j| j < amps.len()));
        // SAFETY: index_quad returns 4 distinct in-bounds indices (Kani-proven).
        unsafe {
            let tmp = *amps.get_unchecked(idx[2]);
            *amps.get_unchecked_mut(idx[2]) = *amps.get_unchecked(idx[3]);
            *amps.get_unchecked_mut(idx[3]) = tmp;
        }
    }
}

/// CZ: negate the `i11` amplitude. symmetric in its operands.
fn apply_2q_cz(amps: &mut [Complex64], a: usize, b: usize) {
    let groups = amps.len() / 4;
    for i in 0..groups {
        let idx = index_quad(a, b, i);
        debug_assert!(idx[3] < amps.len());
        // SAFETY: index_quad returns 4 distinct in-bounds indices (Kani-proven).
        unsafe {
            let v = amps.get_unchecked_mut(idx[3]);
            *v = -*v;
        }
    }
}

/// SWAP: exchange the `i01` and `i10` amplitudes.
fn apply_2q_swap(amps: &mut [Complex64], a: usize, b: usize) {
    let groups = amps.len() / 4;
    for i in 0..groups {
        let idx = index_quad(a, b, i);
        debug_assert!(idx.iter().all(|&j| j < amps.len()));
        // SAFETY: index_quad returns 4 distinct in-bounds indices (Kani-proven).
        unsafe {
            let tmp = *amps.get_unchecked(idx[1]);
            *amps.get_unchecked_mut(idx[1]) = *amps.get_unchecked(idx[2]);
            *amps.get_unchecked_mut(idx[2]) = tmp;
        }
    }
}

/// Applies a single-qubit gate to `target`, but only on the subspace where
/// every qubit in `controls` is set. This is the general controlled gate.
///
/// `target` must not appear in `controls`; all indices must be `< n`.
///
/// On `x86_64` with AVX2 + FMA, and when `target >= 1` and no control bit
/// occupies position 0, dispatches to a vectorized path that processes two
/// amplitude pairs per iteration (same FMA core as `apply_1q_avx2`).
pub fn apply_controlled_1q(
    amps: &mut [Complex64],
    controls: &[usize],
    target: usize,
    gate: &Gate1,
) {
    debug_assert!(!controls.contains(&target));
    debug_assert!(amps.len().is_power_of_two());
    let mut control_mask = 0usize;
    for &c in controls {
        control_mask |= 1usize << c;
    }

    // the SIMD path processes two adjacent pairs per step. the two i0 values in
    // a step differ only in bit 0, so they share the same control test iff bit 0
    // is not a control bit. also need target >= 1 (stride >= 2) to load two
    // contiguous complex per register, same condition as apply_1q_avx2.
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    {
        if target >= 1
            && (control_mask & 1) == 0
            && is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
        {
            // SAFETY: guarded by the runtime feature detection just above.
            unsafe { apply_controlled_1q_avx2(amps, control_mask, target, gate) };
            return;
        }
    }

    apply_controlled_1q_scalar(amps, control_mask, target, gate);
}

fn apply_controlled_1q_scalar(
    amps: &mut [Complex64],
    control_mask: usize,
    target: usize,
    gate: &Gate1,
) {
    let n = amps.len();
    let m = &gate.m;
    let pairs = n / 2;
    for i in 0..pairs {
        let (i0, i1) = index_pair(target, i);
        // apply only when all control bits are set in this pair's indices.
        // i0 and i1 agree on every bit except target, so testing i0 suffices.
        if i0 & control_mask == control_mask {
            debug_assert!(i0 < n && i1 < n);
            // SAFETY: identical bounds argument to `apply_1q` — `index_pair`
            // gives `i0, i1 < n` for the target qubit, and the pair is read
            // before being written.
            unsafe {
                let a0 = *amps.get_unchecked(i0);
                let a1 = *amps.get_unchecked(i1);
                *amps.get_unchecked_mut(i0) = m[0] * a0 + m[1] * a1;
                *amps.get_unchecked_mut(i1) = m[2] * a0 + m[3] * a1;
            }
        }
    }
}

/// AVX2 + FMA controlled single-qubit kernel. Processes two adjacent amplitude
/// pairs per iteration when the control test passes.
///
/// Preconditions (caller checks): `target >= 1`, `(control_mask & 1) == 0`,
/// AVX2+FMA available. The bit-0 condition ensures both pairs in each iteration
/// share the same control outcome — the two i0 values differ only in bit 0,
/// which is not a control bit, so `i0a & mask == i0b & mask`.
#[cfg(all(target_arch = "x86_64", not(miri)))]
#[target_feature(enable = "avx2,fma")]
// m00r/m00i etc. are parallel matrix-entry broadcasts; similar names are the point.
#[allow(clippy::similar_names)]
unsafe fn apply_controlled_1q_avx2(
    amps: &mut [Complex64],
    control_mask: usize,
    target: usize,
    gate: &Gate1,
) {
    use std::arch::x86_64::{
        _mm256_add_pd, _mm256_fmaddsub_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_permute_pd,
        _mm256_set1_pd, _mm256_storeu_pd,
    };

    let n = amps.len();
    let stride = 1usize << target; // >= 2
    let m = &gate.m;

    // SAFETY (whole body): same repr aliasing as apply_1q_avx2 — `Complex64`
    // is `#[repr(C)]` so ptr.add(2*i) addresses amplitude i. the stride loop
    // visits each pair (i0, i1 = i0 + stride) exactly once, both in bounds.
    // reads happen before writes; i0/i1 runs are disjoint. the control test
    // uses i0 (both pairs in the block pass or fail identically because bit 0
    // is not a control bit, per the precondition checked by the caller).
    unsafe {
        let m00r = _mm256_set1_pd(m[0].re);
        let m00i = _mm256_set1_pd(m[0].im);
        let m01r = _mm256_set1_pd(m[1].re);
        let m01i = _mm256_set1_pd(m[1].im);
        let m10r = _mm256_set1_pd(m[2].re);
        let m10i = _mm256_set1_pd(m[2].im);
        let m11r = _mm256_set1_pd(m[3].re);
        let m11i = _mm256_set1_pd(m[3].im);

        let ptr = amps.as_mut_ptr().cast::<f64>();
        let mut base = 0;
        while base < n {
            let mut off = 0;
            while off < stride {
                // two adjacent pairs: (base+off, base+off+stride) and
                // (base+off+1, base+off+1+stride). off and off+1 both clear
                // the target bit (off < stride, so bit `target` is clear in
                // off when it starts at 0 and increments by 2 below).
                let i0a = base + off;
                // test control on i0a; i0a+1 gives the same result because bit 0
                // is not a control bit (precondition), so both pairs pass or fail.
                if i0a & control_mask == control_mask {
                    let i1a = i0a + stride;
                    // load two complex from i0a,i0b and i1a,i1b.
                    let v0 = _mm256_loadu_pd(ptr.add(2 * i0a));
                    let v1 = _mm256_loadu_pd(ptr.add(2 * i1a));
                    let v0s = _mm256_permute_pd(v0, 0b0101);
                    let v1s = _mm256_permute_pd(v1, 0b0101);
                    let m00v0 = _mm256_fmaddsub_pd(m00r, v0, _mm256_mul_pd(m00i, v0s));
                    let m01v1 = _mm256_fmaddsub_pd(m01r, v1, _mm256_mul_pd(m01i, v1s));
                    let m10v0 = _mm256_fmaddsub_pd(m10r, v0, _mm256_mul_pd(m10i, v0s));
                    let m11v1 = _mm256_fmaddsub_pd(m11r, v1, _mm256_mul_pd(m11i, v1s));
                    let new0 = _mm256_add_pd(m00v0, m01v1);
                    let new1 = _mm256_add_pd(m10v0, m11v1);
                    // 256-bit storeu writes 4 f64 = 2 complex starting at i0a,
                    // covering both i0a and i0b (adjacent). same for i1a/i1b.
                    _mm256_storeu_pd(ptr.add(2 * i0a), new0);
                    _mm256_storeu_pd(ptr.add(2 * i1a), new1);
                }
                off += 2;
            }
            base += 2 * stride;
        }
    }
}

/// Bounded formal proofs of the index arithmetic, checked by Kani.
///
/// These establish — for all qubit counts up to a model-checking bound — that
/// the computed indices stay in bounds and are distinct, which is the safety
/// obligation the `get_unchecked` calls above rely on. Run with `cargo kani`.
#[cfg(kani)]
mod proofs {
    use super::{index_pair, index_quad};

    // prove the single-qubit pair is in bounds, distinct, and that i1 is i0
    // with bit k set, for every qubit count q in 1..=12.
    #[kani::proof]
    fn index_pair_in_bounds() {
        let q: usize = kani::any();
        kani::assume((1..=12).contains(&q));
        let n: usize = 1 << q; // 2^q amplitudes

        let k: usize = kani::any();
        kani::assume(k < q); // valid qubit, so 2^k < n

        let i: usize = kani::any();
        kani::assume(i < n / 2); // loop counter range

        let (i0, i1) = index_pair(k, i);
        assert!(i0 < n);
        assert!(i1 < n);
        assert!(i0 != i1);
        assert!(i1 == i0 | (1 << k));
        assert!(i0 & (1 << k) == 0);
    }

    // prove injectivity: distinct loop counters give distinct i0, so the loop
    // is a permutation (every amplitude touched exactly once).
    #[kani::proof]
    fn index_pair_injective() {
        let q: usize = kani::any();
        kani::assume((1..=12).contains(&q));
        let k: usize = kani::any();
        kani::assume(k < q);
        let half: usize = (1usize << q) / 2;

        let i: usize = kani::any();
        let j: usize = kani::any();
        kani::assume(i < half && j < half && i != j);

        let (i0, _) = index_pair(k, i);
        let (j0, _) = index_pair(k, j);
        assert!(i0 != j0);
    }

    // prove the two-qubit quad is in bounds and pairwise distinct.
    #[kani::proof]
    fn index_quad_in_bounds() {
        let q: usize = kani::any();
        kani::assume((2..=10).contains(&q));
        let n: usize = 1 << q;
        let a: usize = kani::any();
        let b: usize = kani::any();
        kani::assume(a < q && b < q && a != b);
        let i: usize = kani::any();
        kani::assume(i < n / 4);

        let idx = index_quad(a, b, i);
        let mut x = 0;
        while x < 4 {
            assert!(idx[x] < n);
            let mut y = x + 1;
            while y < 4 {
                assert!(idx[x] != idx[y]);
                y += 1;
            }
            x += 1;
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

    #[test]
    fn miri_apply_1q_no_ub_small() {
        // small-n exercise of the unsafe path for Miri to validate. covers a
        // few qubit counts and targets without being expensive under Miri.
        for n_qubits in 1..=4 {
            let dim = 1usize << n_qubits;
            for k in 0..n_qubits {
                let mut amps = vec![Complex64::ZERO; dim];
                amps[0] = Complex64::ONE;
                apply_1q(&mut amps, k, &Gate1::h());
                // norm is preserved by a unitary.
                let norm: f64 = amps.iter().map(|a| a.norm_sqr()).sum();
                assert!((norm - 1.0).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn miri_apply_2q_and_controlled_no_ub_small() {
        // exercise the two-qubit and controlled unsafe paths for Miri.
        for n_qubits in 2..=4 {
            let dim = 1usize << n_qubits;
            let mut amps = vec![Complex64::ZERO; dim];
            amps[0] = Complex64::ONE;
            apply_1q(&mut amps, 0, &Gate1::h());
            apply_2q(&mut amps, 0, 1, &Gate2::cnot());
            apply_controlled_1q(&mut amps, &[0], 1, &Gate1::z());
            let norm: f64 = amps.iter().map(|a| a.norm_sqr()).sum();
            assert!((norm - 1.0).abs() < 1e-12);
        }
    }

    // a deterministic but non-trivial state for equivalence checks.
    fn sample_state(n: usize) -> Vec<Complex64> {
        (0..(1usize << n))
            .map(|j| Complex64::new(0.1 + j as f64 * 0.03, -0.2 + j as f64 * 0.017))
            .collect()
    }

    #[test]
    fn specialized_2q_matches_dense() {
        // the CNOT/CZ/SWAP fast paths must agree with the general dense kernel
        // across all qubit counts n=2..10 and every distinct operand pair (a,b).
        for n in 2..=10 {
            for a in 0..n {
                for b in 0..n {
                    if a == b {
                        continue;
                    }
                    let base = sample_state(n);
                    for gate in [Gate2::cnot(), Gate2::cz(), Gate2::swap()] {
                        let mut via_dispatch = base.clone();
                        apply_2q(&mut via_dispatch, a, b, &gate);

                        // run the general dense path directly by bypassing dispatch:
                        // build a fresh gate that is != the const matrices so the
                        // specialized branch is NOT taken. instead scale by 1.0 to
                        // produce a semantically identical but pointer-distinct gate.
                        // easier: just call apply_2q on a copy for the scalar ref by
                        // using a non-matching gate wrapped in the dense path.
                        // simplest correct ref: a dedicated dense-only fn.
                        let mut via_dense = base.clone();
                        apply_2q_dense_ref(&mut via_dense, a, b, &gate);

                        for (x, y) in via_dispatch.iter().zip(&via_dense) {
                            assert!(
                                (*x - *y).norm() < 1e-12,
                                "specialized/dense mismatch n={n} a={a} b={b}"
                            );
                        }
                    }
                }
            }
        }
    }

    // reference dense path that never takes the specialized branches.
    fn apply_2q_dense_ref(amps: &mut [Complex64], a: usize, b: usize, gate: &Gate2) {
        let n = amps.len();
        let mat = &gate.m;
        let groups = n / 4;
        for i in 0..groups {
            let idx = index_quad(a, b, i);
            let amp = [amps[idx[0]], amps[idx[1]], amps[idx[2]], amps[idx[3]]];
            for (row, &out) in idx.iter().enumerate() {
                let base = row * 4;
                amps[out] = mat[base] * amp[0]
                    + mat[base + 1] * amp[1]
                    + mat[base + 2] * amp[2]
                    + mat[base + 3] * amp[3];
            }
        }
    }

    #[test]
    fn simd_controlled_matches_scalar() {
        // apply_controlled_1q dispatches to SIMD on x86_64+avx2 when target>=1
        // and (control_mask & 1)==0. check SIMD≡scalar across qubit counts,
        // operand placements, and a representative gate set.
        let gates = [
            Gate1::x(),
            Gate1::y(),
            Gate1::z(),
            Gate1::h(),
            Gate1::s(),
            Gate1::rx(1.1),
            Gate1::ry(-0.7),
            Gate1::rz(2.3),
        ];
        for n in 2..=10 {
            for target in 0..n {
                for ctrl in 0..n {
                    if ctrl == target {
                        continue;
                    }
                    let control_mask = 1usize << ctrl;
                    let base = sample_state(n);
                    for g in &gates {
                        let mut via_dispatch = base.clone();
                        apply_controlled_1q(&mut via_dispatch, &[ctrl], target, g);

                        let mut via_scalar = base.clone();
                        apply_controlled_1q_scalar(&mut via_scalar, control_mask, target, g);

                        for (x, y) in via_dispatch.iter().zip(&via_scalar) {
                            assert!(
                                (*x - *y).norm() < 1e-12,
                                "controlled SIMD/scalar mismatch n={n} target={target} ctrl={ctrl}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn simd_path_matches_scalar() {
        // apply_1q dispatches to SIMD on x86_64+avx2; it must agree with the
        // scalar reference for every qubit and a representative gate set. on
        // non-x86 or non-avx2 both calls are scalar, so this is still a valid
        // (if trivial) check there.
        let gates = [
            Gate1::h(),
            Gate1::x(),
            Gate1::y(),
            Gate1::s(),
            Gate1::t(),
            Gate1::rx(0.7),
            Gate1::ry(-1.3),
            Gate1::rz(2.1),
            Gate1::phase(0.9),
        ];
        for n in 1..=10 {
            for k in 0..n {
                for g in &gates {
                    let mut via_dispatch = sample_state(n);
                    let mut via_scalar = via_dispatch.clone();
                    apply_1q(&mut via_dispatch, k, g);
                    apply_1q_scalar(&mut via_scalar, k, g);
                    for (a, b) in via_dispatch.iter().zip(&via_scalar) {
                        // FMA in the SIMD path can differ from separate mul+add
                        // by a rounding step, so allow a tiny tolerance.
                        assert!(
                            (*a - *b).norm() < 1e-12,
                            "mismatch at n={n} k={k}: {a} vs {b}"
                        );
                    }
                }
            }
        }
    }
}
