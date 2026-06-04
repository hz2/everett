//! Projective measurement of a single qubit in the computational basis.

use crate::rng::Rng;
use crate::state::State;

/// Measures qubit `k` in the computational basis, collapsing the state.
///
/// Returns the observed outcome (`false` for `|0>`, `true` for `|1>`), sampled
/// according to the Born rule, and renormalizes the post-measurement state onto
/// the observed subspace.
///
/// # Panics
///
/// Panics if `k` is not a valid qubit index for `state`.
pub fn measure_qubit(state: &mut State, k: usize, rng: &mut Rng) -> bool {
    assert!(k < state.num_qubits(), "qubit {k} out of range");
    let bit = 1usize << k;

    // probability of observing |1> on qubit k: sum of |amp|^2 over indices with
    // bit k set.
    let amps = state.amplitudes();
    let mut p1 = 0.0;
    for (j, amp) in amps.iter().enumerate() {
        if j & bit != 0 {
            p1 += amp.norm_sqr();
        }
    }

    let outcome = rng.next_f64() < p1;

    // project onto the observed subspace and renormalize. the surviving norm is
    // p1 if we saw |1>, else 1 - p1.
    let surviving = if outcome { p1 } else { 1.0 - p1 };
    let scale = if surviving > 0.0 {
        1.0 / surviving.sqrt()
    } else {
        0.0
    };

    let amps = state.amplitudes_mut();
    for (j, amp) in amps.iter_mut().enumerate() {
        let keep = (j & bit != 0) == outcome;
        if keep {
            *amp *= scale;
        } else {
            *amp = crate::complex::Complex64::ZERO;
        }
    }

    outcome
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate::Gate1;
    use crate::kernel::apply_1q;

    #[test]
    fn measuring_zero_state_yields_zero() {
        let mut state = State::zero(1);
        let mut rng = Rng::seed_from_u64(0);
        assert!(!measure_qubit(&mut state, 0, &mut rng));
        // still |0>.
        assert!((state.probability(0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn measurement_collapses_and_renormalizes() {
        // |+> = H|0>; after measuring, the state must be a basis state.
        let mut state = State::zero(1);
        apply_1q(state.amplitudes_mut(), 0, &Gate1::h());
        let mut rng = Rng::seed_from_u64(99);
        let outcome = measure_qubit(&mut state, 0, &mut rng);
        assert!((state.norm_sqr() - 1.0).abs() < 1e-12);
        let collapsed = usize::from(outcome);
        assert!((state.probability(collapsed) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn plus_state_statistics_are_balanced() {
        // measure |+> many times; outcomes should be ~50/50.
        let mut rng = Rng::seed_from_u64(2024);
        let trials = 4000;
        let mut ones = 0;
        for _ in 0..trials {
            let mut state = State::zero(1);
            apply_1q(state.amplitudes_mut(), 0, &Gate1::h());
            if measure_qubit(&mut state, 0, &mut rng) {
                ones += 1;
            }
        }
        let frac = f64::from(ones) / f64::from(trials);
        assert!((frac - 0.5).abs() < 0.05, "fraction of ones was {frac}");
    }

    #[test]
    fn deterministic_after_measurement() {
        // measuring the same qubit twice must agree (no further randomness).
        let mut state = State::zero(2);
        apply_1q(state.amplitudes_mut(), 0, &Gate1::h());
        let mut rng = Rng::seed_from_u64(5);
        let first = measure_qubit(&mut state, 0, &mut rng);
        let second = measure_qubit(&mut state, 0, &mut rng);
        assert_eq!(first, second);
    }
}
