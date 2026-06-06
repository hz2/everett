//! property-based tests for the everett quantum simulator.
//
// these tests probe the algebraic laws that must hold for a correct simulator:
//   A. complex arithmetic identities
//   B. gate unitarity and adjoint laws
//   C. state normalization under gate sequences
//   D. gate identities (H^2=I, HXH=Z, etc.) applied to states
//   E. measurement sanity
//   F. kernel vs naive reference cross-check

use approx::assert_abs_diff_eq;
use everett::{Circuit, Complex64, Gate1, Gate2, State, StateVectorBackend};
use proptest::prelude::*;

// ─── shared strategies ────────────────────────────────────────────────────────

// complex number with components in a bounded range to avoid catastrophic
// cancellation that would inflate floating-point tolerances.
fn arb_complex(max_mag: f64) -> impl Strategy<Value = Complex64> {
    (-max_mag..max_mag, -max_mag..max_mag).prop_map(|(re, im)| Complex64::new(re, im))
}

// angle in a wide range; no need to restrict to [0, 2π).
fn arb_angle() -> impl Strategy<Value = f64> {
    -10.0f64..10.0
}

// qubit count n and a valid target index: returns (n, target) with target < n.
fn arb_n_and_target() -> impl Strategy<Value = (usize, usize)> {
    (1usize..=8).prop_flat_map(|n| (Just(n), 0..n))
}

// two distinct valid targets for n qubits: returns (n, a, b) with a != b.
fn arb_n_and_two_targets() -> impl Strategy<Value = (usize, usize, usize)> {
    (2usize..=8).prop_flat_map(|n| {
        (Just(n), 0..n).prop_flat_map(move |(_n, a)| {
            let others: Vec<usize> = (0..n).filter(|&x| x != a).collect();
            let b_idx = 0..others.len();
            (Just(n), Just(a), b_idx.prop_map(move |i| others[i]))
        })
    })
}

// a random normalized 1-qubit state prepared by ry(theta)*rz(lambda) on |0>.
fn arb_1q_state() -> impl Strategy<Value = State> {
    (arb_angle(), arb_angle()).prop_map(|(theta, lambda)| {
        let mut c = Circuit::new(1);
        c.ry(0, theta).rz(0, lambda);
        StateVectorBackend::run(&c).unwrap().into_state()
    })
}

// a random normalized 2-qubit product state.
fn arb_2q_state() -> impl Strategy<Value = State> {
    (arb_angle(), arb_angle(), arb_angle(), arb_angle()).prop_map(|(t0, l0, t1, l1)| {
        let mut c = Circuit::new(2);
        c.ry(0, t0).rz(0, l0).ry(1, t1).rz(1, l1);
        StateVectorBackend::run(&c).unwrap().into_state()
    })
}

// ─── A. complex arithmetic ────────────────────────────────────────────────────

proptest! {
    #[test]
    fn complex_mul_commutative(
        a in arb_complex(100.0),
        b in arb_complex(100.0),
    ) {
        let ab = a * b;
        let ba = b * a;
        assert_abs_diff_eq!(ab.re, ba.re, epsilon = 1e-9);
        assert_abs_diff_eq!(ab.im, ba.im, epsilon = 1e-9);
    }

    #[test]
    fn complex_mul_associative(
        a in arb_complex(10.0),
        b in arb_complex(10.0),
        c in arb_complex(10.0),
    ) {
        // (a*b)*c == a*(b*c). rounding accumulates with large inputs; use 10.0.
        let lhs = (a * b) * c;
        let rhs = a * (b * c);
        assert_abs_diff_eq!(lhs.re, rhs.re, epsilon = 1e-9);
        assert_abs_diff_eq!(lhs.im, rhs.im, epsilon = 1e-9);
    }

    #[test]
    fn complex_conj_norm(a in arb_complex(100.0)) {
        // z * conj(z) is real and equals |z|^2. the imaginary part is exactly 0
        // in exact arithmetic; in f64 rounding it's bounded by the product of the
        // ULP of |a|^2 (~1e4) and machine epsilon (~1e-16), so ~1e-12.
        let p = a * a.conj();
        assert_abs_diff_eq!(p.re, a.norm_sqr(), epsilon = 1e-9);
        assert_abs_diff_eq!(p.im, 0.0, epsilon = 1e-9);
    }

    #[test]
    fn expi_is_unit_modulus(theta in arb_angle()) {
        assert_abs_diff_eq!(Complex64::expi(theta).norm_sqr(), 1.0, epsilon = 1e-14);
    }

    #[test]
    fn complex_distributive(
        a in arb_complex(10.0),
        b in arb_complex(10.0),
        c in arb_complex(10.0),
    ) {
        let lhs = a * (b + c);
        let rhs = a * b + a * c;
        assert_abs_diff_eq!(lhs.re, rhs.re, epsilon = 1e-9);
        assert_abs_diff_eq!(lhs.im, rhs.im, epsilon = 1e-9);
    }
}

// ─── B. gate unitarity ────────────────────────────────────────────────────────

proptest! {
    #[test]
    fn parametric_gates_are_unitary(theta in arb_angle()) {
        for gate in [Gate1::rx(theta), Gate1::ry(theta), Gate1::rz(theta), Gate1::phase(theta)] {
            prop_assert!(gate.is_unitary(1e-12), "gate not unitary for theta={theta}: {gate:?}");
        }
    }

    #[test]
    fn adjoint_is_left_inverse(theta in arb_angle()) {
        for gate in [Gate1::rx(theta), Gate1::ry(theta), Gate1::rz(theta), Gate1::phase(theta)] {
            prop_assert!(
                gate.adjoint().compose(&gate).is_identity(1e-12),
                "U†U not identity for theta={theta}"
            );
        }
    }

    #[test]
    fn adjoint_is_right_inverse(theta in arb_angle()) {
        for gate in [Gate1::rx(theta), Gate1::ry(theta), Gate1::rz(theta), Gate1::phase(theta)] {
            prop_assert!(
                gate.compose(&gate.adjoint()).is_identity(1e-12),
                "UU† not identity for theta={theta}"
            );
        }
    }

    #[test]
    fn controlled_gate1_is_unitary(theta in arb_angle()) {
        for u in [Gate1::rx(theta), Gate1::ry(theta), Gate1::rz(theta)] {
            let cu = Gate2::controlled(&u);
            prop_assert!(cu.is_unitary(1e-12));
        }
    }
}

// ─── C. state normalization under arbitrary gate sequences ───────────────────

proptest! {
    // reduced cases because we generate circuits up to n=8 with several gates.
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn single_gate_preserves_norm(
        (n, k) in arb_n_and_target(),
        theta in arb_angle(),
    ) {
        let gate = Gate1::ry(theta);
        let mut c = Circuit::new(n);
        c.gate1(gate, k);
        let state = StateVectorBackend::run(&c).unwrap().into_state();
        assert_abs_diff_eq!(state.norm_sqr(), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn sequence_of_gates_preserves_norm(
        (n, k) in arb_n_and_target(),
        t0 in arb_angle(),
        t1 in arb_angle(),
        t2 in arb_angle(),
    ) {
        let mut c = Circuit::new(n);
        c.gate1(Gate1::rx(t0), k)
            .gate1(Gate1::ry(t1), k)
            .gate1(Gate1::rz(t2), k);
        let state = StateVectorBackend::run(&c).unwrap().into_state();
        assert_abs_diff_eq!(state.norm_sqr(), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn two_qubit_gate_preserves_norm(
        (n, a, b) in arb_n_and_two_targets(),
        t0 in arb_angle(),
        t1 in arb_angle(),
    ) {
        // prepare non-trivial state then apply a two-qubit gate.
        let mut c = Circuit::new(n);
        c.gate1(Gate1::ry(t0), a)
            .gate1(Gate1::rz(t1), b)
            .gate2(Gate2::cnot(), a, b);
        let state = StateVectorBackend::run(&c).unwrap().into_state();
        assert_abs_diff_eq!(state.norm_sqr(), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn probabilities_sum_to_one(
        (n, k) in arb_n_and_target(),
        theta in arb_angle(),
    ) {
        let mut c = Circuit::new(n);
        c.gate1(Gate1::ry(theta), k);
        let state = StateVectorBackend::run(&c).unwrap().into_state();
        let sum: f64 = (0..state.dim()).map(|j| state.probability(j)).sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-12);
    }
}

// ─── D. gate identities applied to states ────────────────────────────────────

proptest! {
    #[test]
    fn h_squared_is_identity_on_state(state in arb_1q_state()) {
        let original = state.clone();
        let mut c = Circuit::new(1);
        c.h(0).h(0);
        // apply circuit to the prepared state by composing with the prep circuit.
        // simpler: prepare via H·H = I, i.e. run H twice on |0> after the state.
        // we test it at the state level: H applied twice should return to original.
        let mut amps = state.amplitudes().to_vec();
        everett::kernel::apply_1q(&mut amps, 0, &Gate1::h());
        everett::kernel::apply_1q(&mut amps, 0, &Gate1::h());
        let final_state = State::from_amplitudes(amps).unwrap();
        // fidelity is phase-agnostic; H is real so global phase is +1 anyway.
        assert_abs_diff_eq!(final_state.fidelity(&original), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn x_squared_is_identity_on_state(state in arb_1q_state()) {
        let original = state.clone();
        let mut amps = state.amplitudes().to_vec();
        everett::kernel::apply_1q(&mut amps, 0, &Gate1::x());
        everett::kernel::apply_1q(&mut amps, 0, &Gate1::x());
        let final_state = State::from_amplitudes(amps).unwrap();
        assert_abs_diff_eq!(final_state.fidelity(&original), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn hxh_equals_z_on_state(state in arb_1q_state()) {
        // HXH|ψ> = Z|ψ>  for any |ψ>.
        let mut lhs = state.amplitudes().to_vec();
        everett::kernel::apply_1q(&mut lhs, 0, &Gate1::h());
        everett::kernel::apply_1q(&mut lhs, 0, &Gate1::x());
        everett::kernel::apply_1q(&mut lhs, 0, &Gate1::h());

        let mut rhs = state.amplitudes().to_vec();
        everett::kernel::apply_1q(&mut rhs, 0, &Gate1::z());

        let s_lhs = State::from_amplitudes(lhs).unwrap();
        let s_rhs = State::from_amplitudes(rhs).unwrap();
        assert_abs_diff_eq!(s_lhs.fidelity(&s_rhs), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn hzh_equals_x_on_state(state in arb_1q_state()) {
        // HZH|ψ> = X|ψ>.
        let mut lhs = state.amplitudes().to_vec();
        everett::kernel::apply_1q(&mut lhs, 0, &Gate1::h());
        everett::kernel::apply_1q(&mut lhs, 0, &Gate1::z());
        everett::kernel::apply_1q(&mut lhs, 0, &Gate1::h());

        let mut rhs = state.amplitudes().to_vec();
        everett::kernel::apply_1q(&mut rhs, 0, &Gate1::x());

        let s_lhs = State::from_amplitudes(lhs).unwrap();
        let s_rhs = State::from_amplitudes(rhs).unwrap();
        assert_abs_diff_eq!(s_lhs.fidelity(&s_rhs), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn cnot_twice_is_identity(state in arb_2q_state()) {
        let original = state.clone();
        let mut amps = state.amplitudes().to_vec();
        everett::kernel::apply_2q(&mut amps, 1, 0, &Gate2::cnot());
        everett::kernel::apply_2q(&mut amps, 1, 0, &Gate2::cnot());
        let final_state = State::from_amplitudes(amps).unwrap();
        assert_abs_diff_eq!(final_state.fidelity(&original), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn swap_exchanges_bloch_vectors(t0 in arb_angle(), t1 in arb_angle()) {
        // prepare a product state |ψ0>⊗|ψ1>, swap, check qubit 0 looks like ψ1
        // and qubit 1 looks like ψ0. use a 2-qubit state built as a product state
        // by applying independent rotations on each qubit.
        let mut c = Circuit::new(2);
        // prepare qubit 0 with t0, qubit 1 with t1 (distinct rotation angles).
        c.ry(0, t0).ry(1, t1);
        let before = StateVectorBackend::run(&c).unwrap().into_state();
        let bloch0_before = before.bloch_vector(0);
        let bloch1_before = before.bloch_vector(1);

        let mut amps = before.amplitudes().to_vec();
        everett::kernel::apply_2q(&mut amps, 0, 1, &Gate2::swap());
        let after = State::from_amplitudes(amps).unwrap();
        let bloch0_after = after.bloch_vector(0);
        let bloch1_after = after.bloch_vector(1);

        // qubit 0 after swap should equal qubit 1 before.
        for i in 0..3 {
            assert_abs_diff_eq!(bloch0_after[i], bloch1_before[i], epsilon = 1e-11);
            assert_abs_diff_eq!(bloch1_after[i], bloch0_before[i], epsilon = 1e-11);
        }
    }

    #[test]
    fn rz_pi_equals_z_up_to_phase(state in arb_1q_state()) {
        // R_z(π) = e^{-iπ/2} Z; as a gate on a state they give fidelity 1.
        let mut rz = state.amplitudes().to_vec();
        everett::kernel::apply_1q(&mut rz, 0, &Gate1::rz(std::f64::consts::PI));

        let mut z = state.amplitudes().to_vec();
        everett::kernel::apply_1q(&mut z, 0, &Gate1::z());

        let s_rz = State::from_amplitudes(rz).unwrap();
        let s_z = State::from_amplitudes(z).unwrap();
        assert_abs_diff_eq!(s_rz.fidelity(&s_z), 1.0, epsilon = 1e-12);
    }
}

// ─── E. measurement sanity ────────────────────────────────────────────────────

// analytic check: P(|1>) for |+> = 1/2 exactly; no sampling needed.
#[test]
fn plus_state_has_equal_analytic_probabilities() {
    let mut c = Circuit::new(1);
    c.h(0);
    let exec = StateVectorBackend::run(&c).unwrap();
    let state = exec.state();
    assert_abs_diff_eq!(state.probability(0), 0.5, epsilon = 1e-12);
    assert_abs_diff_eq!(state.probability(1), 0.5, epsilon = 1e-12);
}

// analytic: Bell state has P(|00>) = P(|11>) = 1/2, P(|01>) = P(|10>) = 0.
#[test]
fn bell_state_analytic_probabilities() {
    let mut c = Circuit::new(2);
    c.h(0).cnot(0, 1);
    let exec = StateVectorBackend::run(&c).unwrap();
    let state = exec.state();
    assert_abs_diff_eq!(state.probability(0b00), 0.5, epsilon = 1e-12);
    assert_abs_diff_eq!(state.probability(0b11), 0.5, epsilon = 1e-12);
    assert_abs_diff_eq!(state.probability(0b01), 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(state.probability(0b10), 0.0, epsilon = 1e-12);
}

proptest! {
    // after measuring ALL qubits, the state must be a computational basis state.
    #![proptest_config(ProptestConfig::with_cases(64))]
    #[test]
    fn measurement_collapses_to_basis_state(
        theta in arb_angle(),
        seed in 0u64..10000,
    ) {
        let mut c = Circuit::with_classical(2, 2);
        c.h(0).ry(1, theta).measure(0, 0).measure(1, 1);
        let exec = StateVectorBackend::run_seeded(&c, seed).unwrap();
        let state = exec.state();

        // exactly one basis state should have probability ~1.
        let probs: Vec<f64> = (0..4).map(|j| state.probability(j)).collect();
        let max_prob = probs.iter().copied().fold(0.0_f64, f64::max);
        let near_one_count = probs.iter().filter(|&&p| (p - 1.0).abs() < 1e-10).count();
        let near_zero_count = probs.iter().filter(|&&p| p < 1e-10).count();

        prop_assert!(near_one_count == 1, "expected exactly one prob~1, got probs={probs:?}");
        prop_assert!(near_zero_count == 3, "expected exactly three prob~0, got probs={probs:?}");
        assert_abs_diff_eq!(state.norm_sqr(), 1.0, epsilon = 1e-12);
        // max probability is ~1.
        assert_abs_diff_eq!(max_prob, 1.0, epsilon = 1e-10);
    }

    // measuring the same qubit twice gives the same outcome (collapse is idempotent).
    #[test]
    fn measuring_twice_agrees(theta in arb_angle(), seed in 0u64..1000) {
        let mut c = Circuit::with_classical(1, 2);
        c.h(0).ry(0, theta).measure(0, 0).measure(0, 1);
        let exec = StateVectorBackend::run_seeded(&c, seed).unwrap();
        let bits = exec.classical();
        prop_assert_eq!(bits[0], bits[1], "second measurement disagreed with first");
    }
}

// ─── F. kernel vs naive reference ────────────────────────────────────────────

// reference implementation of single-qubit gate application that does NOT use
// the bit-insertion trick: iterates every pair by testing the k-th bit.
fn naive_apply_1q(amps: &mut [Complex64], k: usize, gate: &Gate1) {
    let m = &gate.m;
    let n = amps.len();
    let mut i = 0;
    while i < n {
        if (i >> k) & 1 == 0 {
            let partner = i | (1 << k);
            let a0 = amps[i];
            let a1 = amps[partner];
            amps[i] = m[0] * a0 + m[1] * a1;
            amps[partner] = m[2] * a0 + m[3] * a1;
        }
        i += 1;
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]
    #[test]
    fn kernel_1q_matches_naive(
        (n, k) in arb_n_and_target(),
        theta in arb_angle(),
        lambda in arb_angle(),
    ) {
        // prepare a random state via ry(theta)*rz(lambda) on |0> for each qubit.
        let mut c_prep = Circuit::new(n);
        c_prep.ry(0, theta).rz(0, lambda);
        let base = StateVectorBackend::run(&c_prep).unwrap().into_state();

        let gate = Gate1::ry(theta);

        // apply via the bit-insertion kernel.
        let mut kernel_amps = base.amplitudes().to_vec();
        everett::kernel::apply_1q(&mut kernel_amps, k, &gate);

        // apply via the naive reference.
        let mut naive_amps = base.amplitudes().to_vec();
        naive_apply_1q(&mut naive_amps, k, &gate);

        for (a, b) in kernel_amps.iter().zip(&naive_amps) {
            prop_assert!(
                (*a - *b).norm() < 1e-12,
                "kernel and naive differ at amplitude: {a} vs {b}"
            );
        }
    }

    #[test]
    fn controlled_x_matches_apply_2q_cnot(state in arb_2q_state()) {
        // apply_controlled_1q with control=1, target=0, gate=X
        // must equal apply_2q with Gate2::cnot(), operands a=1, b=0.
        let mut via_controlled = state.amplitudes().to_vec();
        everett::kernel::apply_controlled_1q(&mut via_controlled, &[1], 0, &Gate1::x());

        let mut via_cnot = state.amplitudes().to_vec();
        everett::kernel::apply_2q(&mut via_cnot, 1, 0, &Gate2::cnot());

        for (a, b) in via_controlled.iter().zip(&via_cnot) {
            prop_assert!((*a - *b).norm() < 1e-12);
        }
    }
}

// ─── G. OpenQASM 3 round-trip ───────────────────────────────────────────────────

// one named-gate op over an n-qubit register. only the gates the emitter can
// name are generated, since arbitrary matrices have no QASM form.
fn arb_named_op(n: usize) -> impl Strategy<Value = Circuit> {
    (0usize..n, 0u8..11, arb_angle()).prop_map(move |(q, which, angle)| {
        let mut c = Circuit::new(n);
        match which {
            0 => c.h(q),
            1 => c.x(q),
            2 => c.y(q),
            3 => c.z(q),
            4 => c.s(q),
            5 => c.t(q),
            6 => c.rx(q, angle),
            7 => c.ry(q, angle),
            8 => c.rz(q, angle),
            9 => c.phase(q, angle),
            _ => c.gate1(Gate1::id(), q),
        };
        c
    })
}

// a random circuit of named gates on n qubits: a sequence of single-qubit ops
// plus some two-qubit gates, appended into one circuit.
fn arb_named_circuit() -> impl Strategy<Value = Circuit> {
    (2usize..=5)
        .prop_flat_map(|n| (Just(n), proptest::collection::vec(arb_named_op(n), 1..12)))
        .prop_map(|(n, parts)| {
            let mut c = Circuit::new(n);
            for part in &parts {
                c.compose(part);
            }
            // add a couple of two-qubit gates to exercise Apply2 and Controlled.
            if n >= 2 {
                c.cnot(0, 1).cz(0, 1);
                c.controlled(&[0], Gate1::x(), 1);
            }
            c
        })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]
    // emit a random named-gate circuit to QASM, parse it back, and confirm the
    // two circuits produce the same statevector (fidelity 1).
    #[test]
    fn qasm_roundtrip_preserves_state(c in arb_named_circuit()) {
        let src = c.to_qasm().expect("named-gate circuit must emit");
        let reparsed = Circuit::from_qasm(&src).expect("emitted QASM must parse");

        let s1 = StateVectorBackend::run(&c).unwrap().into_state();
        let s2 = StateVectorBackend::run(&reparsed).unwrap().into_state();
        prop_assert!(
            s1.fidelity(&s2) > 1.0 - 1e-10,
            "fidelity {} too low; qasm:\n{src}",
            s1.fidelity(&s2)
        );
    }
}
