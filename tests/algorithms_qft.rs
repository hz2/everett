//! Integration tests for the quantum Fourier transform.

use everett::algorithms::qft;
use everett::{Circuit, Complex64, State, StateVectorBackend};
use std::f64::consts::TAU;

fn run(c: &Circuit) -> State {
    StateVectorBackend::run(c).unwrap().into_state()
}

#[test]
fn qft_of_zero_is_uniform_for_n_1_to_5() {
    for n in 1..=5 {
        let state = run(&qft::qft(n));
        let expected_prob = 1.0 / (1usize << n) as f64;
        for basis in 0..(1usize << n) {
            assert!(
                (state.probability(basis) - expected_prob).abs() < 1e-12,
                "n={n} basis={basis}: got {}",
                state.probability(basis)
            );
        }
    }
}

#[test]
fn qft_then_iqft_is_identity_on_basis_states() {
    let n = 4;
    for input_basis in [0usize, 1, 3, 7, 8, 15] {
        let mut amps = vec![Complex64::ZERO; 1 << n];
        amps[input_basis] = Complex64::ONE;
        let input = State::from_amplitudes(amps).unwrap();

        let mut c = Circuit::new(n);
        for bit in 0..n {
            if input_basis & (1 << bit) != 0 {
                c.x(bit);
            }
        }
        c.compose(&qft::qft(n)).compose(&qft::iqft(n));

        let output = run(&c);
        assert!(
            output.fidelity(&input) > 1.0 - 1e-12,
            "basis={input_basis}, fidelity={}",
            output.fidelity(&input)
        );
    }
}

#[test]
fn qft_of_basis_one_matches_closed_form() {
    // QFT |1⟩ = (1/√2^n) Σ_k e^{2πi·1·k/2^n} |k⟩
    // = (1/√2^n) Σ_k e^{2πik/2^n} |k⟩
    let n = 3;
    let nn = 1usize << n;

    let mut c = Circuit::new(n);
    c.x(0); // prepare |1⟩ (qubit 0 is the LSB)
    c.compose(&qft::qft(n));
    let state = run(&c);

    let scale = 1.0 / (nn as f64).sqrt();
    for k in 0..nn {
        // j=1 (input basis), so phase = 2πi·1·k/2^n
        let expected = Complex64::expi(TAU * k as f64 / nn as f64) * scale;
        let got = state.amplitudes()[k];
        let diff = (got - expected).norm();
        assert!(
            diff < 1e-12,
            "k={k}: expected {expected}, got {got}, diff={diff}"
        );
    }
}
