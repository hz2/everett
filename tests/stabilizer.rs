//! Integration tests for the stabilizer backend.
//!
//! The central check ties the stabilizer backend to the already-trusted
//! statevector backend: for any Clifford circuit, the statevector it produces
//! must be a `+1` eigenstate of every stabilizer generator the tableau reports.
//! This validates the tableau's signs and phases without depending on
//! measurement randomness.

use everett::{Circuit, Rng, StabilizerBackend, StateVectorBackend};

// asserts that the statevector and stabilizer backends agree on `circuit`:
// every generator from the tableau has expectation +1 on the statevector.
fn assert_backends_agree(circuit: &Circuit) {
    let sv = StateVectorBackend::run(circuit).unwrap().into_state();
    let stab = StabilizerBackend::run(circuit).unwrap();
    for generator in stab.generators() {
        let exp = generator.expectation(&sv);
        assert!(
            (exp - 1.0).abs() < 1e-9,
            "generator {generator} has expectation {exp}, expected +1"
        );
    }
    // n generators for an n-qubit pure stabilizer state.
    assert_eq!(stab.generators().len(), circuit.num_qubits());
}

#[test]
fn bell_state_agrees() {
    let mut c = Circuit::new(2);
    c.h(0).cnot(0, 1);
    assert_backends_agree(&c);
}

#[test]
fn ghz_states_agree() {
    for n in 2..=8 {
        let mut c = Circuit::new(n);
        c.h(0);
        for k in 0..n - 1 {
            c.cnot(k, k + 1);
        }
        assert_backends_agree(&c);
    }
}

#[test]
fn all_clifford_generators_agree() {
    // exercise every Clifford gate the backend recognizes.
    let mut c = Circuit::new(3);
    c.h(0)
        .s(1)
        .z(2)
        .x(0)
        .y(1)
        .cnot(0, 1)
        .cz(1, 2)
        .swap(0, 2)
        .s(0)
        .h(2);
    assert_backends_agree(&c);
}

#[test]
fn phase_equivalent_gates_agree() {
    // rz(pi/2) == S and rz(pi) == Z up to global phase: the stabilizer backend
    // must treat them identically to the named gates.
    use std::f64::consts::PI;
    let mut c = Circuit::new(2);
    c.h(0).rz(0, PI / 2.0).cnot(0, 1).rz(1, PI);
    assert_backends_agree(&c);
}

#[test]
fn random_clifford_circuits_agree() {
    // build pseudo-random Clifford circuits and check both backends agree.
    let mut rng = Rng::seed_from_u64(0xC11F_F09D_5E00);
    for _ in 0..50 {
        let n = 2 + (rng.next_u64() % 5) as usize; // 2..=6 qubits
        let depth = 10 + (rng.next_u64() % 20) as usize;
        let mut c = Circuit::new(n);
        for _ in 0..depth {
            match rng.next_u64() % 6 {
                0 => {
                    c.h((rng.next_u64() as usize) % n);
                }
                1 => {
                    c.s((rng.next_u64() as usize) % n);
                }
                2 => {
                    c.x((rng.next_u64() as usize) % n);
                }
                3 => {
                    c.z((rng.next_u64() as usize) % n);
                }
                4 => {
                    // a CNOT on two distinct qubits.
                    let a = (rng.next_u64() as usize) % n;
                    let b = (a + 1 + (rng.next_u64() as usize) % (n - 1)) % n;
                    c.cnot(a, b);
                }
                _ => {
                    let a = (rng.next_u64() as usize) % n;
                    let b = (a + 1 + (rng.next_u64() as usize) % (n - 1)) % n;
                    c.cz(a, b);
                }
            }
        }
        assert_backends_agree(&c);
    }
}

#[test]
fn teleportation_runs_on_stabilizer_backend() {
    // teleporting a Clifford state (|+>, prepared with H) uses only Clifford
    // gates plus measurement and classical control, so it runs on the
    // stabilizer backend. regardless of the two (random) measurement outcomes,
    // the feed-forward corrections leave qubit 2 in |+>. we confirm this
    // physically across many seeds: rotating qubit 2 to the Z basis (H) and
    // measuring it must deterministically yield 0 if it was truly |+>.
    for seed in 0..32 {
        let mut c = Circuit::with_classical(3, 3);
        c.h(0); // message = |+>
        c.h(1).cnot(1, 2); // bell pair on (1,2)
        c.cnot(0, 1).h(0); // bell measurement of (0,1)
        c.measure(0, 0).measure(1, 1);
        c.x_if(1, 2).z_if(0, 2); // feed-forward corrections
        // read out qubit 2 in the X basis: H then measure. |+> -> |0> -> 0.
        c.h(2).measure(2, 2);

        let stab = StabilizerBackend::run_seeded(&c, seed).unwrap();
        assert!(
            !stab.classical()[2],
            "seed {seed}: qubit 2 not in |+> (X-basis readout was 1)"
        );
    }
}

#[test]
fn non_clifford_circuit_is_rejected() {
    // a circuit with a T gate must be refused by the stabilizer backend, while
    // the statevector backend handles it fine.
    let mut c = Circuit::new(1);
    c.h(0).t(0);
    assert!(StabilizerBackend::run(&c).is_err());
    assert!(StateVectorBackend::run(&c).is_ok());
}
