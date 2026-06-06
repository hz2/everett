//! Stabilizer backend: simulate Clifford circuits far beyond statevector reach.
//!
//! Run with `cargo run --example stabilizer`.

use everett::{Circuit, StabilizerBackend};

fn main() {
    // a 1000-qubit GHZ state. the statevector backend would need 2^1000
    // amplitudes, more than the number of atoms in the universe. the stabilizer
    // backend handles it in O(n^2) bits.
    let n = 1000;
    let mut ghz = Circuit::new(n);
    ghz.h(0);
    for k in 0..n - 1 {
        ghz.cnot(k, k + 1);
    }

    let exec = StabilizerBackend::run(&ghz).expect("GHZ is a Clifford circuit");
    println!("{n}-qubit GHZ state prepared on the stabilizer backend.");
    println!(
        "stabilizer group has {} generators.",
        exec.generators().len()
    );
    println!("first generator:  {}", exec.generators()[0]);
    println!("second generator: {}", exec.generators()[1]);

    // measure all qubits: a GHZ state collapses to all-zeros or all-ones, so
    // every measured bit must agree.
    let mut measured = Circuit::with_classical(n, n);
    measured.compose(&ghz);
    for q in 0..n {
        measured.measure(q, q);
    }
    let exec = StabilizerBackend::run_seeded(&measured, 1).expect("still Clifford");
    let bits = exec.classical();
    let all_same = bits.iter().all(|&b| b == bits[0]);
    println!(
        "measured all {n} qubits: {} (GHZ correlation holds: {all_same})",
        if bits[0] { "all ones" } else { "all zeros" },
    );

    // a non-Clifford gate is cleanly rejected.
    let mut with_t = Circuit::new(1);
    with_t.t(0);
    match StabilizerBackend::run(&with_t) {
        Err(e) => println!("\nas expected, a T gate is refused: {e}"),
        Ok(_) => unreachable!("T is not Clifford"),
    }
}
