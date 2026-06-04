//! Teleportation example: teleport the |+> state and confirm qubit 2 holds it.

use everett::Complex64;
use everett::algorithms::teleport;
use everett::prelude::*;

fn main() -> everett::Result<()> {
    let s = 1.0_f64 / 2.0_f64.sqrt();
    // message: |+> = (|0> + |1>) / sqrt(2), Bloch vector (1, 0, 0).
    let circuit = teleport::teleport_state(Complex64::new(s, 0.0), Complex64::new(s, 0.0));
    let exec = StateVectorBackend::run(&circuit)?;
    let [x, y, z] = exec.state().bloch_vector(2);
    println!("Teleportation of |+>:");
    println!("  qubit 2 Bloch vector: ({x:.6}, {y:.6}, {z:.6})");
    println!("  expected:             ( 1.000000,  0.000000,  0.000000)");
    println!("  classical bits: {:?}", exec.classical());
    Ok(())
}
