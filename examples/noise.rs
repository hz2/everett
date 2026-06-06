//! Demonstrates the density-matrix backend with noise models.
//!
//! Shows how a Bell state degrades under depolarizing noise, amplitude
//! damping, and phase damping, by tracking purity and expectation values.

use everett::algorithms::prep;
use everett::{Circuit, DensityMatrixBackend, NoiseModel};

fn main() -> everett::Result<()> {
    // ideal Bell state
    let bell = prep::bell();
    let ideal = DensityMatrixBackend::run(&bell)?;
    let rho_ideal = ideal.density_matrix();
    println!("ideal Bell state:");
    println!("  purity:      {:.6}", rho_ideal.purity());
    println!("  P(|00>):     {:.6}", rho_ideal.probability(0b00));
    println!("  P(|11>):     {:.6}", rho_ideal.probability(0b11));
    println!();

    // depolarizing noise
    for &p in &[0.01f64, 0.05, 0.10] {
        let noise = NoiseModel::uniform_depolarizing(p);
        let exec = DensityMatrixBackend::run_with_noise(&bell, &noise)?;
        let rho = exec.density_matrix();
        println!(
            "depolarizing p={p:.2}: purity={:.4}  P(00)={:.4}  P(11)={:.4}",
            rho.purity(),
            rho.probability(0b00),
            rho.probability(0b11)
        );
    }
    println!();

    // amplitude damping (T1 relaxation)
    // prepare |1> then damp; expect population to leak back to |0>.
    let mut c = Circuit::new(1);
    c.x(0);
    for &gamma in &[0.0f64, 0.1, 0.5, 0.9] {
        let noise = NoiseModel::amplitude_damping(gamma);
        let exec = DensityMatrixBackend::run_with_noise(&c, &noise)?;
        let rho = exec.density_matrix();
        println!(
            "amplitude damping gamma={gamma:.1}: P(|0>)={:.4}  P(|1>)={:.4}",
            rho.probability(0),
            rho.probability(1)
        );
    }
    println!();

    // dephasing (T2): |+> coherence decays
    let mut c = Circuit::new(1);
    c.h(0);
    println!("dephasing of |+> state (off-diagonal magnitude):");
    for &p in &[0.0f64, 0.1, 0.3, 0.5] {
        let noise = NoiseModel::dephasing(p);
        let exec = DensityMatrixBackend::run_with_noise(&c, &noise)?;
        let rho = exec.density_matrix();
        println!(
            "  p={p:.1}: |rho_01|={:.4}  <X>={:.4}",
            rho.get(0, 1).norm(),
            rho.expectation_pauli(0, 'X')
        );
    }

    Ok(())
}
