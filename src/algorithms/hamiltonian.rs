//! Hamiltonian simulation via first-order Trotterization.
// LaTeX math notation is intentional in this module's docs.
#![allow(clippy::doc_markdown)]
//!
//! # Overview
//!
//! For a Hamiltonian $H = A + B$ and time $t$, the Trotter approximation is
//!
//! $$e^{-iHt} \approx \left(e^{-iAt/r}\, e^{-iBt/r}\right)^r$$
//!
//! where $r$ is the number of Trotter steps. The error is $O(t^2/r)$, so
//! increasing $r$ improves fidelity.
//!
//! # Transverse-field Ising model (TFIM)
//!
//! The TFIM on $n$ qubits with periodic boundary conditions is
//!
//! $$H = -J_{zz}\sum_{i=0}^{n-1} Z_i Z_{i+1} - h_x \sum_{i=0}^{n-1} X_i$$
//!
//! where indices wrap ($Z_{n} \equiv Z_0$).
//!
//! ## Trotter decomposition
//!
//! Split $H = H_{ZZ} + H_X$ where
//! $H_{ZZ} = -J_{zz}\sum_i Z_i Z_{i+1}$ and
//! $H_X = -h_x \sum_i X_i$.
//!
//! **X-field term** ($e^{-i\delta t H_X}$): each site independently:
//! $$e^{i h_x \delta t X_i} = R_x(-2 h_x \delta t)$$
//! (using the convention $R_x(\theta) = e^{-i\theta X/2}$, so
//! $e^{i h_x \delta t X} = R_x(-2 h_x \delta t)$).
//!
//! **ZZ-coupling term** ($e^{-i\delta t H_{ZZ}}$): for each pair $(i, i+1)$:
//! $$e^{i J_{zz} \delta t Z_i Z_{i+1}} = \text{CNOT}(i, i{+}1)\; R_z(-2 J_{zz} \delta t)\text{ on } i{+}1\; \text{CNOT}(i, i{+}1)$$
//!
//! The CNOT pair maps $Z_i Z_{i+1}$ to $Z_{i+1}$ on the second qubit, applies
//! $R_z$, then maps back.

use crate::circuit::Circuit;

/// Simulates $e^{-iZt}$ on a single qubit via Trotterisation.
///
/// Since there is only one term, Trotter is exact at any number of steps:
/// $r$ steps of $R_z(2t/r)$ compose to $R_z(2t)$. This function is mainly
/// useful for testing the Trotter scaffolding.
///
/// # Examples
///
/// ```
/// use everett::{StateVectorBackend, algorithms::hamiltonian};
///
/// let c = hamiltonian::trotter_single_z(0.5, 4);
/// let exec = StateVectorBackend::run(&c).unwrap();
/// assert!((exec.state().norm_sqr() - 1.0).abs() < 1e-12);
/// ```
#[must_use]
pub fn trotter_single_z(t: f64, steps: usize) -> Circuit {
    let mut c = Circuit::new(1);
    // e^{-iZt} = Rz(2t) up to global phase; split into `steps` sub-steps.
    let dt = t / steps as f64;
    for _ in 0..steps {
        c.rz(0, 2.0 * dt);
    }
    c
}

/// Trotterised time-evolution of the transverse-field Ising model.
///
/// Simulates $e^{-iHt}$ where
/// $H = -J_{zz}\sum_{i} Z_i Z_{i+1} - h_x \sum_i X_i$
/// using `steps` first-order Trotter steps. Boundary conditions are open
/// (pairs $0\text{--}1$, $1\text{--}2$, …, $(n-2)\text{--}(n-1)$).
///
/// See the module documentation for the explicit gate decompositions.
///
/// # Panics
///
/// Panics if `n == 0`.
///
/// # Examples
///
/// ```
/// use everett::{StateVectorBackend, algorithms::hamiltonian};
///
/// // 2-qubit TFIM, purely transverse field, short time
/// let c = hamiltonian::trotter_tfim(2, 0.0, 1.0, 0.1, 10);
/// let exec = StateVectorBackend::run(&c).unwrap();
/// assert!((exec.state().norm_sqr() - 1.0).abs() < 1e-12);
/// ```
#[must_use]
pub fn trotter_tfim(n: usize, jzz: f64, hx: f64, t: f64, steps: usize) -> Circuit {
    assert!(n >= 1);
    let mut c = Circuit::new(n);
    let dt = t / steps as f64;

    for _ in 0..steps {
        // --- X-field layer: e^{i hx dt X_i} = Rx(-2 hx dt) on each qubit ---
        // Rx(θ) = e^{-iθX/2}, so e^{i hx dt X} = Rx(-2 hx dt).
        let rx_angle = -2.0 * hx * dt;
        for i in 0..n {
            c.rx(i, rx_angle);
        }

        // --- ZZ-coupling layer: for each neighbouring pair (i, i+1) ---
        // e^{i Jzz dt Z_i Z_{i+1}} via CNOT-Rz-CNOT decomposition:
        //   CNOT maps Z_i⊗Z_{i+1} -> I⊗Z_{i+1},
        //   then Rz(-2 Jzz dt) on qubit i+1 applies the phase,
        //   then CNOT maps back.
        let zz_angle = -2.0 * jzz * dt;
        for i in 0..n - 1 {
            c.cnot(i, i + 1);
            c.rz(i + 1, zz_angle);
            c.cnot(i, i + 1);
        }
    }

    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StateVectorBackend;
    use crate::circuit::Circuit;
    use crate::state::State;

    fn run(c: &Circuit) -> State {
        StateVectorBackend::run(c).unwrap().into_state()
    }

    #[test]
    fn single_z_norm_preserved() {
        for steps in [1, 5, 20] {
            let state = run(&trotter_single_z(1.2, steps));
            assert!((state.norm_sqr() - 1.0).abs() < 1e-12, "steps={steps}");
        }
    }

    #[test]
    fn pure_x_field_matches_exact_rx() {
        // H = -hx X; e^{-iHt} = e^{i hx t X} = Rx(-2 hx t).
        let hx = 0.8;
        let t = 0.5;
        let steps = 200;

        // Trotter evolution (no ZZ coupling)
        let trotter_state = run(&trotter_tfim(1, 0.0, hx, t, steps));

        // exact: a single Rx(-2*hx*t) gate
        let mut exact = Circuit::new(1);
        exact.rx(0, -2.0 * hx * t);
        let exact_state = run(&exact);

        // fidelity should be close to 1 at 200 steps
        assert!(
            trotter_state.fidelity(&exact_state) > 0.999,
            "fidelity = {}",
            trotter_state.fidelity(&exact_state)
        );
    }

    #[test]
    fn tfim_norm_preserved() {
        // any unitary preserves norm; Trotter is approximately unitary.
        let state = run(&trotter_tfim(4, 1.0, 0.5, 1.0, 100));
        assert!((state.norm_sqr() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn trotter_converges_as_steps_increase() {
        // increasing steps should improve fidelity between coarse and fine runs.
        let (jzz, hx, t) = (0.5, 1.0, 0.3);

        let coarse = run(&trotter_tfim(2, jzz, hx, t, 10));
        let fine = run(&trotter_tfim(2, jzz, hx, t, 500));

        // fine vs coarse fidelity must exceed some threshold
        let f = coarse.fidelity(&fine);
        assert!(f > 0.99, "fidelity coarse vs fine = {f}");
    }
}
