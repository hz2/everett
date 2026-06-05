//! Noise models for the density-matrix backend.
//!
//! A [`NoiseModel`] assigns noise channels to circuit operations: a gate-error
//! channel applied immediately after each gate, and a measurement-error
//! probability that flips the observed bit. Channels are expressed as Kraus
//! operators `{K_k}` satisfying `sum_k K_k† K_k = I` (trace preservation).
//!
//! # Usage
//!
//! ```
//! use everett::{Circuit, NoiseModel};
//! use everett::DensityMatrixBackend;
//!
//! let mut c = Circuit::new(1);
//! c.h(0).x(0);
//!
//! // 1 % depolarizing noise on every single-qubit gate.
//! let noise = NoiseModel::uniform_depolarizing(0.01);
//! let exec = DensityMatrixBackend::run_with_noise(&c, &noise)?;
//! # Ok::<(), everett::Error>(())
//! ```

use crate::complex::Complex64;
use crate::density::KrausOp;

/// Specifies what noise to apply after gates and during measurement.
///
/// Noise is applied per-qubit immediately after each gate that touches that
/// qubit. Only single-qubit channels are supported; two-qubit gate noise is
/// applied independently to each operand.
#[derive(Clone, Debug, Default)]
pub struct NoiseModel {
    /// Channel applied after every single-qubit gate on any qubit.
    pub after_1q: Option<Channel>,
    /// Channel applied after every two-qubit gate, to each operand.
    pub after_2q: Option<Channel>,
    /// Probability `p` of a classical bit flip in the measurement outcome.
    /// The post-measurement state is not flipped — only the recorded bit.
    pub readout_error: f64,
}

impl NoiseModel {
    /// No noise at all. Equivalent to running the statevector backend.
    #[must_use]
    pub fn ideal() -> Self {
        Self::default()
    }

    /// Uniform depolarizing noise with parameter `p` after every gate.
    ///
    /// The single-qubit depolarizing channel maps
    /// `rho -> (1-p) rho + (p/3)(X rho X + Y rho Y + Z rho Z)`.
    /// This is equivalent to applying a random Pauli error (`X`, `Y`, or `Z`)
    /// each with probability `p/3`. The Kraus representation uses four
    /// operators: `sqrt(1-p) I, sqrt(p/3) X, sqrt(p/3) Y, sqrt(p/3) Z`.
    ///
    /// # Panics
    ///
    /// Panics if `p` is not in `[0.0, 1.0]`.
    #[must_use]
    pub fn uniform_depolarizing(p: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&p),
            "depolarizing probability {p} not in [0, 1]"
        );
        let ch = Channel::Depolarizing(p);
        Self {
            after_1q: Some(ch.clone()),
            after_2q: Some(ch),
            readout_error: 0.0,
        }
    }

    /// Amplitude-damping noise (T1 relaxation) with parameter `gamma`.
    ///
    /// Models energy loss: `|1> -> |0>` with probability `gamma`. The Kraus
    /// operators are `K0 = [[1,0],[0,sqrt(1-gamma)]]` and
    /// `K1 = [[0,sqrt(gamma)],[0,0]]`.
    ///
    /// # Panics
    ///
    /// Panics if `gamma` is not in `[0.0, 1.0]`.
    #[must_use]
    pub fn amplitude_damping(gamma: f64) -> Self {
        assert!((0.0..=1.0).contains(&gamma), "gamma {gamma} not in [0, 1]");
        let ch = Channel::AmplitudeDamping(gamma);
        Self {
            after_1q: Some(ch.clone()),
            after_2q: Some(ch),
            readout_error: 0.0,
        }
    }

    /// Phase-damping noise (T2 dephasing) with parameter `lambda`.
    ///
    /// Models dephasing without energy exchange: off-diagonal elements decay.
    /// Kraus operators: `K0 = [[1,0],[0,sqrt(1-lambda)]]`,
    /// `K1 = [[0,0],[0,sqrt(lambda)]]`.
    ///
    /// # Panics
    ///
    /// Panics if `lambda` is not in `[0.0, 1.0]`.
    #[must_use]
    pub fn phase_damping(lambda: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&lambda),
            "lambda {lambda} not in [0, 1]"
        );
        let ch = Channel::PhaseDamping(lambda);
        Self {
            after_1q: Some(ch),
            after_2q: None,
            readout_error: 0.0,
        }
    }

    /// A bit-flip channel with probability `p`: applies X with probability `p`.
    ///
    /// Kraus operators: `sqrt(1-p) I`, `sqrt(p) X`.
    ///
    /// # Panics
    ///
    /// Panics if `p` is not in `[0.0, 1.0]`.
    #[must_use]
    pub fn bit_flip(p: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&p),
            "bit-flip probability {p} not in [0, 1]"
        );
        let ch = Channel::BitFlip(p);
        Self {
            after_1q: Some(ch),
            after_2q: None,
            readout_error: 0.0,
        }
    }

    /// A dephasing (phase-flip) channel with probability `p`: applies Z with
    /// probability `p`. Kraus operators: `sqrt(1-p) I`, `sqrt(p) Z`.
    ///
    /// # Panics
    ///
    /// Panics if `p` is not in `[0.0, 1.0]`.
    #[must_use]
    pub fn dephasing(p: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&p),
            "dephasing probability {p} not in [0, 1]"
        );
        let ch = Channel::Dephasing(p);
        Self {
            after_1q: Some(ch),
            after_2q: None,
            readout_error: 0.0,
        }
    }
}

/// A quantum noise channel, parameterized by a single error probability.
#[derive(Clone, Debug)]
pub enum Channel {
    /// `rho -> (1-p) rho + (p/3)(X rho X + Y rho Y + Z rho Z)`.
    Depolarizing(f64),
    /// Energy relaxation: `|1> -> |0>` with probability `gamma`.
    AmplitudeDamping(f64),
    /// Pure dephasing: off-diagonal terms decay by `sqrt(1-lambda)`.
    PhaseDamping(f64),
    /// Bit-flip: applies X with probability `p`.
    BitFlip(f64),
    /// Phase-flip: applies Z with probability `p`.
    Dephasing(f64),
    /// User-supplied Kraus operators (must satisfy `sum K† K = I`).
    Custom(Vec<KrausOp>),
}

impl Channel {
    /// Expand the channel into its Kraus operators.
    #[must_use]
    pub fn kraus_ops(&self) -> Vec<KrausOp> {
        let c = |re, im| Complex64::new(re, im);
        match *self {
            Self::Depolarizing(p) => {
                let a = (1.0 - p).sqrt();
                let b = (p / 3.0).sqrt();
                vec![
                    // sqrt(1-p) * I
                    KrausOp([c(a, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(a, 0.0)]),
                    // sqrt(p/3) * X
                    KrausOp([c(0.0, 0.0), c(b, 0.0), c(b, 0.0), c(0.0, 0.0)]),
                    // sqrt(p/3) * Y
                    KrausOp([c(0.0, 0.0), c(0.0, -b), c(0.0, b), c(0.0, 0.0)]),
                    // sqrt(p/3) * Z
                    KrausOp([c(b, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(-b, 0.0)]),
                ]
            }
            Self::AmplitudeDamping(gamma) => {
                let k0_11 = (1.0 - gamma).sqrt();
                let k1_01 = gamma.sqrt();
                vec![
                    // K0 = [[1, 0], [0, sqrt(1-gamma)]]
                    KrausOp([c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(k0_11, 0.0)]),
                    // K1 = [[0, sqrt(gamma)], [0, 0]]
                    KrausOp([c(0.0, 0.0), c(k1_01, 0.0), c(0.0, 0.0), c(0.0, 0.0)]),
                ]
            }
            Self::PhaseDamping(lambda) => {
                let k0_11 = (1.0 - lambda).sqrt();
                let k1_11 = lambda.sqrt();
                vec![
                    // K0 = [[1, 0], [0, sqrt(1-lambda)]]
                    KrausOp([c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(k0_11, 0.0)]),
                    // K1 = [[0, 0], [0, sqrt(lambda)]]
                    KrausOp([c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(k1_11, 0.0)]),
                ]
            }
            Self::BitFlip(p) => {
                let a = (1.0 - p).sqrt();
                let b = p.sqrt();
                vec![
                    KrausOp([c(a, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(a, 0.0)]),
                    KrausOp([c(0.0, 0.0), c(b, 0.0), c(b, 0.0), c(0.0, 0.0)]),
                ]
            }
            Self::Dephasing(p) => {
                let a = (1.0 - p).sqrt();
                let b = p.sqrt();
                vec![
                    KrausOp([c(a, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(a, 0.0)]),
                    KrausOp([c(b, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(-b, 0.0)]),
                ]
            }
            Self::Custom(ref ops) => ops.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // validate that the Kraus completeness condition holds: sum_k K_k† K_k = I.
    fn kraus_complete(ops: &[KrausOp]) -> bool {
        let mut sum = [Complex64::ZERO; 4]; // 2x2, row-major
        for k in ops {
            let m = &k.0;
            // K† K: (K†)_{ij} = conj(K_{ji}), so (K† K)_{ij} = sum_l conj(K_{li}) K_{lj}.
            for i in 0..2 {
                for j in 0..2 {
                    for l in 0..2 {
                        sum[i * 2 + j] += m[l * 2 + i].conj() * m[l * 2 + j];
                    }
                }
            }
        }
        let identity = [
            Complex64::ONE,
            Complex64::ZERO,
            Complex64::ZERO,
            Complex64::ONE,
        ];
        sum.iter()
            .zip(&identity)
            .all(|(a, b)| (*a - *b).norm() < 1e-12)
    }

    #[test]
    fn depolarizing_kraus_complete() {
        for p in [0.0, 0.01, 0.1, 0.3, 0.75] {
            let ops = Channel::Depolarizing(p).kraus_ops();
            assert!(kraus_complete(&ops), "depolarizing p={p} not complete");
        }
    }

    #[test]
    fn amplitude_damping_kraus_complete() {
        for gamma in [0.0, 0.05, 0.5, 1.0] {
            let ops = Channel::AmplitudeDamping(gamma).kraus_ops();
            assert!(
                kraus_complete(&ops),
                "amplitude_damping gamma={gamma} not complete"
            );
        }
    }

    #[test]
    fn phase_damping_kraus_complete() {
        for lambda in [0.0, 0.1, 0.9, 1.0] {
            let ops = Channel::PhaseDamping(lambda).kraus_ops();
            assert!(
                kraus_complete(&ops),
                "phase_damping lambda={lambda} not complete"
            );
        }
    }

    #[test]
    fn bit_flip_kraus_complete() {
        for p in [0.0, 0.2, 0.5, 1.0] {
            let ops = Channel::BitFlip(p).kraus_ops();
            assert!(kraus_complete(&ops), "bit_flip p={p} not complete");
        }
    }

    #[test]
    fn dephasing_kraus_complete() {
        for p in [0.0, 0.3, 1.0] {
            let ops = Channel::Dephasing(p).kraus_ops();
            assert!(kraus_complete(&ops), "dephasing p={p} not complete");
        }
    }
}
