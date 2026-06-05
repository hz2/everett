# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial statevector simulator scaffolding.
- Hand-rolled `Complex64` amplitude type.
- Bit-insertion gate-application kernel.
- Fluent `Circuit` builder over a backend-agnostic execution seam.
- Universal single-qubit gate set, two-qubit gates, and controlled gates.
- Born-rule measurement with mid-circuit classical control.
- Worked algorithms: Bell, GHZ, teleportation, superdense coding, QFT, phase
  estimation, and Hamiltonian simulation.
- `unsafe` fast path in the kernel, justified by Kani proofs of the index
  arithmetic (bounds + injectivity) and validated for undefined behavior by Miri
  under both the Stacked Borrows and Tree Borrows aliasing models.
- Hand-rolled AVX2 + FMA fast path for single-qubit gate application on
  `x86_64` (no external dependency), selected at runtime with a scalar fallback
  for other targets and under Miri. ~7–28× faster than the scalar kernel,
  depending on qubit count.
- Criterion benchmark suite covering gate application across qubit counts and a
  full QFT circuit.
- Specialized fast paths for CNOT, CZ, and SWAP in `apply_2q`: exact-match
  dispatch to tight permutation/sign-flip loops that need no complex multiply.
  ~14–45× faster than the general dense path (n=22 memory-bound to n=8
  compute-bound).
- AVX2 + FMA fast path for `apply_controlled_1q` on `x86_64`, selected at
  runtime when `target >= 1` and no control bit occupies position 0 (same
  2-lane FMA core as `apply_1q_avx2`, with a per-block control test). ~2.3–13×
  faster than the scalar path. Scalar fallback retained for other targets, older
  CPUs, qubit/control configurations outside the condition, and Miri.
  The general dense `apply_2q` (arbitrary non-CNOT/CZ/SWAP gates) remains
  scalar; that path is rare in real circuits and the cross-lane 4×4 reductions
  would yield modest gain for significant complexity.
- `StabilizerBackend`: a Clifford-circuit backend using the Aaronson–Gottesman
  tableau, simulating `H`/`S`/`CNOT`/Pauli circuits and measurement in `O(n^2)`
  — thousands of qubits, far beyond the statevector backend. Non-Clifford gates
  are rejected with `Error::NonClifford`. Cross-validated against the statevector
  backend (the statevector is a `+1` eigenstate of every reported stabilizer
  generator) over random Clifford circuits.
- `PauliString` type with Pauli-operator expectation values.
- `DensityMatrix` type: a `2^n x 2^n` complex matrix representing mixed states,
  supporting unitary evolution (`rho -> U rho U†`), quantum channels (Kraus
  operators, `rho -> sum_k K_k rho K_k†`), projective measurement with collapse
  and renormalization, Pauli expectation values, and single-qubit partial trace.
- `NoiseModel` and `Channel` types: built-in channels for depolarizing noise,
  amplitude damping (T1), phase damping (T2), bit-flip, and dephasing, plus
  custom Kraus operators. All satisfy the trace-preservation completeness
  condition (`sum_k K_k† K_k = I`), validated by unit tests.
- `DensityMatrixBackend`: implements the `Backend` trait, running circuits as
  density matrices with per-gate and readout noise from a `NoiseModel`. Ideal
  runs (no noise) reproduce statevector backend probabilities exactly. Cross-
  validated against `StateVectorBackend` for all circuits up to n=4.

### Planned (future work)

- Python bindings via PyO3 (`pip install everett`): expose `Circuit`, `NoiseModel`,
  and all backends with numpy amplitude arrays. Zero runtime-dep constraint is
  preserved (PyO3 is a build dependency only).
- OpenQASM 3.0 import/export: interoperability with Qiskit, Cirq, and IBM/Google
  hardware — circuits designed in any standard toolchain can run on everett.
