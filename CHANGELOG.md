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
