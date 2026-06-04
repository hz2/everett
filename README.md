# everett

A clean, zero-dependency statevector quantum simulator in Rust.

`everett` represents the quantum state of `n` qubits as a complex vector of
dimension `2^n` and applies gates in place over amplitude index-groups — never
by materializing a `2^n x 2^n` matrix. The name is a nod to the many-worlds
interpretation: a statevector holds the entire branching wavefunction at once.

## Goals

- **Ergonomic.** A fluent circuit builder backed by a rich, type-safe core.
- **Self-contained.** Zero runtime dependencies. The complex-number type, the
  numerical kernel, and the RNG are all hand-rolled.
- **Correct.** A tiered verification strategy: property tests for the numerics,
  bounded formal proofs (Kani) for the index arithmetic, and Miri for the
  `unsafe` kernel.

## Example

```rust
use everett::prelude::*;

// prepare a Bell state: (|00> + |11>) / sqrt(2)
let mut circuit = Circuit::new(2);
circuit.h(0).cnot(0, 1);

let state = StateVectorBackend::run(&circuit)?;
assert!((state.probability(0b00) - 0.5).abs() < 1e-12);
assert!((state.probability(0b11) - 0.5).abs() < 1e-12);
# Ok::<(), everett::Error>(())
```

## What's included (v1)

- A universal gate set (`H`, `X`, `Y`, `Z`, `S`, `T`, rotations, phase) plus
  two-qubit gates (`CNOT`, `CZ`, `SWAP`) and arbitrary controlled gates.
- Projective measurement with Born-rule sampling and mid-circuit,
  classically-controlled operations.
- Worked algorithms: Bell/GHZ state preparation, teleportation, superdense
  coding, the quantum Fourier transform, phase estimation, and Hamiltonian
  simulation via Trotterization.

## Development

This project uses Nix flakes. The default shell provides the stable toolchain;
separate shells provide the nightly toolchain for Miri and the Kani verifier.

```sh
nix develop            # stable toolchain + rust-analyzer
nix flake check        # build, clippy, fmt, tests, doctests

nix develop .#miri     # nightly + miri, for undefined-behavior checking
nix develop .#kani     # kani, for bounded formal verification
```

Without Nix, a standard `cargo` toolchain (1.95+) works for everything except
the Miri and Kani steps.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
