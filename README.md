# everett

A clean, zero-dependency statevector quantum simulator in Rust.

`everett` represents the quantum state of `n` qubits as a complex vector of
`2^n` amplitudes and applies gates in place over amplitude index-groups, never
by materializing a full `2^n x 2^n` matrix. The name is a nod to the
many-worlds interpretation: a statevector holds the entire branching wavefunction
at once.

## Example

```rust
use everett::prelude::*;

// Bell state: (|00> + |11>) / sqrt(2)
let mut circuit = Circuit::new(2);
circuit.h(0).cnot(0, 1);

let exec = StateVectorBackend::run(&circuit)?;
let state = exec.state();
assert!((state.probability(0b00) - 0.5).abs() < 1e-12);
assert!((state.probability(0b11) - 0.5).abs() < 1e-12);
# Ok::<(), everett::Error>(())
```

## Features

**Three backends**

| Backend | State representation | Best for |
|---|---|---|
| `StateVectorBackend` | `2^n` amplitudes | general simulation up to ~30 qubits |
| `DensityMatrixBackend` | `4^n` density matrix | noise simulation, mixed states |
| `StabilizerBackend` | `O(n^2)` tableau | Clifford circuits on hundreds of qubits |

**Gate set** -- single-qubit: `H`, `X`, `Y`, `Z`, `S`, `T`, `Rx/Ry/Rz`,
`Phase`; two-qubit: `CNOT`, `CZ`, `SWAP`; plus arbitrary controlled gates.

**Measurement** -- projective measurement with Born-rule sampling and
mid-circuit classically-controlled operations.

**Noise** -- uniform depolarizing, Kraus-operator channels, and readout errors
via `NoiseModel`, applied through `DensityMatrixBackend`.

**OpenQASM 3.0** -- import and export circuits with `Circuit::from_qasm` /
`Circuit::to_qasm`.

**Algorithms** -- Bell/GHZ state preparation, quantum teleportation, superdense
coding, quantum Fourier transform, phase estimation, and Hamiltonian simulation
via Trotterization.

## Optional: parallelism

Gate application can be parallelized across amplitude chunks with
[rayon](https://github.com/rayon-rs/rayon). This is off by default to keep the
core dependency-free and Miri-friendly.

```toml
everett = { version = "0.1", features = ["parallel"] }
```

## Development

This project uses Nix flakes. The default shell provides the stable toolchain;
separate shells provide nightly for Miri and the Kani verifier.

```sh
nix develop            # stable toolchain + rust-analyzer
nix flake check        # build, clippy, fmt, tests, doctests

nix develop .#miri     # nightly + miri
nix develop .#kani     # kani bounded formal verification
```

Without Nix, a standard `cargo` toolchain (1.85+) works for everything except
Miri and Kani.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
