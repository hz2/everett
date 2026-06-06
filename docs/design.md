# Design

This note records the architecture of `everett` and the reasoning behind the
choices that shaped it.

## Goals

1. **Ergonomic.** Building a circuit should read like writing one on paper.
2. **Self-contained.** Zero runtime dependencies; the complex type, kernel, and
   RNG are all hand-rolled.
3. **Correct, demonstrably.** Numerics are property-tested; the `unsafe` index
   arithmetic is formally proven; the `unsafe` itself is checked by Miri.

## The layers

```
Circuit  --describes-->  Op (instruction set)
   |                         |
   |                     drive(): walks ops, owns the classical register
   v                         v
Backend trait  <--implements--  StateVectorBackend
                                    | dispatches to
                                    v
                                 kernel  --mutates-->  State (Vec<Complex64>)
```

- **`Circuit`** is pure data: an ordered `Vec<Op>` plus register sizes. Building
  it never runs anything and never fails; validation is a separate, explicit
  step. The fluent `&mut Self` methods (`c.h(0).cnot(0, 1)`) are the primary
  surface.
- **`Op`** is the instruction set: single-qubit, two-qubit, controlled,
  measure, and classical-conditional. Keeping operations as data (rather than as
  direct backend calls) makes a circuit inspectable, reusable across backends,
  and serializable later.
- **`Backend`** is the seam. It declares four primitives (apply a 1-qubit gate,
  a 2-qubit gate, a controlled gate, and measure), and a shared `drive` function
  layers circuit traversal and classical control on top. A new backend
  implements only those four methods.
- **`StateVectorBackend`** evolves a dense `State` of `2^n` amplitudes: exact
  for any circuit, but exponential in memory.
- **`StabilizerBackend`** simulates *Clifford* circuits (`H`/`S`/`CNOT`/Pauli) in
  `O(n^2)` via the Aaronson–Gottesman tableau, reaching thousands of qubits;
  non-Clifford gates are rejected with `Error::NonClifford`. It is the payoff of
  the `Backend` seam: a second backend that reuses `drive` (so measurement and
  classical control come for free) and is cross-validated against the
  statevector backend.
- **`kernel`** is the statevector numerical core (see
  [kernel-math.md](kernel-math.md)).

## Decisions

### Runtime `n`, not const generics

Qubit count is a runtime value. The tempting alternative, `State<const N:
usize>` backed by `[Complex64; 1 << N]`, requires the `generic_const_exprs`
nightly feature, which has been incomplete and unsound for years. A const marker
that *doesn't* size the array buys nothing on stable: it can't prevent an
out-of-range qubit index either. So we spend the type-system budget where it pays
off instead: `QubitId`/`ClassicalBit` newtypes, the `Op` enum, the `Backend`
trait, and `Result`-based validation. This also lets a subroutine be built at a
width chosen at runtime (a width-$k$ QFT), which a const-generic API cannot.

### Separate `Circuit` from `Backend`

A circuit is a reusable description; a backend is one way to execute it. Keeping
them apart means a circuit can be validated once, run many times, run on
different backends, and (later) serialized, none of which is possible if the
simulator state lives inside the circuit object.

### Measurement and classical control are first-class

`Op::Measure` and `Op::IfClassic` are part of the instruction set, not a
terminal-only afterthought. This is what makes teleportation and superdense
coding expressible: both branch on a mid-circuit measurement outcome and apply
feed-forward corrections (`x_if`, `z_if`). An API that only measures at the end
cannot express half of an introductory quantum-algorithms course.

### Zero runtime dependencies

`Complex64`, the kernel, and the `Rng` (a `xoshiro256++` seeded via `SplitMix64`)
are hand-rolled. This maximizes control over layout and `unsafe`, keeps the crate
trivially auditable and Miri-friendly, and suits the project's educational aim.
`rayon` is the only optional dependency, behind the off-by-default `parallel`
feature.

## Verification strategy

Correctness is checked at the layer each tool is suited to:

| Property | Tool |
| --- | --- |
| Index arithmetic in bounds, injective | Kani (bounded model checking) |
| No undefined behavior in the `unsafe` kernel | Miri |
| Gate unitarity, normalization, circuit identities | `proptest` with tolerances |
| Known-circuit outputs (Bell, teleport, QFT, …) | integration tests |

The split matters: floating-point properties (is this gate unitary?) are a poor
fit for SMT solvers and live in property tests with tolerances, while the
integer/memory-safety properties (is this index in bounds?) are exactly what
bounded model checking proves cleanly.

## Roadmap

The single-qubit kernel has a hand-rolled AVX2 + FMA fast path on `x86_64`
(`apply_1q_avx2`), chosen at runtime when the CPU supports it and the target
qubit has stride ≥ 2; the scalar path remains the fallback for other targets,
older CPUs, and Miri (which cannot execute SIMD intrinsics). The SIMD and scalar
paths are pinned equal by a dedicated equivalence test.

CNOT, CZ, and SWAP dispatch to specialized permutation/sign-flip loops in
`apply_2q` (no complex multiply; ~14–45× over the dense path). `apply_controlled_1q`
has an AVX2 + FMA fast path (the same 2-lane 2×2 FMA core as `apply_1q_avx2`),
active when `target >= 1` and no control qubit occupies bit position 0 (~2.3–13×
over scalar). Both paths are pinned equal to their scalar references by dedicated
equivalence tests; Miri exercises the scalar fallbacks.

`DensityMatrixBackend` simulates circuits as mixed states (`rho = sum_k p_k
|psi_k><psi_k|`, stored as a `2^n x 2^n` matrix) with a `NoiseModel` that applies
Kraus channels after gates and readout errors during measurement. Supported channels:
depolarizing, amplitude damping (T1), phase damping (T2), bit-flip, dephasing, and
custom Kraus operators. Ideal runs reproduce statevector backend results exactly.
Cost is O(4^n) memory and O(4^n) per gate, so practical for n <= 12 or so.

The `qasm` module reads and writes OpenQASM 3.0 over the stdgates.inc named-gate
subset. `emit` walks `circuit.ops()` and formats each into a statement (angles at
17 significant digits for exact f64 round-trip); `parse` is a hand-rolled,
zero-dependency recursive-descent lexer + parser. The parser evaluates `pi`/`tau`
angle expressions, accepts both `c[i] = measure q[j]` and `measure q[j] -> c[i]`,
takes any register identifier, and skips gate definitions and statements outside
the subset (`reset`, `barrier`). Gates built from arbitrary matrices have no QASM
name and return `Error::Qasm` on emit. Round-trip is property-tested through the
statevector backend.

Out of scope for the current release: SIMD for the general dense two-qubit path
(arbitrary non-CNOT/CZ/SWAP `Gate2`, with cross-lane 4×4 reductions for modest gain),
more algorithms (Grover, Deutsch–Jozsa, Simon, Shor), and Python bindings (PyO3).
