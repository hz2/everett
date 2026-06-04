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
Circuit  ──describes──▶  Op (instruction set)
   │                         │
   │                     drive(): walks ops, owns the classical register
   ▼                         ▼
Backend trait  ◀──implements──  StateVectorBackend
                                    │ dispatches to
                                    ▼
                                 kernel  ──mutates──▶  State (Vec<Complex64>)
```

- **`Circuit`** is pure data: an ordered `Vec<Op>` plus register sizes. Building
  it never runs anything and never fails; validation is a separate, explicit
  step. The fluent `&mut Self` methods (`c.h(0).cnot(0, 1)`) are the primary
  surface.
- **`Op`** is the instruction set — single-qubit, two-qubit, controlled,
  measure, and classical-conditional. Keeping operations as data (rather than as
  direct backend calls) makes a circuit inspectable, reusable across backends,
  and serializable later.
- **`Backend`** is the seam. It declares four primitives — apply a 1-qubit gate,
  a 2-qubit gate, a controlled gate, and measure — and a shared `drive` function
  layers circuit traversal and classical control on top. A new backend
  implements only those four methods.
- **`StateVectorBackend`** is the one v1 backend: it evolves a dense `State`.
- **`kernel`** is the numerical core (see [kernel-math.md](kernel-math.md)).

## Decisions

### Runtime `n`, not const generics

Qubit count is a runtime value. The tempting alternative — `State<const N:
usize>` backed by `[Complex64; 1 << N]` — requires the `generic_const_exprs`
nightly feature, which has been incomplete and unsound for years. A const marker
that *doesn't* size the array buys nothing on stable: it can't prevent an
out-of-range qubit index either. So we spend the type-system budget where it pays
off instead: `QubitId`/`ClassicalBit` newtypes, the `Op` enum, the `Backend`
trait, and `Result`-based validation. This also lets a subroutine be built at a
width chosen at runtime (a width-$k$ QFT), which a const-generic API cannot.

### Separate `Circuit` from `Backend`

A circuit is a reusable description; a backend is one way to execute it. Keeping
them apart means a circuit can be validated once, run many times, run on
different backends, and (later) serialized — none of which is possible if the
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
bounded model checking proves cleanly. See the parent dumping ground's
`.claude/notes/research-miri-formal.md` for the full rationale.

## Roadmap

Out of scope for v1, recorded as `// TODO` markers and in
`.claude/memory/everett-future-work.md`: a stabilizer/Clifford backend behind the
same `Backend` trait, a SIMD kernel (struct-of-arrays + `pulp`) with the scalar
path retained for Miri, more algorithms (Grover, Deutsch–Jozsa, Simon, Shor),
and circuit serialization.
