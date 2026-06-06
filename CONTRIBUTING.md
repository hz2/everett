# Contributing to everett

Thanks for your interest in contributing.

## Ground rules

- **Zero runtime dependencies.** The default build must not pull in any runtime
  crate. `rayon` is the sole exception and only ever behind the `parallel`
  feature. Dev-dependencies (test/bench tooling) are fine.
- **`unsafe` is confined to `src/kernel.rs`.** Every `unsafe` block carries a
  `// safety:` note, a `debug_assert!` of its precondition, and a corresponding
  Kani proof or Miri test.
- **The numerics live in property tests, the index math in formal proofs.**
  Floating-point properties (unitarity, normalization) are checked with
  `proptest` and tolerances; integer/bounds properties are proven with Kani.

## Style

- Inline comments (`//`) are lowercase and terse. The exception is math
  (bit-indexing, gate matrices, Trotter steps), which may be longer.
- Doc comments (`///`, `//!`) follow standard Rust practice: a summary line,
  `# Examples` with doctests on public items, and `# Errors`/`# Panics` where
  relevant.

## Before you push

```sh
cargo fmt
cargo clippy --all-features --all-targets -- -D warnings
cargo test
cargo test --doc
```

Or, with Nix, simply `nix flake check`. For changes to `src/kernel.rs`, also
run `nix develop .#miri --command cargo miri test miri_` and
`nix develop .#kani --command cargo kani`.
