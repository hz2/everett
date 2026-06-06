---
name: project-overview
description: what everett is, its published state, and key crate metadata
metadata:
  type: project
---

everett is a zero-dependency statevector quantum simulator in Rust, published at crates.io/crates/everett.

Current published version: 0.1.2. Git tags match crate versions (v0.1.0, v0.1.1, v0.1.2).

**Why:** personal/open-source project by hz2 (jsondevers@gmail.com), repo at github.com/hz2/everett.

**How to apply:** when bumping versions, update Cargo.toml, commit, push, `cargo publish`, then `git tag vX.Y.Z && git push origin vX.Y.Z`. The release-check CI job runs `cargo publish --dry-run` on version tags.

Three backends: StateVectorBackend (dense), DensityMatrixBackend (noise/mixed states), StabilizerBackend (Clifford/Gottesman-Knill O(n^2)). OpenQASM 3.0 import/export. Optional rayon parallelism via `--features parallel`.
