---
name: ci
description: GitHub Actions CI jobs and what each covers
metadata:
  type: project
---

CI runs on push to main and PRs. All jobs use nix (DeterminateSystems/nix-installer-action@v22, nix-community/cache-nix-action@v7).

**Jobs:**
- `nix flake check` — build, clippy (--deny warnings), nextest, doctests, rustdoc, treefmt formatting
- `miri` — UB checking via `nix develop .#miri`; runs tests prefixed `miri_`
- `kani` — bounded formal verification via `model-checking/kani-github-action@v1` (kani not in nixpkgs)
- `release-check` — `cargo publish --dry-run` via nix; only runs on `refs/tags/v*` pushes
- `coverage` — `cargo llvm-cov --all-features` via nix, uploads to Codecov (token in repo secret `CODECOV_TOKEN`)

**Why kani uses GitHub action not nix:** `pkgs.kani` does not exist in nixpkgs-unstable. The GitHub action bundles its own kani installation.

**How to apply:** when CI fails on formatting, run `nix develop --command cargo fmt` locally and commit. When clippy fails, fix warnings before committing (hook in .claude/settings.json blocks commits on clippy failure).
