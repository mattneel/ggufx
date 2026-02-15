# AGENTS.md

## Purpose
This repository delivers `ggufx`: a pure-Elixir GGUF parser that loads model metadata and tensor data into Elixir maps and Nx tensors.

## Source Of Truth
- `SPEC.md` is canonical for scope, architecture, behavior, and acceptance criteria.
- If implementation details in code or docs conflict with `SPEC.md`, follow `SPEC.md` and update the stale files.

## `/init` Workflow
1. Read `SPEC.md` end-to-end and extract concrete tasks.
2. Update `mix.exs` for Hex publishing metadata and required deps.
3. Build the module layout from `SPEC.md`:
   - `lib/ggufx.ex`
   - `lib/ggufx/parser.ex`
   - `lib/ggufx/metadata.ex`
   - `lib/ggufx/tensor_info.ex`
   - `lib/ggufx/dequantize.ex`
   - `lib/ggufx/types.ex`
4. Implement parsing first, then tensor loading/dequantization, then lazy loading APIs.
5. Build test fixture helpers that generate hand-crafted GGUF binaries.
6. Add/finish README, CHANGELOG, and LICENSE.
7. Run all quality gates before handoff.

## Required Technical Constraints
- Pure Elixir only. No NIFs, no ports, no Rust helpers.
- Use binary pattern matching for parsing and block iteration.
- Support GGUF v2 and v3 string length differences.
- Reverse GGUF dims to Nx shape order.
- Respect alignment rules and `general.alignment` override.
- Public API returns `{:ok, _}` / `{:error, _}`; bang variants raise `GGUFX.Error`.
- Keep runtime deps minimal (Nx only; ExDoc dev-only).

## Dependency Intelligence For SOTA Elixir ML
Use both local dependency source and Hex metadata before adding/updating packages.

1. Check latest package metadata with Hex:
   - `mix hex.info nx`
   - `mix hex.info <package>`
   - `mix hex.search <term>`
2. Inspect currently resolved versions:
   - `mix deps.tree`
   - `mix hex.outdated`
   - `mix.lock`
3. Read dependency source directly under `deps/`:
   - `rg -n "<symbol_or_function>" deps/<package>`
   - open the exact module files in `deps/<package>/lib/...`
4. Prefer latest stable versions that are compatible with project constraints.
5. When bumping versions, document what changed and why (API changes, perf, bugfixes).

## Test Expectations
- Do not download real model files in tests.
- Build binary fixtures in test helpers.
- Cover parser edge cases, quantization behavior, lazy loading, filters, alignment, and metadata convenience helpers.

## Quality Gates
Run and pass:
- `mix format --check-formatted`
- `mix compile --warnings-as-errors`
- `mix test`
- `mix docs`

## Delivery Definition
Delivery is complete only when:
- Implementation matches `SPEC.md`.
- Public docs are coherent and publish-ready.
- All quality gates pass without warnings.
