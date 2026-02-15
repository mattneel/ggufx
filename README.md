# GGUFX

GGUF parser for Elixir. `GGUFX` loads quantized LLM model files and returns metadata and tensors as Elixir maps and Nx tensors. The implementation is pure Elixir and uses binary pattern matching throughout (no NIFs).

## Installation

Add `ggufx` to your dependencies:

```elixir
def deps do
  [
    {:ggufx, "~> 0.1.0"}
  ]
end
```

## Quick Start

```elixir
{:ok, model} = GGUFX.load("path/to/model.gguf")

model.metadata["general.architecture"]
#=> "llama"

tensor = model.tensors["blk.0.attn_q.weight"]
Nx.shape(tensor)
```

## Lazy Loading

```elixir
{:ok, model} = GGUFX.load("path/to/model.gguf", lazy: true)
# no tensors loaded into memory yet

{:ok, w} = GGUFX.fetch_tensor(model, "blk.0.attn_q.weight")
```

## Tensor Filtering

```elixir
{:ok, model} =
  GGUFX.load("path/to/model.gguf",
    tensor_filter: fn name -> String.starts_with?(name, "blk.0") end
  )

Map.keys(model.tensors)
#=> ["blk.0....", ...]
```

## Supported Tensor Types

| Type | Status |
| --- | --- |
| `f32`, `f16`, `bf16`, `f64` | Supported |
| `i8`, `i16`, `i32`, `i64` | Supported |
| `q4_0`, `q8_0` | Supported |
| `q4_k`, `q6_k` | Supported |
| Other GGML quant types | Parsed in tensor info, dequantization returns `{:unsupported_quant, type}` |

## Docs

HexDocs: <https://hexdocs.pm/ggufx>

## Notes

This library is intended to support Elixir-native GGUF workflows and can help upstream efforts such as Bumblebee GGUF support discussions (for example, issue `bumblebee#413`).
