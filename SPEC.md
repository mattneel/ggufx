# GGUFX — Coding Agent Prompt

## Identity

You are building `ggufx`, an Elixir library that parses GGUF (GGML Universal File) model files and returns their contents as Nx tensors and Elixir maps. This is a hex.pm-publishable library. The repo is `mattneel/ggufx`.

## Project Setup

The project has already been created with `mix new ggufx`. You need to:

1. Set up `mix.exs` for hex publishing (description, package, source_url, docs, licenses, links)
2. Add dependencies: `{:nx, "~> 0.9"}` (only runtime dep). Dev/test deps: `{:ex_doc, "~> 0.34", only: :dev, runtime: false}`
3. Create the full library implementation
4. Create comprehensive tests with small hand-crafted GGUF binaries (do NOT depend on downloading real model files for tests)

## What GGUF Is

GGUF is a binary file format for storing ML model weights and metadata. It is the standard format used by llama.cpp and Ollama. The format is little-endian and designed to be mmap'd.

### Binary Layout

```
┌─────────────────────────────────┐
│ Header                          │
│   magic: "GGUF" (4 bytes)       │
│   version: uint32_le            │
│   tensor_count: uint64_le       │
│   metadata_kv_count: uint64_le  │
├─────────────────────────────────┤
│ Metadata KV Pairs               │
│   (repeated metadata_kv_count)  │
│   key: gguf_string              │
│   value_type: uint32_le         │
│   value: (type-dependent)       │
├─────────────────────────────────┤
│ Tensor Info Array                │
│   (repeated tensor_count)       │
│   name: gguf_string             │
│   n_dims: uint32_le             │
│   dims: uint64_le × n_dims      │
│   type: uint32_le (ggml_type)   │
│   offset: uint64_le             │
├─────────────────────────────────┤
│ Padding to ALIGNMENT            │
├─────────────────────────────────┤
│ Tensor Data Blob                │
│   (contiguous, each tensor      │
│    aligned to ALIGNMENT)        │
└─────────────────────────────────┘
```

Default ALIGNMENT is 32 bytes. Can be overridden by `general.alignment` metadata key.

### gguf_string format

```
length: uint64_le
data: bytes × length    (NOT null-terminated)
```

### Metadata Value Types

```
GGUF_TYPE_UINT8    = 0   → uint8
GGUF_TYPE_INT8     = 1   → int8
GGUF_TYPE_UINT16   = 2   → uint16_le
GGUF_TYPE_INT16    = 3   → int16_le
GGUF_TYPE_UINT32   = 4   → uint32_le
GGUF_TYPE_INT32    = 5   → int32_le
GGUF_TYPE_FLOAT32  = 6   → float32_le
GGUF_TYPE_BOOL     = 7   → uint8 (0 = false, else true)
GGUF_TYPE_STRING   = 8   → gguf_string
GGUF_TYPE_ARRAY    = 9   → element_type(uint32_le) + length(uint64_le) + elements
GGUF_TYPE_UINT64   = 10  → uint64_le
GGUF_TYPE_INT64    = 11  → int64_le
GGUF_TYPE_FLOAT64  = 12  → float64_le
```

### GGML Tensor Types (ggml_type enum)

```
GGML_TYPE_F32      = 0
GGML_TYPE_F16      = 1
GGML_TYPE_Q4_0     = 2
GGML_TYPE_Q4_1     = 3
GGML_TYPE_Q5_0     = 6
GGML_TYPE_Q5_1     = 7
GGML_TYPE_Q8_0     = 8
GGML_TYPE_Q8_1     = 9
GGML_TYPE_Q2_K     = 10
GGML_TYPE_Q3_K     = 11
GGML_TYPE_Q4_K     = 12
GGML_TYPE_Q5_K     = 13
GGML_TYPE_Q6_K     = 14
GGML_TYPE_Q8_K     = 15
GGML_TYPE_IQ2_XXS  = 16
GGML_TYPE_IQ2_XS   = 17
GGML_TYPE_IQ3_XXS  = 18
GGML_TYPE_IQ1_S    = 19
GGML_TYPE_IQ4_NL   = 20
GGML_TYPE_IQ3_S    = 21
GGML_TYPE_IQ2_S    = 22
GGML_TYPE_IQ4_XS   = 23
GGML_TYPE_I8       = 24
GGML_TYPE_I16      = 25
GGML_TYPE_I32      = 26
GGML_TYPE_I64      = 27
GGML_TYPE_F64      = 28
GGML_TYPE_IQ1_M    = 29
GGML_TYPE_BF16     = 30
```

### Quantization Block Formats

Each quantized type packs weights into fixed-size blocks. Here are the critical ones:

**Q4_0** (block_size=32, 18 bytes per block):
```
scale: float16 (2 bytes) — the dequantization scale factor
quants: uint8[16] (16 bytes) — 32 4-bit values packed into 16 bytes
```
Dequant: for each pair of 4-bit values in a byte, low nibble = (byte & 0x0F) - 8, high nibble = (byte >> 4) - 8, then multiply by scale.

**Q4_1** (block_size=32, 20 bytes per block):
```
scale: float16 (2 bytes)
min: float16 (2 bytes)
quants: uint8[16] (16 bytes)
```
Dequant: low = (byte & 0x0F), high = (byte >> 4), value = low * scale + min

**Q8_0** (block_size=32, 34 bytes per block):
```
scale: float16 (2 bytes)
quants: int8[32] (32 bytes)
```
Dequant: value = quant * scale

**Q4_K** (block_size=256, 144 bytes per block — also called Q4_K_M):
```
d: float16 (2 bytes) — super-block scale
dmin: float16 (2 bytes) — super-block min
scales: uint8[12] (12 bytes) — quantized sub-block scales/mins
qs: uint8[128] (128 bytes) — 256 4-bit quantized values
```
This is a K-quant type with nested quantization (scales are themselves quantized). Dequantization is more complex — sub-blocks of 32 values each get their own scale/min derived from the 12-byte scales array.

**Q6_K** (block_size=256, 210 bytes per block):
```
ql: uint8[128] — lower 4 bits of 6-bit quants
qh: uint8[64] — upper 2 bits of 6-bit quants
scales: int8[16] — sub-block scales
d: float16 (2 bytes) — super-block scale
```

**Q5_K** (block_size=256, 176 bytes per block):
```
d: float16
dmin: float16
scales: uint8[12]
qh: uint8[32]
qs: uint8[128]
```

**F16**: Standard IEEE 754 half-precision. 2 bytes per value.

**BF16**: Brain floating point. 2 bytes per value. Convert to F32 by left-shifting 16 bits.

## Architecture

### Module Structure

```
lib/
  ggufx.ex                    # Main public API
  ggufx/
    parser.ex                 # Low-level binary parsing
    metadata.ex               # Metadata extraction and typing
    tensor_info.ex            # Tensor info structs
    dequantize.ex             # Quantization format handlers
    types.ex                  # Type definitions and constants
```

### Public API (lib/ggufx.ex)

```elixir
defmodule GGUFX do
  @moduledoc """
  GGUF file parser for Elixir.

  Parses GGUF model files and returns metadata and tensors as Nx tensors.

  ## Quick Start

      {:ok, model} = GGUFX.load("path/to/model.gguf")
      model.metadata["general.architecture"]
      #=> "llama"

      model.tensors["blk.0.attn_q.weight"]
      #=> #Nx.Tensor<f32[4096][4096]>

  ## Lazy Loading

      {:ok, model} = GGUFX.load("path/to/model.gguf", lazy: true)
      # Tensors are not loaded into memory yet
      tensor = GGUFX.fetch_tensor!(model, "blk.0.attn_q.weight")

  ## Streaming / Tensor Selection

      # Only load specific tensors
      {:ok, model} = GGUFX.load("path/to/model.gguf",
        tensor_filter: fn name -> String.starts_with?(name, "blk.0") end
      )
  """

  @type t :: %__MODULE__{
    version: pos_integer(),
    metadata: %{String.t() => metadata_value()},
    tensor_info: %{String.t() => GGUFX.TensorInfo.t()},
    tensors: %{String.t() => Nx.Tensor.t()} | nil,
    source: String.t() | nil
  }

  @type metadata_value ::
    integer() | float() | boolean() | String.t() | [metadata_value()]

  defstruct [:version, :metadata, :tensor_info, :tensors, :source]

  @doc "Load and parse a GGUF file. Options: `:lazy`, `:tensor_filter`, `:dequantize` (default true)"
  @spec load(Path.t(), keyword()) :: {:ok, t()} | {:error, term()}

  @doc "Same as load/2 but raises on error."
  @spec load!(Path.t(), keyword()) :: t()

  @doc "Fetch a single tensor from a lazily-loaded model."
  @spec fetch_tensor(t(), String.t()) :: {:ok, Nx.Tensor.t()} | {:error, term()}

  @doc "Same as fetch_tensor/2 but raises on error."
  @spec fetch_tensor!(t(), String.t()) :: Nx.Tensor.t()

  @doc "List all tensor names and their info without loading data."
  @spec tensor_names(t()) :: [String.t()]

  @doc "Return metadata as a map."
  @spec metadata(t()) :: %{String.t() => metadata_value()}

  @doc """
  Parse only the header and metadata from a GGUF file.
  Useful for inspecting a model without loading tensors.
  """
  @spec peek(Path.t()) :: {:ok, t()} | {:error, term()}
end
```

### GGUFX.TensorInfo (lib/ggufx/tensor_info.ex)

```elixir
defmodule GGUFX.TensorInfo do
  @moduledoc "Describes a tensor's location and format within a GGUF file."

  @type t :: %__MODULE__{
    name: String.t(),
    shape: tuple(),
    type: atom(),
    offset: non_neg_integer(),
    byte_size: non_neg_integer()
  }

  defstruct [:name, :shape, :type, :offset, :byte_size]
end
```

### GGUFX.Types (lib/ggufx/types.ex)

Module mapping GGML type integers to atoms and providing block size / byte size info:

```elixir
defmodule GGUFX.Types do
  @moduledoc "GGML type constants and utilities."

  @type ggml_type ::
    :f32 | :f16 | :bf16 | :f64 |
    :q4_0 | :q4_1 | :q5_0 | :q5_1 |
    :q8_0 | :q8_1 |
    :q2_k | :q3_k | :q4_k | :q5_k | :q6_k | :q8_k |
    :i8 | :i16 | :i32 | :i64

  @doc "Convert integer type ID to atom."
  @spec from_id(non_neg_integer()) :: {:ok, ggml_type()} | {:error, :unknown_type}

  @doc "Block size for a given type (number of values per block)."
  @spec block_size(ggml_type()) :: pos_integer()

  @doc "Byte size of one block for a given type."
  @spec type_size(ggml_type()) :: pos_integer()

  @doc "Convert to Nx type where applicable (for unquantized types)."
  @spec to_nx_type(ggml_type()) :: {:ok, Nx.Type.t()} | {:error, :quantized}
end
```

### GGUFX.Dequantize (lib/ggufx/dequantize.ex)

```elixir
defmodule GGUFX.Dequantize do
  @moduledoc """
  Dequantization functions for GGML quantized types.

  Converts packed quantized binary data into Nx tensors of type `{:f, 32}`.
  """

  @doc "Dequantize binary data of the given type and element count to an f32 Nx tensor."
  @spec dequantize(binary(), GGUFX.Types.ggml_type(), non_neg_integer()) :: Nx.Tensor.t()
end
```

This module must implement dequantization for AT MINIMUM: `:f32`, `:f16`, `:bf16`, `:q4_0`, `:q8_0`, `:q4_k`, `:q6_k`. For unsupported types, return `{:error, {:unsupported_quant, type}}` or provide a raw binary fallback.

### GGUFX.Parser (lib/ggufx/parser.ex)

Low-level binary parsing. All functions take a binary and return `{parsed_value, rest_binary}` tuples for composability:

```elixir
defmodule GGUFX.Parser do
  @moduledoc false

  # Parse the full GGUF header
  def parse_header(binary)

  # Parse a gguf_string: uint64_le length + bytes
  def parse_string(binary)

  # Parse N metadata KV pairs
  def parse_metadata(binary, count)

  # Parse a single metadata value given its type
  def parse_value(binary, type_id)

  # Parse N tensor info entries
  def parse_tensor_infos(binary, count)
end
```

### GGUFX.Metadata (lib/ggufx/metadata.ex)

```elixir
defmodule GGUFX.Metadata do
  @moduledoc "Convenience accessors for common GGUF metadata fields."

  @doc "Get the model architecture (e.g., \"llama\", \"mistral\")."
  @spec architecture(GGUFX.t()) :: String.t() | nil

  @doc "Get the context length."
  @spec context_length(GGUFX.t()) :: pos_integer() | nil

  @doc "Get the embedding/hidden size."
  @spec embedding_length(GGUFX.t()) :: pos_integer() | nil

  @doc "Get the number of attention heads."
  @spec head_count(GGUFX.t()) :: pos_integer() | nil

  @doc "Get the number of layers."
  @spec block_count(GGUFX.t()) :: pos_integer() | nil

  @doc "Get the vocabulary size."
  @spec vocab_size(GGUFX.t()) :: pos_integer() | nil
end
```

## Implementation Requirements

### Binary Parsing

- Use Elixir binary pattern matching everywhere. No NIFs, no ports, no Rust.
- Parse greedily using `<<value::little-size(N), rest::binary>>` patterns.
- The parser must handle version 2 and version 3 GGUF files. Version 2 uses uint32 for string lengths instead of uint64. Check the version field and dispatch accordingly.
- Shapes in GGUF are stored in reverse order (innermost dimension first). Reverse them to get standard row-major shape for Nx.

### Dequantization

- For unquantized types (F32, F16, BF16, F64), convert directly to Nx tensors.
- F16 → Use `Nx.from_binary(data, {:f, 16})` if Nx supports it, otherwise decode IEEE 754 half-precision manually and cast to f32.
- BF16 → Each 2-byte value becomes f32 by appending 16 zero bits (pad on the right). In practice: `<<value::binary-size(2), 0, 0>>` per element, then read as f32. Implement with binary comprehension.
- For quantized types, dequantize block-by-block into f32. Process blocks using binary pattern matching and `for` comprehensions over the binary.
- Q4_0 dequantization (MUST be correct — this is the most common quantization):
  ```
  For each 18-byte block (32 values):
    <<scale::float-16-little, quants::binary-size(16)>> = block
    For each byte in quants (2 values per byte):
      low_nibble  = (byte &&& 0x0F) - 8
      high_nibble = (byte >>> 4) - 8
      value_low   = low_nibble * scale
      value_high  = high_nibble * scale
  ```
- Q8_0 dequantization:
  ```
  For each 34-byte block (32 values):
    <<scale::float-16-little, quants::binary-size(32)>> = block
    For each int8 in quants:
      value = quant * scale
  ```
- For K-quant types (Q4_K, Q6_K, etc.), implement the nested scale dequantization. These are more complex but follow documented patterns. If you're unsure about a K-quant implementation, stub it with a clear error message and `@doc` noting it's not yet supported.

### Lazy Loading

- When `lazy: true`, parse header + metadata + tensor_info but do NOT read tensor data.
- Store the file path in the struct's `:source` field.
- `fetch_tensor/2` reads just that tensor's bytes from disk using `:file.pread/3` (positioned read — do NOT read the whole file).
- The offset in tensor_info is relative to the START of the tensor data section, not the file. You need to track the absolute offset of the tensor data section during parsing.

### Tensor Filter

- When `:tensor_filter` function is provided, only load tensors whose names pass the filter.
- Always parse ALL tensor_info entries (they're small), but only load/dequantize data for filtered tensors.

### Error Handling

- Return `{:ok, result}` / `{:error, reason}` tuples from all public functions.
- Bang variants raise `GGUFX.Error` (define a simple exception module).
- Error reasons should be descriptive atoms or `{atom, detail}` tuples:
  - `:invalid_magic` — not a GGUF file
  - `:unsupported_version` — version not 2 or 3
  - `{:unsupported_quant, type_atom}` — quantization type not implemented
  - `:file_not_found`
  - `{:parse_error, detail}` — malformed binary data

### Float16 Handling

Elixir/Nx may or may not support `:f16` natively depending on the backend. Implement a pure-Elixir f16-to-f32 decoder as fallback:

```
<<sign::1, exponent::5, mantissa::10>> = f16_bytes
# Handle special cases: zero, denorm, inf, nan
# Normal: (-1)^sign * 2^(exponent-15) * (1 + mantissa/1024)
```

Use this for decoding the scale factors in quantized blocks (which are always stored as f16).

## Test Strategy

### Test Fixtures

Create test fixtures by BUILDING GGUF binaries in Elixir test helpers. Do NOT download real models.

```elixir
defmodule GGUFX.TestHelpers do
  @moduledoc false

  @doc "Build a minimal valid GGUF v3 binary with given metadata and tensors."
  def build_gguf(metadata \\ [], tensors \\ [])

  @doc "Build a single F32 tensor entry."
  def f32_tensor(name, shape, data)

  @doc "Build a single Q4_0 tensor entry."
  def q4_0_tensor(name, shape, data)
end
```

### Test Cases

1. **Header parsing** — valid v3 file, valid v2 file, invalid magic, unsupported version
2. **Metadata parsing** — all value types (uint8 through float64, string, bool, arrays, nested arrays)
3. **Tensor info parsing** — 1D, 2D, 3D, 4D shapes, shape reversal
4. **F32 tensor loading** — round-trip: create known tensor → write GGUF → parse → compare values
5. **F16 tensor loading** — same round-trip with f16 values
6. **Q4_0 dequantization** — create known quantized block → dequantize → verify against expected f32 values
7. **Q8_0 dequantization** — same
8. **Lazy loading** — load with `lazy: true`, verify tensors are nil, fetch one, verify it matches
9. **Tensor filter** — load with filter, verify only matching tensors present
10. **Peek** — verify metadata loads but tensors don't
11. **Metadata convenience functions** — architecture, context_length, etc.
12. **Alignment** — verify padding is handled correctly for non-default alignment
13. **Multiple tensors** — file with 3+ tensors of different types
14. **Empty metadata** — file with zero KV pairs
15. **Large string metadata** — verify long strings parse correctly

## mix.exs

```elixir
defmodule GGUFX.MixProject do
  use Mix.Project

  @version "0.1.0"
  @source_url "https://github.com/mattneel/ggufx"

  def project do
    [
      app: :ggufx,
      version: @version,
      elixir: "~> 1.15",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      name: "GGUFX",
      description: "GGUF file parser for Elixir — load quantized LLM weights as Nx tensors",
      source_url: @source_url,
      homepage_url: @source_url,
      package: package(),
      docs: docs()
    ]
  end

  def application do
    [extra_applications: [:logger]]
  end

  defp deps do
    [
      {:nx, "~> 0.9"},
      {:ex_doc, "~> 0.34", only: :dev, runtime: false}
    ]
  end

  defp package do
    [
      licenses: ["MIT"],
      links: %{"GitHub" => @source_url},
      files: ~w(lib .formatter.exs mix.exs README.md LICENSE CHANGELOG.md)
    ]
  end

  defp docs do
    [
      main: "GGUFX",
      extras: ["README.md", "CHANGELOG.md"],
      source_ref: "v#{@version}"
    ]
  end
end
```

## README.md

Write a README with:
- One-paragraph description: "GGUF parser for Elixir. Loads quantized LLM model files and returns Nx tensors. Pure Elixir, no NIFs."
- Installation (hex dep)
- Quick start example (load a model, inspect metadata, access tensors)
- Supported quantization types table
- Lazy loading example
- Tensor filtering example
- Link to hex docs
- Note about contributing to Bumblebee GGUF support (issue #413)

## CHANGELOG.md

Standard keepachangelog format with a single `## [0.1.0] - Unreleased` entry listing initial features.

## LICENSE

MIT license, copyright 2025 Matt Neel.

## Code Style

- Use `@moduledoc` on every public module
- Use `@doc` on every public function
- Use `@spec` on every public function
- Use `@type` for all custom types
- No warnings on `mix compile`
- Pass `mix format`
- No dependencies beyond Nx and ex_doc
- Elixir 1.15+ (for binary pattern matching improvements)
- Prefer binary pattern matching over `:binary.part/3` or `:binary.copy/2`
- Use `for <<chunk::binary-size(N) <- data>>` comprehensions for block iteration
- No GenServers, no processes, no OTP — this is a pure parsing library

## Important Implementation Notes

1. The tensor data section starts AFTER all tensor info entries, padded to ALIGNMENT. You must track your position after parsing all tensor info to know where tensor data begins.

2. Tensor offsets in the info array are RELATIVE to the start of tensor data, not the file. Add the tensor data section's file offset to get absolute positions.

3. GGUF v2 uses uint32 for string lengths. GGUF v3 uses uint64. Check version and branch accordingly. This affects EVERY string parse (metadata keys, metadata string values, tensor names).

4. The `general.alignment` metadata key, if present, overrides the default 32-byte alignment. You must parse metadata BEFORE computing tensor data offsets if you want to respect custom alignment. In practice, almost all files use the default.

5. When computing tensor byte size from shape and type:
   ```
   n_elements = product of all dimensions
   byte_size = (n_elements * type_size_per_block) / block_size
   ```
   Where `type_size_per_block` and `block_size` come from GGUFX.Types.

6. Shape dimensions in GGUF are stored smallest-to-largest (column-major convention). Reverse them for Nx's row-major convention. A GGUF tensor with dims `[128, 32, 4096]` becomes Nx shape `{4096, 32, 128}`.

7. For F16 scale factors in quantized blocks: these are IEEE 754 binary16 (half-precision). Elixir's binary pattern matching doesn't natively support `::float-16-little`. You need to decode them manually or use the fallback decoder.

## Deliver

Write every file. The project should compile cleanly with `mix compile`, pass `mix format --check-formatted`, generate docs with `mix docs`, and pass all tests with `mix test`. Every module has full documentation. The README is polished. This is publishable to hex.pm as-is.
