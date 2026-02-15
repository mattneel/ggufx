defmodule GGUFX.Types do
  @moduledoc """
  GGML type constants and utilities.
  """

  @typedoc "GGML tensor type"
  @type ggml_type ::
          :f32
          | :f16
          | :q4_0
          | :q4_1
          | :q5_0
          | :q5_1
          | :q8_0
          | :q8_1
          | :q2_k
          | :q3_k
          | :q4_k
          | :q5_k
          | :q6_k
          | :q8_k
          | :iq2_xxs
          | :iq2_xs
          | :iq3_xxs
          | :iq1_s
          | :iq4_nl
          | :iq3_s
          | :iq2_s
          | :iq4_xs
          | :i8
          | :i16
          | :i32
          | :i64
          | :f64
          | :iq1_m
          | :bf16

  @type nx_or_quantized :: {:ok, Nx.Type.t()} | {:error, :quantized}

  @id_to_type %{
    0 => :f32,
    1 => :f16,
    2 => :q4_0,
    3 => :q4_1,
    6 => :q5_0,
    7 => :q5_1,
    8 => :q8_0,
    9 => :q8_1,
    10 => :q2_k,
    11 => :q3_k,
    12 => :q4_k,
    13 => :q5_k,
    14 => :q6_k,
    15 => :q8_k,
    16 => :iq2_xxs,
    17 => :iq2_xs,
    18 => :iq3_xxs,
    19 => :iq1_s,
    20 => :iq4_nl,
    21 => :iq3_s,
    22 => :iq2_s,
    23 => :iq4_xs,
    24 => :i8,
    25 => :i16,
    26 => :i32,
    27 => :i64,
    28 => :f64,
    29 => :iq1_m,
    30 => :bf16
  }

  @type_to_id for {id, type} <- @id_to_type, into: %{}, do: {type, id}

  @block_size %{
    f32: 1,
    f16: 1,
    bf16: 1,
    f64: 1,
    i8: 1,
    i16: 1,
    i32: 1,
    i64: 1,
    q4_0: 32,
    q4_1: 32,
    q5_0: 32,
    q5_1: 32,
    q8_0: 32,
    q8_1: 32,
    q2_k: 256,
    q3_k: 256,
    q4_k: 256,
    q5_k: 256,
    q6_k: 256,
    q8_k: 256,
    iq2_xxs: 256,
    iq2_xs: 256,
    iq3_xxs: 256,
    iq1_s: 256,
    iq4_nl: 32,
    iq3_s: 256,
    iq2_s: 256,
    iq4_xs: 256,
    iq1_m: 256
  }

  @type_size %{
    f32: 4,
    f16: 2,
    bf16: 2,
    f64: 8,
    i8: 1,
    i16: 2,
    i32: 4,
    i64: 8,
    q4_0: 18,
    q4_1: 20,
    q5_0: 22,
    q5_1: 24,
    q8_0: 34,
    q8_1: 36,
    q2_k: 84,
    q3_k: 110,
    q4_k: 144,
    q5_k: 176,
    q6_k: 210,
    q8_k: 292,
    iq2_xxs: 66,
    iq2_xs: 74,
    iq3_xxs: 98,
    iq1_s: 50,
    iq4_nl: 18,
    iq3_s: 110,
    iq2_s: 82,
    iq4_xs: 136,
    iq1_m: 56
  }

  @doc "Convert integer type ID to atom."
  @spec from_id(non_neg_integer()) :: {:ok, ggml_type()} | {:error, :unknown_type}
  def from_id(id) do
    case Map.fetch(@id_to_type, id) do
      {:ok, type} -> {:ok, type}
      :error -> {:error, :unknown_type}
    end
  end

  @doc "Convert a type atom to integer GGML type ID."
  @spec to_id(ggml_type()) :: {:ok, non_neg_integer()} | {:error, :unknown_type}
  def to_id(type) do
    case Map.fetch(@type_to_id, type) do
      {:ok, id} -> {:ok, id}
      :error -> {:error, :unknown_type}
    end
  end

  @doc "Block size for a given type (number of values per block)."
  @spec block_size(ggml_type()) :: pos_integer()
  def block_size(type), do: Map.fetch!(@block_size, type)

  @doc "Byte size of one block for a given type."
  @spec type_size(ggml_type()) :: pos_integer()
  def type_size(type), do: Map.fetch!(@type_size, type)

  @doc "Convert to Nx type where applicable (for unquantized types)."
  @spec to_nx_type(ggml_type()) :: nx_or_quantized()
  def to_nx_type(:f32), do: {:ok, {:f, 32}}
  def to_nx_type(:f16), do: {:ok, {:f, 16}}
  def to_nx_type(:bf16), do: {:ok, {:bf, 16}}
  def to_nx_type(:f64), do: {:ok, {:f, 64}}
  def to_nx_type(:i8), do: {:ok, {:s, 8}}
  def to_nx_type(:i16), do: {:ok, {:s, 16}}
  def to_nx_type(:i32), do: {:ok, {:s, 32}}
  def to_nx_type(:i64), do: {:ok, {:s, 64}}
  def to_nx_type(_), do: {:error, :quantized}

  @doc "Compute byte size for `n_elements` of a given type."
  @spec tensor_byte_size(ggml_type(), non_neg_integer()) ::
          {:ok, non_neg_integer()} | {:error, {:invalid_size, ggml_type(), non_neg_integer()}}
  def tensor_byte_size(type, n_elements) when n_elements >= 0 do
    block = block_size(type)
    size = type_size(type)

    if rem(n_elements, block) == 0 do
      {:ok, div(n_elements * size, block)}
    else
      {:error, {:invalid_size, type, n_elements}}
    end
  end
end
