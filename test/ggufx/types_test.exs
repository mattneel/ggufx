defmodule GGUFX.TypesTest do
  use ExUnit.Case, async: true

  alias GGUFX.Types

  test "maps ids to types and back" do
    assert {:ok, :f32} = Types.from_id(0)
    assert {:ok, 0} = Types.to_id(:f32)
    assert {:ok, :q4_k} = Types.from_id(12)
    assert {:ok, 30} = Types.to_id(:bf16)
    assert {:error, :unknown_type} = Types.from_id(999)
  end

  test "returns block and type sizes" do
    assert Types.block_size(:q4_0) == 32
    assert Types.type_size(:q4_0) == 18
    assert Types.block_size(:q4_k) == 256
    assert Types.type_size(:q6_k) == 210
  end

  test "returns nx types for unquantized and error for quantized" do
    assert {:ok, {:f, 32}} = Types.to_nx_type(:f32)
    assert {:ok, {:s, 16}} = Types.to_nx_type(:i16)
    assert {:error, :quantized} = Types.to_nx_type(:q8_0)
  end

  test "computes tensor byte sizes" do
    assert {:ok, 72} = Types.tensor_byte_size(:q4_0, 128)
    assert {:ok, 144} = Types.tensor_byte_size(:q4_k, 256)
    assert {:error, {:invalid_size, :q4_0, 33}} = Types.tensor_byte_size(:q4_0, 33)
  end
end
