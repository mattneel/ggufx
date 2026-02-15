defmodule GGUFX.DequantizeTest do
  use ExUnit.Case, async: true

  alias GGUFX.Dequantize
  alias GGUFX.TestHelpers

  test "decodes f16 values" do
    bits = [TestHelpers.f32_to_f16_bits(1.0), TestHelpers.f32_to_f16_bits(-2.0)]
    binary = IO.iodata_to_binary(for b <- bits, do: <<b::little-16>>)

    assert {:ok, tensor} = Dequantize.dequantize(binary, :f16, 2)
    assert Nx.to_flat_list(tensor) == [1.0, -2.0]
  end

  test "decodes bf16 values" do
    binary = <<0x3F80::little-16, 0xC000::little-16>>

    assert {:ok, tensor} = Dequantize.dequantize(binary, :bf16, 2)
    assert Nx.to_flat_list(tensor) == [1.0, -2.0]
  end

  test "dequantizes q4_0 blocks" do
    quants = Enum.to_list(-8..7) ++ Enum.to_list(-8..7)
    binary = TestHelpers.pack_q4_0_blocks(0.5, quants)

    assert {:ok, tensor} = Dequantize.dequantize(binary, :q4_0, 32)

    expected = Enum.map(quants, &(&1 * 0.5))
    assert Nx.to_flat_list(tensor) == expected
  end

  test "dequantizes q8_0 blocks" do
    quants = Enum.to_list(-16..15)
    binary = TestHelpers.pack_q8_0_blocks(0.25, quants)

    assert {:ok, tensor} = Dequantize.dequantize(binary, :q8_0, 32)

    expected = Enum.map(quants, &(&1 * 0.25))
    assert Nx.to_flat_list(tensor) == expected
  end

  test "dequantizes q4_k blocks" do
    block = make_q4_k_block()

    assert {:ok, tensor} = Dequantize.dequantize(block, :q4_k, 256)

    expected =
      1..4
      |> Enum.flat_map(fn _ -> List.duplicate(1.0, 32) ++ List.duplicate(2.0, 32) end)

    assert Nx.to_flat_list(tensor) == expected
  end

  test "dequantizes q6_k blocks" do
    block = make_q6_k_block()

    assert {:ok, tensor} = Dequantize.dequantize(block, :q6_k, 256)
    assert Nx.to_flat_list(tensor) == List.duplicate(-32.0, 256)
  end

  test "returns unsupported quant error" do
    assert {:error, {:unsupported_quant, :q5_0}} = Dequantize.dequantize(<<>>, :q5_0, 0)
  end

  defp make_q4_k_block do
    d = TestHelpers.f32_to_f16_bits(1.0)
    dmin = TestHelpers.f32_to_f16_bits(1.0)

    scales =
      <<1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1>>

    # low nibble => 1, high nibble => 2
    qs = :binary.copy(<<0x21>>, 128)

    <<d::little-16, dmin::little-16, scales::binary, qs::binary>>
  end

  defp make_q6_k_block do
    d = TestHelpers.f32_to_f16_bits(1.0)
    ql = :binary.copy(<<0>>, 128)
    qh = :binary.copy(<<0>>, 64)
    scales = :binary.copy(<<1>>, 16)

    <<ql::binary, qh::binary, scales::binary, d::little-16>>
  end
end
