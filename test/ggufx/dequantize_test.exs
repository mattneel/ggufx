defmodule GGUFX.DequantizeTest do
  use ExUnit.Case, async: true

  import Bitwise

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
    quants = Enum.to_list(-8..7) ++ Enum.to_list(7..-8//-1)
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
    {block, expected} = make_q4_k_block()

    assert {:ok, tensor} = Dequantize.dequantize(block, :q4_k, 256)

    assert Nx.to_flat_list(tensor) == expected
  end

  test "dequantizes q6_k blocks" do
    {block, expected} = make_q6_k_block()

    assert {:ok, tensor} = Dequantize.dequantize(block, :q6_k, 256)
    assert Nx.to_flat_list(tensor) == expected
  end

  test "returns unsupported quant error" do
    assert {:error, {:unsupported_quant, :q5_0}} = Dequantize.dequantize(<<>>, :q5_0, 0)
  end

  defp make_q4_k_block do
    d = TestHelpers.f32_to_f16_bits(1.0)
    dmin = TestHelpers.f32_to_f16_bits(1.0)
    scales = [1, 2, 3, 4, 35, 36, 37, 38]
    mins = [5, 6, 7, 8, 45, 46, 47, 48]
    packed_scales = encode_q4_k_scales(scales, mins)

    # low nibble => 1, high nibble => 2
    qs = :binary.copy(<<0x21>>, 128)

    expected =
      0..3
      |> Enum.flat_map(fn chunk ->
        is = chunk * 2
        d1 = Enum.at(scales, is) * 1.0
        m1 = Enum.at(mins, is) * 1.0
        d2 = Enum.at(scales, is + 1) * 2.0
        m2 = Enum.at(mins, is + 1) * 1.0
        List.duplicate(d1 - m1, 32) ++ List.duplicate(d2 - m2, 32)
      end)

    {<<d::little-16, dmin::little-16, packed_scales::binary, qs::binary>>, expected}
  end

  defp make_q6_k_block do
    d = TestHelpers.f32_to_f16_bits(1.0)
    scales = Enum.to_list(1..16)

    {ql, qh, expected} =
      Enum.reduce(0..1, {List.duplicate(0, 128), List.duplicate(0, 64), []}, fn chunk_idx,
                                                                                {ql_acc, qh_acc,
                                                                                 out_acc} ->
        chunk_scales = Enum.slice(scales, chunk_idx * 8, 8)

        {ql_next, qh_next, chunk_values} =
          Enum.reduce(0..31, {ql_acc, qh_acc, List.duplicate(0.0, 128)}, fn l, {ql0, qh0, vals} ->
            q1 = rem(l + chunk_idx * 7, 64)
            q2 = rem(l + 1 + chunk_idx * 7, 64)
            q3 = rem(l + 2 + chunk_idx * 7, 64)
            q4 = rem(l + 3 + chunk_idx * 7, 64)

            ql_l = (q1 &&& 0x0F) ||| (q3 &&& 0x0F) <<< 4
            ql_h = (q2 &&& 0x0F) ||| (q4 &&& 0x0F) <<< 4

            qh_v =
              (q1 >>> 4 &&& 0x03) ||| (q2 >>> 4 &&& 0x03) <<< 2 |||
                (q3 >>> 4 &&& 0x03) <<< 4 ||| (q4 >>> 4 &&& 0x03) <<< 6

            ql0 =
              ql0
              |> List.replace_at(chunk_idx * 64 + l, ql_l)
              |> List.replace_at(chunk_idx * 64 + 32 + l, ql_h)

            qh0 = List.replace_at(qh0, chunk_idx * 32 + l, qh_v)

            is = div(l, 16)

            vals =
              vals
              |> List.replace_at(l, Enum.at(chunk_scales, is + 0) * (q1 - 32) * 1.0)
              |> List.replace_at(l + 32, Enum.at(chunk_scales, is + 2) * (q2 - 32) * 1.0)
              |> List.replace_at(l + 64, Enum.at(chunk_scales, is + 4) * (q3 - 32) * 1.0)
              |> List.replace_at(l + 96, Enum.at(chunk_scales, is + 6) * (q4 - 32) * 1.0)

            {ql0, qh0, vals}
          end)

        {ql_next, qh_next, out_acc ++ chunk_values}
      end)

    ql_bin = IO.iodata_to_binary(for b <- ql, do: <<b::unsigned-8>>)
    qh_bin = IO.iodata_to_binary(for b <- qh, do: <<b::unsigned-8>>)
    scales_bin = IO.iodata_to_binary(for s <- scales, do: <<s::signed-little-8>>)

    {<<ql_bin::binary, qh_bin::binary, scales_bin::binary, d::little-16>>, expected}
  end

  defp encode_q4_k_scales(scales, mins) do
    q0_3 =
      for j <- 0..3 do
        (Enum.at(scales, j) &&& 0x3F) ||| (Enum.at(scales, j + 4) >>> 4 &&& 0x03) <<< 6
      end

    q4_7 =
      for j <- 0..3 do
        (Enum.at(mins, j) &&& 0x3F) ||| (Enum.at(mins, j + 4) >>> 4 &&& 0x03) <<< 6
      end

    q8_11 =
      for j <- 0..3 do
        (Enum.at(scales, j + 4) &&& 0x0F) ||| (Enum.at(mins, j + 4) &&& 0x0F) <<< 4
      end

    IO.iodata_to_binary(for b <- q0_3 ++ q4_7 ++ q8_11, do: <<b::unsigned-8>>)
  end
end
