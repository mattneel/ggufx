defmodule GGUFX.Dequantize do
  @moduledoc """
  Dequantization helpers for GGML tensor formats.

  Quantized formats are decoded to `{:f, 32}` tensors.
  """

  import Bitwise

  alias GGUFX.Types

  @q4_0_block_size 32
  @q4_0_type_size 18
  @q8_0_block_size 32
  @q8_0_type_size 34
  @q4_k_block_size 256
  @q4_k_type_size 144
  @q6_k_block_size 256
  @q6_k_type_size 210

  @type dequant_result :: {:ok, Nx.Tensor.t()} | {:error, term()}

  @doc "Dequantize binary data of the given type and element count."
  @spec dequantize(binary(), Types.ggml_type(), non_neg_integer()) :: dequant_result()
  def dequantize(binary, :f32, count), do: decode_direct(binary, {:f, 32}, count)
  def dequantize(binary, :f64, count), do: decode_direct(binary, {:f, 64}, count)
  def dequantize(binary, :i8, count), do: decode_direct(binary, {:s, 8}, count)
  def dequantize(binary, :i16, count), do: decode_direct(binary, {:s, 16}, count)
  def dequantize(binary, :i32, count), do: decode_direct(binary, {:s, 32}, count)
  def dequantize(binary, :i64, count), do: decode_direct(binary, {:s, 64}, count)
  def dequantize(binary, :f16, count), do: decode_f16(binary, count)
  def dequantize(binary, :bf16, count), do: decode_bf16(binary, count)
  def dequantize(binary, :q4_0, count), do: decode_q4_0(binary, count)
  def dequantize(binary, :q8_0, count), do: decode_q8_0(binary, count)
  def dequantize(binary, :q4_k, count), do: decode_q4_k(binary, count)
  def dequantize(binary, :q6_k, count), do: decode_q6_k(binary, count)

  def dequantize(_binary, type, _count) do
    {:error, {:unsupported_quant, type}}
  end

  @doc "Decode a raw IEEE754 half-float to float32."
  @spec f16_to_f32(non_neg_integer()) :: float()
  def f16_to_f32(bits) when bits >= 0 and bits <= 0xFFFF do
    sign = if (bits &&& 0x8000) == 0, do: 1.0, else: -1.0
    exponent = bits >>> 10 &&& 0x1F
    mantissa = bits &&& 0x03FF

    cond do
      exponent == 0 and mantissa == 0 ->
        sign * 0.0

      exponent == 0 ->
        sign * :math.pow(2.0, -14.0) * (mantissa / 1024.0)

      exponent == 31 and mantissa == 0 ->
        if sign > 0.0, do: pos_inf(), else: neg_inf()

      exponent == 31 ->
        nan()

      true ->
        sign * :math.pow(2.0, exponent - 15.0) * (1.0 + mantissa / 1024.0)
    end
  end

  defp decode_direct(binary, type, count) do
    tensor = Nx.from_binary(binary, type)

    if Nx.size(tensor) < count do
      {:error, {:parse_error, {:tensor_data_truncated, count}}}
    else
      {:ok, slice_1d(tensor, count)}
    end
  end

  defp decode_f16(binary, count) do
    expected = count * 2

    if byte_size(binary) < expected do
      {:error, {:parse_error, {:tensor_data_truncated, expected}}}
    else
      values =
        for <<bits::unsigned-little-16 <- binary_part(binary, 0, expected)>> do
          f16_to_f32(bits)
        end

      floats_to_tensor(values, count)
    end
  end

  defp decode_bf16(binary, count) do
    expected = count * 2

    if byte_size(binary) < expected do
      {:error, {:parse_error, {:tensor_data_truncated, expected}}}
    else
      values =
        for <<bits::unsigned-little-16 <- binary_part(binary, 0, expected)>> do
          <<f::float-little-32>> = <<bits <<< 16::unsigned-little-32>>
          f
        end

      floats_to_tensor(values, count)
    end
  end

  defp decode_q4_0(binary, count) do
    with {:ok, blocks} <- split_blocks(binary, @q4_0_type_size, @q4_0_block_size, count) do
      values =
        Enum.flat_map(blocks, fn <<d::unsigned-little-16, quants::binary-size(16)>> ->
          scale = f16_to_f32(d)
          lows = for <<packed::unsigned-8 <- quants>>, do: ((packed &&& 0x0F) - 8) * scale
          highs = for <<packed::unsigned-8 <- quants>>, do: ((packed >>> 4) - 8) * scale
          lows ++ highs
        end)

      floats_to_tensor(values, count)
    end
  end

  defp decode_q8_0(binary, count) do
    with {:ok, blocks} <- split_blocks(binary, @q8_0_type_size, @q8_0_block_size, count) do
      values =
        Enum.flat_map(blocks, fn <<d::unsigned-little-16, quants::binary-size(32)>> ->
          scale = f16_to_f32(d)

          for <<q::signed-little-8 <- quants>> do
            q * scale
          end
        end)

      floats_to_tensor(values, count)
    end
  end

  defp decode_q4_k(binary, count) do
    with {:ok, blocks} <- split_blocks(binary, @q4_k_type_size, @q4_k_block_size, count) do
      values = Enum.flat_map(blocks, &decode_q4_k_block/1)
      floats_to_tensor(values, count)
    end
  end

  defp decode_q6_k(binary, count) do
    with {:ok, blocks} <- split_blocks(binary, @q6_k_type_size, @q6_k_block_size, count) do
      values = Enum.flat_map(blocks, &decode_q6_k_block/1)
      floats_to_tensor(values, count)
    end
  end

  defp split_blocks(binary, type_size, block_size, count) do
    if rem(count, block_size) != 0 do
      {:error, {:parse_error, {:invalid_size, count, block_size}}}
    else
      expected = div(count, block_size) * type_size

      if byte_size(binary) < expected do
        {:error, {:parse_error, {:tensor_data_truncated, expected}}}
      else
        payload = binary_part(binary, 0, expected)
        {:ok, for(<<block::binary-size(type_size) <- payload>>, do: block)}
      end
    end
  end

  defp decode_q4_k_block(
         <<d_bits::unsigned-little-16, dmin_bits::unsigned-little-16, scales::binary-size(12),
           qs::binary-size(128)>>
       ) do
    d = f16_to_f32(d_bits)
    dmin = f16_to_f32(dmin_bits)

    {_, values} =
      Enum.reduce(for(<<chunk::binary-size(32) <- qs>>, do: chunk), {0, []}, fn chunk,
                                                                                {is, acc} ->
        {s1, m1} = q4_k_scale_min(is, scales)
        {s2, m2} = q4_k_scale_min(is + 1, scales)

        ds1 = d * s1
        dm1 = dmin * m1
        ds2 = d * s2
        dm2 = dmin * m2

        lows = for <<byte::unsigned-8 <- chunk>>, do: ds1 * (byte &&& 0x0F) - dm1
        highs = for <<byte::unsigned-8 <- chunk>>, do: ds2 * (byte >>> 4) - dm2

        {is + 2, [acc, lows, highs]}
      end)

    List.flatten(values)
  end

  defp decode_q6_k_block(
         <<ql::binary-size(128), qh::binary-size(64), scales::binary-size(16),
           d_bits::unsigned-little-16>>
       ) do
    d = f16_to_f32(d_bits)
    scale_values = for <<s::signed-little-8 <- scales>>, do: s

    0..1
    |> Enum.flat_map(fn chunk_idx ->
      ql_chunk = binary_part(ql, chunk_idx * 64, 64)
      qh_chunk = binary_part(qh, chunk_idx * 32, 32)
      sc = Enum.slice(scale_values, chunk_idx * 8, 8)

      lows =
        for l <- 0..31 do
          is = div(l, 16)
          b0 = :binary.at(ql_chunk, l)
          bh = :binary.at(qh_chunk, l)
          q = (b0 &&& 0x0F) ||| (bh >>> 0 &&& 0x03) <<< 4
          d * Enum.at(sc, is + 0) * (q - 32)
        end

      mids =
        for l <- 0..31 do
          is = div(l, 16)
          b1 = :binary.at(ql_chunk, l + 32)
          bh = :binary.at(qh_chunk, l)
          q = (b1 &&& 0x0F) ||| (bh >>> 2 &&& 0x03) <<< 4
          d * Enum.at(sc, is + 2) * (q - 32)
        end

      highs =
        for l <- 0..31 do
          is = div(l, 16)
          b0 = :binary.at(ql_chunk, l)
          bh = :binary.at(qh_chunk, l)
          q = b0 >>> 4 ||| (bh >>> 4 &&& 0x03) <<< 4
          d * Enum.at(sc, is + 4) * (q - 32)
        end

      top =
        for l <- 0..31 do
          is = div(l, 16)
          b1 = :binary.at(ql_chunk, l + 32)
          bh = :binary.at(qh_chunk, l)
          q = b1 >>> 4 ||| (bh >>> 6 &&& 0x03) <<< 4
          d * Enum.at(sc, is + 6) * (q - 32)
        end

      lows ++ mids ++ highs ++ top
    end)
  end

  defp q4_k_scale_min(j, scales) when j < 4 do
    d = :binary.at(scales, j) &&& 0x3F
    m = :binary.at(scales, j + 4) &&& 0x3F
    {d, m}
  end

  defp q4_k_scale_min(j, scales) do
    a = :binary.at(scales, j + 4)
    b = :binary.at(scales, j - 4)
    c = :binary.at(scales, j)

    d = (a &&& 0x0F) ||| (b >>> 6) <<< 4
    m = a >>> 4 ||| (c >>> 6) <<< 4
    {d, m}
  end

  defp floats_to_tensor(values, count) do
    if length(values) < count do
      {:error, {:parse_error, {:tensor_data_truncated, count}}}
    else
      payload =
        values
        |> Enum.take(count)
        |> Enum.map(&<<&1::float-little-32>>)
        |> IO.iodata_to_binary()

      {:ok, Nx.from_binary(payload, {:f, 32})}
    end
  end

  defp slice_1d(tensor, count) do
    if Nx.size(tensor) == count do
      tensor
    else
      Nx.slice_along_axis(tensor, 0, count, axis: 0)
    end
  end

  defp pos_inf do
    <<v::float-little-32>> = <<0x7F80_0000::unsigned-little-32>>
    v
  end

  defp neg_inf do
    <<v::float-little-32>> = <<0xFF80_0000::unsigned-little-32>>
    v
  end

  defp nan do
    <<v::float-little-32>> = <<0x7FC0_0000::unsigned-little-32>>
    v
  end
end
