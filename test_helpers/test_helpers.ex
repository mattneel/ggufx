defmodule GGUFX.TestHelpers do
  @moduledoc false

  import Bitwise

  alias GGUFX.Types

  @type typed_metadata ::
          {:u8, non_neg_integer()}
          | {:i8, integer()}
          | {:u16, non_neg_integer()}
          | {:i16, integer()}
          | {:u32, non_neg_integer()}
          | {:i32, integer()}
          | {:f32, float()}
          | {:bool, boolean()}
          | {:string, String.t()}
          | {:array, atom() | tuple(), list()}
          | {:u64, non_neg_integer()}
          | {:i64, integer()}
          | {:f64, float()}

  @gguf_type %{
    u8: 0,
    i8: 1,
    u16: 2,
    i16: 3,
    u32: 4,
    i32: 5,
    f32: 6,
    bool: 7,
    string: 8,
    array: 9,
    u64: 10,
    i64: 11,
    f64: 12
  }

  @spec build_gguf([{String.t(), typed_metadata()}], [map()], keyword()) :: binary()
  def build_gguf(metadata \\ [], tensors \\ [], opts \\ []) do
    version = Keyword.get(opts, :version, 3)
    metadata = maybe_add_alignment_metadata(metadata, Keyword.get(opts, :alignment))
    alignment = metadata_alignment(metadata)

    metadata_blob =
      metadata
      |> Enum.map(&encode_metadata_entry(&1, version))
      |> IO.iodata_to_binary()

    {tensor_infos, tensor_blob} = build_tensor_data_and_infos(tensors, alignment, version)

    header =
      <<"GGUF", version::little-32, length(tensors)::little-64, length(metadata)::little-64>>

    prefix = header <> metadata_blob <> IO.iodata_to_binary(tensor_infos)
    data_offset = align_up(byte_size(prefix), alignment)
    padding = :binary.copy(<<0>>, data_offset - byte_size(prefix))

    prefix <> padding <> tensor_blob
  end

  @spec tensor(String.t(), tuple(), Types.ggml_type() | non_neg_integer(), binary()) :: map()
  def tensor(name, shape, type, data) when is_tuple(shape) and is_binary(data) do
    %{name: name, shape: shape, type: type, data: data}
  end

  @spec f32_tensor(String.t(), tuple(), [number()]) :: map()
  def f32_tensor(name, shape, values) do
    payload = IO.iodata_to_binary(for value <- values, do: <<value::float-little-32>>)
    tensor(name, shape, :f32, payload)
  end

  @spec f16_tensor(String.t(), tuple(), [number()]) :: map()
  def f16_tensor(name, shape, values) do
    payload = IO.iodata_to_binary(for value <- values, do: <<f32_to_f16_bits(value)::little-16>>)
    tensor(name, shape, :f16, payload)
  end

  @spec q4_0_tensor(String.t(), tuple(), float(), [integer()]) :: map()
  def q4_0_tensor(name, shape, scale, quants) do
    payload = pack_q4_0_blocks(scale, quants)
    tensor(name, shape, :q4_0, payload)
  end

  @spec q8_0_tensor(String.t(), tuple(), float(), [integer()]) :: map()
  def q8_0_tensor(name, shape, scale, quants) do
    payload = pack_q8_0_blocks(scale, quants)
    tensor(name, shape, :q8_0, payload)
  end

  @spec pack_q4_0_blocks(float(), [integer()]) :: binary()
  def pack_q4_0_blocks(scale, quants) do
    if rem(length(quants), 32) != 0 do
      raise ArgumentError, "q4_0 quant list length must be a multiple of 32"
    end

    for chunk <- Enum.chunk_every(quants, 32), into: <<>> do
      lows = Enum.take(chunk, 16)
      highs = Enum.drop(chunk, 16)

      packed =
        for i <- 0..15 do
          low_n = (Enum.at(lows, i) + 8) &&& 0x0F
          high_n = (Enum.at(highs, i) + 8) &&& 0x0F
          low_n ||| (high_n <<< 4)
        end

      <<f32_to_f16_bits(scale)::little-16, IO.iodata_to_binary(packed)::binary>>
    end
  end

  @spec pack_q8_0_blocks(float(), [integer()]) :: binary()
  def pack_q8_0_blocks(scale, quants) do
    if rem(length(quants), 32) != 0 do
      raise ArgumentError, "q8_0 quant list length must be a multiple of 32"
    end

    for chunk <- Enum.chunk_every(quants, 32), into: <<>> do
      <<f32_to_f16_bits(scale)::little-16,
        IO.iodata_to_binary(for q <- chunk, do: <<q::signed-little-8>>)::binary>>
    end
  end

  @spec f32_to_f16_bits(number()) :: non_neg_integer()
  def f32_to_f16_bits(value) when is_integer(value), do: f32_to_f16_bits(value * 1.0)

  def f32_to_f16_bits(value) when is_float(value) do
    <<bits::unsigned-little-32>> = <<value::float-little-32>>

    sign = bits >>> 31 &&& 0x1
    exponent = bits >>> 23 &&& 0xFF
    mantissa = bits &&& 0x7FFFFF

    cond do
      exponent == 255 ->
        half_exp = 0x1F
        half_mant = if mantissa == 0, do: 0, else: 0x200
        sign <<< 15 ||| half_exp <<< 10 ||| half_mant

      true ->
        adjusted = exponent - 127 + 15

        cond do
          adjusted >= 31 ->
            sign <<< 15 ||| 0x1F <<< 10

          adjusted <= 0 ->
            if adjusted < -10 do
              sign <<< 15
            else
              mant = mantissa ||| 0x800000
              shift = 14 - adjusted
              half_mant = mant >>> shift
              round_bit = mant >>> (shift - 1) &&& 1
              half_mant = if round_bit == 1, do: half_mant + 1, else: half_mant
              sign <<< 15 ||| (half_mant &&& 0x3FF)
            end

          true ->
            half_exp = adjusted
            half_mant = mantissa >>> 13
            round = (mantissa &&& 0x1000) != 0
            half_mant = if round, do: half_mant + 1, else: half_mant

            {half_exp, half_mant} =
              if half_mant == 0x400 do
                {half_exp + 1, 0}
              else
                {half_exp, half_mant}
              end

            if half_exp >= 31 do
              sign <<< 15 ||| 0x1F <<< 10
            else
              sign <<< 15 ||| half_exp <<< 10 ||| half_mant
            end
        end
    end
  end

  defp maybe_add_alignment_metadata(metadata, nil), do: metadata

  defp maybe_add_alignment_metadata(metadata, alignment) do
    if Enum.any?(metadata, fn {key, _value} -> key == "general.alignment" end) do
      metadata
    else
      [{"general.alignment", {:u32, alignment}} | metadata]
    end
  end

  defp metadata_alignment(metadata) do
    case Enum.find(metadata, fn {key, _value} -> key == "general.alignment" end) do
      {"general.alignment", {tag, value}}
      when tag in [:u32, :u64] and is_integer(value) and value > 0 ->
        value

      _ ->
        32
    end
  end

  defp build_tensor_data_and_infos(tensors, alignment, version) do
    {infos, {_offset, data_blob}} =
      Enum.map_reduce(tensors, {0, <<>>}, fn tensor, {offset, blob} ->
        aligned = align_up(offset, alignment)
        pad = aligned - offset

        data = Map.fetch!(tensor, :data)
        type = Map.fetch!(tensor, :type)
        name = Map.fetch!(tensor, :name)
        shape = Map.fetch!(tensor, :shape)

        type_id =
          if is_integer(type) do
            type
          else
            {:ok, id} = Types.to_id(type)
            id
          end

        dims = shape |> Tuple.to_list() |> Enum.reverse()

        info = [
          encode_string(name, version),
          <<tuple_size(shape)::little-32>>,
          IO.iodata_to_binary(for dim <- dims, do: <<dim::little-64>>),
          <<type_id::little-32, aligned::little-64>>
        ]

        padded_blob = <<blob::binary, 0::size(pad * 8), data::binary>>

        {IO.iodata_to_binary(info), {aligned + byte_size(data), padded_blob}}
      end)

    {infos, data_blob}
  end

  defp encode_metadata_entry({key, value}, version) do
    {type_id, encoded_value} = encode_metadata_value(value, version)
    [encode_string(key, version), <<type_id::little-32>>, encoded_value]
  end

  defp encode_metadata_value({:u8, value}, _version),
    do: {@gguf_type.u8, <<value::unsigned-little-8>>}

  defp encode_metadata_value({:i8, value}, _version),
    do: {@gguf_type.i8, <<value::signed-little-8>>}

  defp encode_metadata_value({:u16, value}, _version),
    do: {@gguf_type.u16, <<value::unsigned-little-16>>}

  defp encode_metadata_value({:i16, value}, _version),
    do: {@gguf_type.i16, <<value::signed-little-16>>}

  defp encode_metadata_value({:u32, value}, _version),
    do: {@gguf_type.u32, <<value::unsigned-little-32>>}

  defp encode_metadata_value({:i32, value}, _version),
    do: {@gguf_type.i32, <<value::signed-little-32>>}

  defp encode_metadata_value({:f32, value}, _version),
    do: {@gguf_type.f32, <<value::float-little-32>>}

  defp encode_metadata_value({:bool, value}, _version),
    do: {@gguf_type.bool, <<if(value, do: 1, else: 0)>>}

  defp encode_metadata_value({:string, value}, version),
    do: {@gguf_type.string, encode_string(value, version)}

  defp encode_metadata_value({:u64, value}, _version),
    do: {@gguf_type.u64, <<value::unsigned-little-64>>}

  defp encode_metadata_value({:i64, value}, _version),
    do: {@gguf_type.i64, <<value::signed-little-64>>}

  defp encode_metadata_value({:f64, value}, _version),
    do: {@gguf_type.f64, <<value::float-little-64>>}

  defp encode_metadata_value({:array, element_tag, values}, version) do
    element_type = type_id_for_tag(element_tag)

    payload =
      IO.iodata_to_binary(
        for value <- values, do: encode_array_value(element_tag, value, version)
      )

    {@gguf_type.array, <<element_type::little-32, length(values)::little-64, payload::binary>>}
  end

  defp encode_metadata_value(value, version) when is_binary(value),
    do: encode_metadata_value({:string, value}, version)

  defp encode_metadata_value(value, version) when is_boolean(value),
    do: encode_metadata_value({:bool, value}, version)

  defp encode_metadata_value(value, version) when is_integer(value),
    do: encode_metadata_value({:i64, value}, version)

  defp encode_metadata_value(value, version) when is_float(value),
    do: encode_metadata_value({:f64, value}, version)

  defp encode_metadata_value(value, version) when is_list(value),
    do: encode_metadata_value({:array, :i64, value}, version)

  defp encode_array_value({:array, element_tag}, values, version) do
    {_type, encoded} = encode_metadata_value({:array, element_tag, values}, version)
    encoded
  end

  defp encode_array_value(:string, value, version), do: encode_string(value, version)
  defp encode_array_value(tag, value, _version), do: encode_primitive(tag, value)

  defp encode_primitive(:u8, value), do: <<value::unsigned-little-8>>
  defp encode_primitive(:i8, value), do: <<value::signed-little-8>>
  defp encode_primitive(:u16, value), do: <<value::unsigned-little-16>>
  defp encode_primitive(:i16, value), do: <<value::signed-little-16>>
  defp encode_primitive(:u32, value), do: <<value::unsigned-little-32>>
  defp encode_primitive(:i32, value), do: <<value::signed-little-32>>
  defp encode_primitive(:u64, value), do: <<value::unsigned-little-64>>
  defp encode_primitive(:i64, value), do: <<value::signed-little-64>>
  defp encode_primitive(:f32, value), do: <<value::float-little-32>>
  defp encode_primitive(:f64, value), do: <<value::float-little-64>>
  defp encode_primitive(:bool, value), do: <<if(value, do: 1, else: 0)>>

  defp type_id_for_tag({:array, _inner_tag}), do: @gguf_type.array
  defp type_id_for_tag(tag), do: Map.fetch!(@gguf_type, tag)

  defp encode_string(value, 2), do: <<byte_size(value)::little-32, value::binary>>
  defp encode_string(value, 3), do: <<byte_size(value)::little-64, value::binary>>

  defp align_up(offset, alignment) do
    rem = rem(offset, alignment)
    if rem == 0, do: offset, else: offset + (alignment - rem)
  end
end
