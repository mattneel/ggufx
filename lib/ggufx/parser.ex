defmodule GGUFX.Parser do
  @moduledoc false

  alias GGUFX.TensorInfo
  alias GGUFX.Types

  @type header :: %{
          version: non_neg_integer(),
          tensor_count: non_neg_integer(),
          metadata_kv_count: non_neg_integer()
        }

  @gguf_type_uint8 0
  @gguf_type_int8 1
  @gguf_type_uint16 2
  @gguf_type_int16 3
  @gguf_type_uint32 4
  @gguf_type_int32 5
  @gguf_type_float32 6
  @gguf_type_bool 7
  @gguf_type_string 8
  @gguf_type_array 9
  @gguf_type_uint64 10
  @gguf_type_int64 11
  @gguf_type_float64 12

  @spec parse_header(binary()) :: {:ok, header(), binary()} | {:error, term()}
  def parse_header(
        <<"GGUF", version::little-32, tensor_count::little-64, metadata_count::little-64,
          rest::binary>>
      )
      when version in [2, 3] do
    {:ok,
     %{
       version: version,
       tensor_count: tensor_count,
       metadata_kv_count: metadata_count
     }, rest}
  end

  def parse_header(<<"GGUF", _version::little-32, _::binary>>) do
    {:error, :unsupported_version}
  end

  def parse_header(_binary) do
    {:error, :invalid_magic}
  end

  @spec parse_string(binary(), pos_integer()) :: {:ok, String.t(), binary()} | {:error, term()}
  def parse_string(binary, 2), do: parse_string_with_len(binary, 32)
  def parse_string(binary, 3), do: parse_string_with_len(binary, 64)

  def parse_string(_binary, version) do
    {:error, {:parse_error, {:unsupported_version, version}}}
  end

  @spec parse_metadata(binary(), non_neg_integer(), pos_integer()) ::
          {:ok, %{String.t() => term()}, binary()} | {:error, term()}
  def parse_metadata(binary, count, version) when count >= 0 do
    do_parse_metadata(binary, count, version, %{})
  end

  @spec parse_value(binary(), non_neg_integer(), pos_integer()) ::
          {:ok, term(), binary()} | {:error, term()}
  def parse_value(binary, @gguf_type_uint8, _version), do: take_uint(binary, 8)
  def parse_value(binary, @gguf_type_int8, _version), do: take_int(binary, 8)
  def parse_value(binary, @gguf_type_uint16, _version), do: take_uint(binary, 16)
  def parse_value(binary, @gguf_type_int16, _version), do: take_int(binary, 16)
  def parse_value(binary, @gguf_type_uint32, _version), do: take_uint(binary, 32)
  def parse_value(binary, @gguf_type_int32, _version), do: take_int(binary, 32)
  def parse_value(binary, @gguf_type_float32, _version), do: take_float(binary, 32)

  def parse_value(<<raw::unsigned-little-8, rest::binary>>, @gguf_type_bool, _version) do
    {:ok, raw != 0, rest}
  end

  def parse_value(binary, @gguf_type_string, version), do: parse_string(binary, version)
  def parse_value(binary, @gguf_type_uint64, _version), do: take_uint(binary, 64)
  def parse_value(binary, @gguf_type_int64, _version), do: take_int(binary, 64)
  def parse_value(binary, @gguf_type_float64, _version), do: take_float(binary, 64)

  def parse_value(
        <<element_type::little-32, length::little-64, rest::binary>>,
        @gguf_type_array,
        version
      ) do
    parse_array_elements(rest, element_type, length, version, [])
  end

  def parse_value(_binary, type_id, _version) do
    {:error, {:parse_error, {:unknown_metadata_type, type_id}}}
  end

  @spec parse_tensor_infos(binary(), non_neg_integer(), pos_integer()) ::
          {:ok, %{String.t() => TensorInfo.t()}, binary()} | {:error, term()}
  def parse_tensor_infos(binary, count, version) when count >= 0 do
    do_parse_tensor_infos(binary, count, version, %{})
  end

  defp parse_string_with_len(binary, 32) do
    case binary do
      <<len::little-32, rest::binary>> ->
        take_binary_string(rest, len)

      _ ->
        {:error, {:parse_error, :string_length_truncated}}
    end
  end

  defp parse_string_with_len(binary, 64) do
    case binary do
      <<len::little-64, rest::binary>> ->
        take_binary_string(rest, len)

      _ ->
        {:error, {:parse_error, :string_length_truncated}}
    end
  end

  defp take_binary_string(rest, len) do
    if byte_size(rest) < len do
      {:error, {:parse_error, {:string_truncated, len}}}
    else
      <<value::binary-size(len), tail::binary>> = rest
      {:ok, value, tail}
    end
  end

  defp do_parse_metadata(binary, 0, _version, acc), do: {:ok, acc, binary}

  defp do_parse_metadata(binary, remaining, version, acc) do
    with {:ok, key, rest1} <- parse_string(binary, version),
         <<type_id::little-32, rest2::binary>> <- rest1,
         {:ok, value, rest3} <- parse_value(rest2, type_id, version) do
      do_parse_metadata(rest3, remaining - 1, version, Map.put(acc, key, value))
    else
      :error ->
        {:error, {:parse_error, :metadata_truncated}}

      {:error, _} = error ->
        error

      _ ->
        {:error, {:parse_error, :metadata_invalid}}
    end
  end

  defp parse_array_elements(binary, _element_type, 0, _version, acc) do
    {:ok, Enum.reverse(acc), binary}
  end

  defp parse_array_elements(binary, element_type, remaining, version, acc) do
    with {:ok, value, rest} <- parse_value(binary, element_type, version) do
      parse_array_elements(rest, element_type, remaining - 1, version, [value | acc])
    end
  end

  defp do_parse_tensor_infos(binary, 0, _version, acc), do: {:ok, acc, binary}

  defp do_parse_tensor_infos(binary, remaining, version, acc) do
    with {:ok, name, rest1} <- parse_string(binary, version),
         <<n_dims::little-32, rest2::binary>> <- rest1,
         {:ok, dims, rest3} <- parse_dims(rest2, n_dims, []),
         <<type_id::little-32, offset::little-64, rest4::binary>> <- rest3,
         {:ok, type} <- Types.from_id(type_id) do
      shape = dims |> Enum.reverse() |> List.to_tuple()
      n_elements = n_elements(shape)

      with {:ok, byte_size} <- Types.tensor_byte_size(type, n_elements) do
        info = %TensorInfo{
          name: name,
          shape: shape,
          type: type,
          offset: offset,
          byte_size: byte_size
        }

        do_parse_tensor_infos(rest4, remaining - 1, version, Map.put(acc, name, info))
      end
    else
      :error ->
        {:error, {:parse_error, :tensor_info_truncated}}

      {:error, :unknown_type} ->
        {:error, {:parse_error, :unknown_tensor_type}}

      {:error, _} = error ->
        error

      _ ->
        {:error, {:parse_error, :tensor_info_invalid}}
    end
  end

  defp parse_dims(binary, 0, acc), do: {:ok, Enum.reverse(acc), binary}

  defp parse_dims(<<dim::little-64, rest::binary>>, remaining, acc) do
    parse_dims(rest, remaining - 1, [dim | acc])
  end

  defp parse_dims(_binary, _remaining, _acc) do
    {:error, {:parse_error, :dims_truncated}}
  end

  defp take_uint(binary, size) do
    case size do
      8 ->
        case binary do
          <<value::unsigned-little-8, rest::binary>> -> {:ok, value, rest}
          _ -> {:error, {:parse_error, {:unsigned_truncated, size}}}
        end

      16 ->
        case binary do
          <<value::unsigned-little-16, rest::binary>> -> {:ok, value, rest}
          _ -> {:error, {:parse_error, {:unsigned_truncated, size}}}
        end

      32 ->
        case binary do
          <<value::unsigned-little-32, rest::binary>> -> {:ok, value, rest}
          _ -> {:error, {:parse_error, {:unsigned_truncated, size}}}
        end

      64 ->
        case binary do
          <<value::unsigned-little-64, rest::binary>> -> {:ok, value, rest}
          _ -> {:error, {:parse_error, {:unsigned_truncated, size}}}
        end
    end
  end

  defp take_int(binary, size) do
    case size do
      8 ->
        case binary do
          <<value::signed-little-8, rest::binary>> -> {:ok, value, rest}
          _ -> {:error, {:parse_error, {:signed_truncated, size}}}
        end

      16 ->
        case binary do
          <<value::signed-little-16, rest::binary>> -> {:ok, value, rest}
          _ -> {:error, {:parse_error, {:signed_truncated, size}}}
        end

      32 ->
        case binary do
          <<value::signed-little-32, rest::binary>> -> {:ok, value, rest}
          _ -> {:error, {:parse_error, {:signed_truncated, size}}}
        end

      64 ->
        case binary do
          <<value::signed-little-64, rest::binary>> -> {:ok, value, rest}
          _ -> {:error, {:parse_error, {:signed_truncated, size}}}
        end
    end
  end

  defp take_float(binary, size) do
    case size do
      32 ->
        case binary do
          <<value::float-little-32, rest::binary>> -> {:ok, value, rest}
          _ -> {:error, {:parse_error, {:float_truncated, size}}}
        end

      64 ->
        case binary do
          <<value::float-little-64, rest::binary>> -> {:ok, value, rest}
          _ -> {:error, {:parse_error, {:float_truncated, size}}}
        end
    end
  end

  defp n_elements({}), do: 1

  defp n_elements(shape) do
    shape
    |> Tuple.to_list()
    |> Enum.reduce(1, &Kernel.*/2)
  end
end
