defmodule GGUFX do
  @moduledoc """
  GGUF file parser for Elixir.

  Parses GGUF model files and returns metadata and tensors as Nx tensors.
  """

  alias GGUFX.Dequantize
  alias GGUFX.Error
  alias GGUFX.Parser
  alias GGUFX.TensorInfo
  alias GGUFX.Types

  @type metadata_value ::
          integer() | float() | boolean() | String.t() | [metadata_value()]

  @type t :: %__MODULE__{
          version: pos_integer(),
          metadata: %{String.t() => metadata_value()},
          tensor_info: %{String.t() => TensorInfo.t()},
          tensors: %{String.t() => Nx.Tensor.t()} | nil,
          source: String.t() | nil,
          tensor_data_offset: non_neg_integer() | nil
        }

  defstruct [:version, :metadata, :tensor_info, :tensors, :source, :tensor_data_offset]

  @doc "Load and parse a GGUF file. Options: `:lazy`, `:tensor_filter`, `:dequantize` (default true)."
  @spec load(Path.t(), keyword()) :: {:ok, t()} | {:error, term()}
  def load(path, opts \\ []) do
    lazy? = Keyword.get(opts, :lazy, false)
    dequantize? = Keyword.get(opts, :dequantize, true)
    filter = Keyword.get(opts, :tensor_filter, fn _name -> true end)

    with {:ok, binary} <- read_file(path),
         {:ok, parsed} <- parse_binary(binary),
         {:ok, tensors} <- maybe_load_tensors(binary, parsed, lazy?, filter, dequantize?) do
      {:ok,
       %__MODULE__{
         version: parsed.version,
         metadata: parsed.metadata,
         tensor_info: parsed.tensor_info,
         tensors: tensors,
         source: if(lazy?, do: Path.expand(path), else: nil),
         tensor_data_offset: parsed.tensor_data_offset
       }}
    end
  end

  @doc "Same as `load/2` but raises on error."
  @spec load!(Path.t(), keyword()) :: t()
  def load!(path, opts \\ []) do
    case load(path, opts) do
      {:ok, model} -> model
      {:error, reason} -> raise(Error, reason)
    end
  end

  @doc "Fetch a single tensor from a lazily-loaded model."
  @spec fetch_tensor(t(), String.t()) :: {:ok, Nx.Tensor.t()} | {:error, term()}
  def fetch_tensor(%__MODULE__{tensors: tensors} = model, name) when is_map(tensors) do
    case Map.fetch(tensors, name) do
      {:ok, tensor} -> {:ok, tensor}
      :error -> fetch_tensor_from_source(model, name)
    end
  end

  def fetch_tensor(%__MODULE__{} = model, name), do: fetch_tensor_from_source(model, name)

  @doc "Same as `fetch_tensor/2` but raises on error."
  @spec fetch_tensor!(t(), String.t()) :: Nx.Tensor.t()
  def fetch_tensor!(%__MODULE__{} = model, name) do
    case fetch_tensor(model, name) do
      {:ok, tensor} -> tensor
      {:error, reason} -> raise(Error, reason)
    end
  end

  @doc "List all tensor names without loading tensor data."
  @spec tensor_names(t()) :: [String.t()]
  def tensor_names(%__MODULE__{tensor_info: tensor_info}) do
    Map.keys(tensor_info)
  end

  @doc "Return metadata as a map."
  @spec metadata(t()) :: %{String.t() => metadata_value()}
  def metadata(%__MODULE__{metadata: metadata}), do: metadata

  @doc "Parse only file structure and metadata without loading tensor values."
  @spec peek(Path.t()) :: {:ok, t()} | {:error, term()}
  def peek(path), do: load(path, lazy: true)

  defp read_file(path) do
    case File.read(path) do
      {:ok, binary} -> {:ok, binary}
      {:error, :enoent} -> {:error, :file_not_found}
      {:error, reason} -> {:error, reason}
    end
  end

  defp parse_binary(binary) do
    with {:ok, header, rest1} <- Parser.parse_header(binary),
         {:ok, metadata, rest2} <-
           Parser.parse_metadata(rest1, header.metadata_kv_count, header.version),
         {:ok, tensor_info, rest3} <-
           Parser.parse_tensor_infos(rest2, header.tensor_count, header.version) do
      alignment = alignment(metadata)
      consumed = byte_size(binary) - byte_size(rest3)
      tensor_data_offset = align_up(consumed, alignment)

      if byte_size(binary) < tensor_data_offset do
        {:error, {:parse_error, :tensor_data_offset_out_of_bounds}}
      else
        {:ok,
         %{
           version: header.version,
           metadata: metadata,
           tensor_info: tensor_info,
           tensor_data_offset: tensor_data_offset
         }}
      end
    end
  end

  defp maybe_load_tensors(_binary, _parsed, true, _filter, _dequantize?), do: {:ok, nil}

  defp maybe_load_tensors(binary, parsed, false, filter, dequantize?) do
    Enum.reduce_while(parsed.tensor_info, {:ok, %{}}, fn {name, info}, {:ok, acc} ->
      if filter.(name) do
        case decode_tensor_from_binary(binary, info, parsed.tensor_data_offset, dequantize?) do
          {:ok, tensor} -> {:cont, {:ok, Map.put(acc, name, tensor)}}
          {:error, _} = error -> {:halt, error}
        end
      else
        {:cont, {:ok, acc}}
      end
    end)
  end

  defp fetch_tensor_from_source(
         %__MODULE__{source: source, tensor_info: tensor_info, tensor_data_offset: base_offset},
         name
       )
       when is_binary(source) and is_integer(base_offset) do
    with {:ok, info} <- map_fetch(tensor_info, name),
         {:ok, tensor_binary} <- pread_tensor(source, base_offset + info.offset, info.byte_size),
         {:ok, tensor} <- decode_tensor_data(tensor_binary, info, true) do
      {:ok, maybe_reshape(tensor, info.shape)}
    end
  end

  defp fetch_tensor_from_source(_model, _name), do: {:error, :tensor_not_found}

  defp decode_tensor_from_binary(binary, %TensorInfo{} = info, tensor_data_offset, dequantize?) do
    start = tensor_data_offset + info.offset
    stop = start + info.byte_size

    if stop > byte_size(binary) do
      {:error, {:parse_error, {:tensor_out_of_bounds, info.name}}}
    else
      bytes = binary_part(binary, start, info.byte_size)

      with {:ok, tensor} <- decode_tensor_data(bytes, info, dequantize?) do
        {:ok, maybe_reshape(tensor, info.shape)}
      end
    end
  end

  defp decode_tensor_data(bytes, %TensorInfo{type: type, shape: shape}, dequantize?) do
    count = n_elements(shape)

    if dequantize? or unquantized?(type) do
      Dequantize.dequantize(bytes, type, count)
    else
      {:ok, Nx.from_binary(bytes, {:u, 8})}
    end
  end

  defp pread_tensor(path, position, size) do
    case :file.open(String.to_charlist(path), [:binary, :read]) do
      {:ok, file} ->
        try do
          case :file.pread(file, position, size) do
            {:ok, binary} -> {:ok, binary}
            :eof -> {:error, {:parse_error, :tensor_data_truncated}}
            {:error, reason} -> {:error, reason}
          end
        after
          :file.close(file)
        end

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp unquantized?(type) do
    match?({:ok, _}, Types.to_nx_type(type))
  end

  defp maybe_reshape(tensor, shape) do
    if tuple_size(shape) == 0 or Nx.size(tensor) != n_elements(shape) do
      tensor
    else
      Nx.reshape(tensor, shape)
    end
  end

  defp n_elements(shape) do
    shape
    |> Tuple.to_list()
    |> Enum.reduce(1, &Kernel.*/2)
  end

  defp align_up(offset, alignment) do
    rem = rem(offset, alignment)
    if rem == 0, do: offset, else: offset + (alignment - rem)
  end

  defp alignment(metadata) do
    case Map.get(metadata, "general.alignment") do
      value when is_integer(value) and value > 0 -> value
      _ -> 32
    end
  end

  defp map_fetch(map, key) do
    case Map.fetch(map, key) do
      {:ok, value} -> {:ok, value}
      :error -> {:error, :tensor_not_found}
    end
  end
end
