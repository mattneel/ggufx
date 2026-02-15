defmodule GGUFX.Metadata do
  @moduledoc """
  Convenience accessors for common GGUF metadata fields.
  """

  alias GGUFX

  @doc "Get the model architecture (for example, `\"llama\"`)."
  @spec architecture(GGUFX.t()) :: String.t() | nil
  def architecture(%GGUFX{metadata: metadata}) do
    get_first(metadata, ["general.architecture", "architecture"])
  end

  @doc "Get the context length."
  @spec context_length(GGUFX.t()) :: pos_integer() | nil
  def context_length(%GGUFX{metadata: metadata}) do
    get_integer(metadata, ["llama.context_length", "general.context_length", "context_length"])
  end

  @doc "Get the embedding/hidden size."
  @spec embedding_length(GGUFX.t()) :: pos_integer() | nil
  def embedding_length(%GGUFX{metadata: metadata}) do
    get_integer(metadata, [
      "llama.embedding_length",
      "general.embedding_length",
      "embedding_length"
    ])
  end

  @doc "Get the number of attention heads."
  @spec head_count(GGUFX.t()) :: pos_integer() | nil
  def head_count(%GGUFX{metadata: metadata}) do
    get_integer(metadata, [
      "llama.attention.head_count",
      "attention.head_count",
      "head_count"
    ])
  end

  @doc "Get the number of transformer blocks/layers."
  @spec block_count(GGUFX.t()) :: pos_integer() | nil
  def block_count(%GGUFX{metadata: metadata}) do
    get_integer(metadata, ["llama.block_count", "general.block_count", "block_count"])
  end

  @doc "Get vocabulary size."
  @spec vocab_size(GGUFX.t()) :: pos_integer() | nil
  def vocab_size(%GGUFX{metadata: metadata}) do
    case get_first(metadata, ["tokenizer.ggml.tokens", "tokenizer.vocab_size", "vocab_size"]) do
      list when is_list(list) -> length(list)
      value when is_integer(value) and value > 0 -> value
      _ -> nil
    end
  end

  defp get_integer(metadata, keys) do
    case get_first(metadata, keys) do
      value when is_integer(value) and value > 0 -> value
      _ -> nil
    end
  end

  defp get_first(metadata, keys) do
    Enum.find_value(keys, fn key -> Map.get(metadata, key) end)
  end
end
