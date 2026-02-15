defmodule GGUFX.TensorInfo do
  @moduledoc """
  Describes a tensor's location and format within a GGUF file.
  """

  alias GGUFX.Types

  @type t :: %__MODULE__{
          name: String.t(),
          shape: tuple(),
          type: Types.ggml_type(),
          offset: non_neg_integer(),
          byte_size: non_neg_integer()
        }

  defstruct [:name, :shape, :type, :offset, :byte_size]
end
