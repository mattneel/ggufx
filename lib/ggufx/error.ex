defmodule GGUFX.Error do
  @moduledoc """
  Exception raised by bang variants of the public API.
  """

  defexception [:reason, :message]

  @impl true
  @spec exception(term()) :: %__MODULE__{}
  def exception(reason) do
    %__MODULE__{reason: reason, message: format_reason(reason)}
  end

  @spec format_reason(term()) :: String.t()
  def format_reason(reason) do
    "GGUFX error: #{inspect(reason)}"
  end
end
