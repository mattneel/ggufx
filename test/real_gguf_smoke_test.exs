defmodule GGUFX.RealGGUFSmokeTest do
  use ExUnit.Case

  @moduletag :real_gguf
  @real_smoke_enabled System.get_env("REAL_GGUF_SMOKE") in ["1", "true", "TRUE", "yes", "YES"]

  if !@real_smoke_enabled do
    @moduletag skip:
                 "set REAL_GGUF_SMOKE=1 to run real GGUF smoke tests (downloads a large GGUF file)"
  end

  @default_real_gguf_url "https://huggingface.co/hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF/resolve/main/llama-3.2-1b-instruct-q4_k_m.gguf"

  @tag timeout: :infinity
  test "downloads and validates a real quantized GGUF end-to-end" do
    path = ensure_real_gguf_downloaded!()

    assert {:ok, peek_model} = GGUFX.peek(path)
    assert peek_model.version in [2, 3]
    assert map_size(peek_model.tensor_info) > 0

    assert {:ok, model} = GGUFX.load(path, lazy: true)
    names = GGUFX.tensor_names(model)
    assert names != []

    # Fetch one real tensor to exercise positioned reads + dequant path.
    first_name = Enum.sort(names) |> hd()
    assert {:ok, tensor} = GGUFX.fetch_tensor(model, first_name)
    assert Nx.size(tensor) > 0

    architecture = peek_model.metadata["general.architecture"]
    assert is_binary(architecture) or is_nil(architecture)
  end

  defp ensure_real_gguf_downloaded! do
    url = System.get_env("REAL_GGUF_URL") || @default_real_gguf_url

    cache_dir =
      System.get_env("REAL_GGUF_CACHE_DIR") || Path.join(System.tmp_dir!(), "ggufx-real-smoke")

    File.mkdir_p!(cache_dir)

    filename =
      url
      |> URI.parse()
      |> Map.fetch!(:path)
      |> Path.basename()
      |> case do
        "" -> "real-smoke.gguf"
        name -> name
      end

    path = Path.join(cache_dir, filename)

    if usable_file?(path) do
      path
    else
      download_with_curl!(url, path)
      path
    end
  end

  defp usable_file?(path) do
    case File.stat(path) do
      {:ok, %{size: size}} when size > 0 -> true
      _ -> false
    end
  end

  defp download_with_curl!(url, path) do
    curl = System.find_executable("curl")

    if is_nil(curl) do
      raise "curl is required for REAL_GGUF_SMOKE tests"
    end

    token = System.get_env("HF_TOKEN") || System.get_env("HUGGINGFACEHUB_API_TOKEN")

    args = [
      "-L",
      "--fail",
      "--retry",
      "5",
      "--retry-delay",
      "2",
      "--continue-at",
      "-",
      "--output",
      path
    ]

    args =
      if token && token != "" do
        ["-H", "Authorization: Bearer #{token}" | args]
      else
        args
      end

    {output, exit_code} = System.cmd(curl, args ++ [url], stderr_to_stdout: true)

    if exit_code != 0 do
      raise "failed to download real GGUF file (exit #{exit_code}):\n#{output}"
    end
  end
end
