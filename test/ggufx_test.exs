defmodule GGUFXTest do
  use ExUnit.Case

  alias GGUFX.Metadata
  alias GGUFX.TestHelpers

  test "loads metadata and f32 tensors" do
    metadata = [
      {"general.architecture", {:string, "llama"}},
      {"llama.context_length", {:u32, 4096}},
      {"llama.embedding_length", {:u32, 1024}},
      {"llama.attention.head_count", {:u32, 16}},
      {"llama.block_count", {:u32, 24}},
      {"tokenizer.vocab_size", {:u32, 32000}}
    ]

    tensor = TestHelpers.f32_tensor("blk.0.w", {2, 2}, [1.0, 2.0, 3.0, 4.0])
    path = write_fixture(TestHelpers.build_gguf(metadata, [tensor]))

    assert {:ok, model} = GGUFX.load(path)

    assert model.metadata["general.architecture"] == "llama"
    assert model.version == 3
    assert model.tensors |> Map.has_key?("blk.0.w")
    assert model |> GGUFX.tensor_names() == ["blk.0.w"]
    assert model |> GGUFX.metadata() |> Map.get("llama.context_length") == 4096

    t = model.tensors["blk.0.w"]
    assert Nx.shape(t) == {2, 2}
    assert Nx.to_flat_list(t) == [1.0, 2.0, 3.0, 4.0]

    assert Metadata.architecture(model) == "llama"
    assert Metadata.context_length(model) == 4096
    assert Metadata.embedding_length(model) == 1024
    assert Metadata.head_count(model) == 16
    assert Metadata.block_count(model) == 24
    assert Metadata.vocab_size(model) == 32000
  end

  test "supports v2 strings" do
    metadata = [{"general.architecture", {:string, "llama"}}]
    tensor = TestHelpers.f32_tensor("w", {1}, [42.0])
    path = write_fixture(TestHelpers.build_gguf(metadata, [tensor], version: 2))

    assert {:ok, model} = GGUFX.load(path)
    assert model.version == 2
    assert model.metadata["general.architecture"] == "llama"
  end

  test "peek only parses structure and metadata" do
    tensor = TestHelpers.f32_tensor("w", {2}, [10.0, 20.0])
    path = write_fixture(TestHelpers.build_gguf([], [tensor]))

    assert {:ok, model} = GGUFX.peek(path)
    assert model.tensors == nil
    assert model.source == Path.expand(path)
    assert Map.has_key?(model.tensor_info, "w")
  end

  test "lazy load fetches one tensor via fetch_tensor" do
    t0 = TestHelpers.f32_tensor("a", {2}, [1.0, 2.0])
    t1 = TestHelpers.f32_tensor("b", {2}, [3.0, 4.0])
    path = write_fixture(TestHelpers.build_gguf([], [t0, t1]))

    assert {:ok, model} = GGUFX.load(path, lazy: true)
    assert model.tensors == nil

    assert {:ok, tensor} = GGUFX.fetch_tensor(model, "b")
    assert Nx.to_flat_list(tensor) == [3.0, 4.0]
  end

  test "tensor filter only loads matching tensors" do
    t0 = TestHelpers.f32_tensor("blk.0.w", {1}, [1.0])
    t1 = TestHelpers.f32_tensor("blk.1.w", {1}, [2.0])
    path = write_fixture(TestHelpers.build_gguf([], [t0, t1]))

    assert {:ok, model} =
             GGUFX.load(path,
               tensor_filter: fn name -> String.starts_with?(name, "blk.0") end
             )

    assert Map.keys(model.tensors) == ["blk.0.w"]
    assert Map.keys(model.tensor_info) |> Enum.sort() == ["blk.0.w", "blk.1.w"]
  end

  test "honors non-default alignment metadata" do
    metadata = [{"general.alignment", {:u32, 64}}]
    t0 = TestHelpers.f32_tensor("a", {1}, [7.0])
    t1 = TestHelpers.f32_tensor("b", {1}, [8.0])

    path = write_fixture(TestHelpers.build_gguf(metadata, [t0, t1], alignment: 64))

    assert {:ok, model} = GGUFX.load(path, lazy: true)
    assert {:ok, tensor} = GGUFX.fetch_tensor(model, "b")
    assert Nx.to_flat_list(tensor) == [8.0]
  end

  test "returns errors and bang variants raise" do
    path = write_fixture(<<"NOPE">>)

    assert {:error, :invalid_magic} = GGUFX.load(path)
    assert_raise GGUFX.Error, fn -> GGUFX.load!(path) end

    assert {:ok, model} = GGUFX.load(write_fixture(TestHelpers.build_gguf([], [])), lazy: true)
    assert {:error, :tensor_not_found} = GGUFX.fetch_tensor(model, "missing")
    assert_raise GGUFX.Error, fn -> GGUFX.fetch_tensor!(model, "missing") end
  end

  defp write_fixture(binary) do
    name = "ggufx-#{System.unique_integer([:positive, :monotonic])}.gguf"
    path = Path.join(System.tmp_dir!(), name)
    File.write!(path, binary)
    on_exit(fn -> File.rm(path) end)
    path
  end
end
