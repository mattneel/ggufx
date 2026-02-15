defmodule GGUFX.ParserTest do
  use ExUnit.Case, async: true

  alias GGUFX.Parser
  alias GGUFX.TestHelpers

  test "parses valid header for v3" do
    binary = TestHelpers.build_gguf([], [])

    assert {:ok, header, _rest} = Parser.parse_header(binary)
    assert header.version == 3
    assert header.tensor_count == 0
    assert header.metadata_kv_count == 0
  end

  test "rejects invalid magic" do
    assert {:error, :invalid_magic} = Parser.parse_header(<<"NOPE", 0::little-32>>)
  end

  test "rejects unsupported version" do
    bad = <<"GGUF", 99::little-32, 0::little-64, 0::little-64>>
    assert {:error, :unsupported_version} = Parser.parse_header(bad)
  end

  test "parses v2 strings and metadata" do
    metadata = [
      {"general.architecture", {:string, "llama"}},
      {"llama.context_length", {:u32, 4096}},
      {"some.array", {:array, :i64, [1, 2, 3]}}
    ]

    binary = TestHelpers.build_gguf(metadata, [], version: 2)

    assert {:ok, header, rest1} = Parser.parse_header(binary)
    assert header.version == 2

    assert {:ok, parsed_metadata, _rest} =
             Parser.parse_metadata(rest1, header.metadata_kv_count, header.version)

    assert parsed_metadata["general.architecture"] == "llama"
    assert parsed_metadata["llama.context_length"] == 4096
    assert parsed_metadata["some.array"] == [1, 2, 3]
  end

  test "parses all scalar metadata value types" do
    metadata = [
      {"u8", {:u8, 255}},
      {"i8", {:i8, -1}},
      {"u16", {:u16, 65_535}},
      {"i16", {:i16, -2}},
      {"u32", {:u32, 123}},
      {"i32", {:i32, -123}},
      {"f32", {:f32, 1.25}},
      {"bool", {:bool, true}},
      {"string", {:string, "ok"}},
      {"u64", {:u64, 9_876_543_210}},
      {"i64", {:i64, -9_876_543_210}},
      {"f64", {:f64, 3.5}},
      {"arr", {:array, :i16, [1, -2, 3]}}
    ]

    binary = TestHelpers.build_gguf(metadata, [])

    assert {:ok, header, rest1} = Parser.parse_header(binary)

    assert {:ok, parsed_metadata, _rest} =
             Parser.parse_metadata(rest1, header.metadata_kv_count, header.version)

    assert parsed_metadata["u8"] == 255
    assert parsed_metadata["i8"] == -1
    assert parsed_metadata["u16"] == 65_535
    assert parsed_metadata["i16"] == -2
    assert parsed_metadata["u32"] == 123
    assert parsed_metadata["i32"] == -123
    assert parsed_metadata["f32"] == 1.25
    assert parsed_metadata["bool"] == true
    assert parsed_metadata["string"] == "ok"
    assert parsed_metadata["u64"] == 9_876_543_210
    assert parsed_metadata["i64"] == -9_876_543_210
    assert parsed_metadata["f64"] == 3.5
    assert parsed_metadata["arr"] == [1, -2, 3]
  end

  test "parses nested metadata arrays" do
    metadata = [
      {"nested", {:array, {:array, :u8}, [[1, 2], [3, 4]]}}
    ]

    binary = TestHelpers.build_gguf(metadata, [])
    assert {:ok, header, rest1} = Parser.parse_header(binary)

    assert {:ok, parsed_metadata, _rest} =
             Parser.parse_metadata(rest1, header.metadata_kv_count, header.version)

    assert parsed_metadata["nested"] == [[1, 2], [3, 4]]
  end

  test "parses tensor infos and reverses shape" do
    tensor = TestHelpers.f32_tensor("w", {4, 3, 2}, Enum.to_list(1..24))
    binary = TestHelpers.build_gguf([], [tensor])

    assert {:ok, header, rest1} = Parser.parse_header(binary)

    assert {:ok, _metadata, rest2} =
             Parser.parse_metadata(rest1, header.metadata_kv_count, header.version)

    assert {:ok, tensor_info, _rest} =
             Parser.parse_tensor_infos(rest2, header.tensor_count, header.version)

    assert tensor_info["w"].shape == {4, 3, 2}
    assert tensor_info["w"].byte_size == 24 * 4
  end
end
