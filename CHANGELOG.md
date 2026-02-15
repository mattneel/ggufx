# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

### Added
- GGUF v2 and v3 parser with metadata and tensor info extraction.
- Public API for eager loading, lazy loading, tensor fetch, and metadata access.
- Dequantization support for `f16`, `bf16`, `q4_0`, `q8_0`, `q4_k`, and `q6_k`.
- Metadata convenience helpers for architecture and core model dimensions.
- Hand-crafted GGUF fixture builder and comprehensive test suite.
