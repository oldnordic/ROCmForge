# External Integrations

**Analysis Date:** 2026-01-18

## APIs & External Services

**HuggingFace Tokenizers:**
- Tokenizer library for text encoding/decoding
- SDK/Client: `tokenizers` crate from HuggingFace
- Auth: None required (local tokenization)
- Used in: `src/tokenizer.rs`

**GGUF Format Support:**
- Model format for LLM weights
- Implementation: Custom parser in `src/loader/gguf.rs`
- No external API - pure Rust implementation

## Data Storage

**Model Files:**
- GGUF format model files
- Location: Configured via `ROCMFORGE_MODELS` env var
- Client: Memory-mapped I/O via `memmap2`

**Tokenizer Files:**
- HuggingFace tokenizer.json
- Location: Configured via `ROCMFORGE_TOKENIZER` env var
- Client: `tokenizers` crate

**No Database:**
- No SQL/NoSQL database
- No external storage services
- All data stored in local files

## Hardware Integration

**AMD ROCm/HIP:**
- GPU runtime for AMD GPUs
- SDK: ROCm (HIP, hipBLAS, HIPRTC)
- Linked libraries: amdhip64, hipblas, hiprtc (`build.rs:9-13`)
- Target architecture: gfx1100 (RX 7900 XT default)

**GPU Kernels:**
- Compiled at build time via HIPRTC
- 14 kernels in `build.rs:42-95`
- Operations: matmul, softmax, rope, swiglu, etc.

## Development Tools

**CodeMCP (AI Assistant):**
- LLM-powered code assistance
- Config: `.codemcp/config.toml`
- Ollama integration for local LLM
- BGE embeddings for semantic indexing

**Build Tools:**
- Make - Build automation
- Cargo - Rust package manager
- Criterion - Benchmarking framework

## Environment Configuration

**Development:**
- Required env vars:
  - `ROCMFORGE_GGUF` - Path to GGUF model file
  - `ROCMFORGE_TOKENIZER` - Path to tokenizer.json
  - `ROCMFORGE_MODELS` - Directory containing models
- Secrets location: None (no authentication)
- Hardware: AMD GPU with ROCm required

**Production:**
- Same as development (no separate environments)
- No cloud deployment configured
- No secrets management (no external services)

## CI/CD & Deployment

**CI Pipeline:**
- Not detected (no .github/workflows/ or similar)

**Hosting:**
- Not applicable (local execution only)

## External Dependencies (None)

**Payment Processing:** None
**Email/SMS:** None
**Authentication:** None
**Monitoring:** None
**Analytics:** None

## Build Dependencies

**HIP Runtime:**
- System libraries: amdhip64, hipblas, hiprtc
- Linked at build time via `build.rs`

**Optional:**
- ROCm SDK for GPU development
- AMD GPU hardware for testing

---

*Integration audit: 2026-01-18*
*Update when adding/removing external services*
