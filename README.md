# mcp-server-hybrid-search

[![CI](https://github.com/wonder-soft/mcp-server-hybrid-search/actions/workflows/ci.yml/badge.svg)](https://github.com/wonder-soft/mcp-server-hybrid-search/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

MCP Server for hybrid document search (Qdrant vector search + Tantivy BM25) with SSE transport.

## Architecture

- **MCP Server** (`mcp-server-hybrid-search`): SSE-based MCP server on port 7070 providing `search` and `get` tools
- **CLI** (`ragctl`): Document indexer that ingests md/txt/pdf/xlsx/docx files into Qdrant and Tantivy
- **Qdrant**: Vector database for semantic search
- **Tantivy**: Full-text search engine for BM25 ranking

## Prerequisites

- Rust toolchain (1.75+)
- Docker (for Qdrant)
- OpenAI API key (for embeddings, not needed with `--features local-embed`)
- Python + `markitdown` (for PDF/Excel/Word support): `pip install markitdown`

## Quick Start

### 1. Start Qdrant

```bash
docker compose up -d
```

### 2. Set up environment

```bash
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY
```

### 3. Build

```bash
# Default build (OpenAI embeddings, whitespace-based BM25 tokenizer)
cargo build --release

# With Japanese tokenizer for BM25 full-text search
cargo build --release --features ja

# With local embedding (no OpenAI API key needed)
cargo build --release --features local-embed

# Combine features as needed
cargo build --release --features "ja,local-embed"
```

### 4. Initialize

Create the default source directory and data directories:

```bash
./target/release/ragctl init
```

This creates:
- `~/.local/share/mcp-hybrid-search/` — default document source directory
- `~/.mcp-hybrid-search/tantivy/` — Tantivy index directory

### 5. Place documents

Copy or symlink your documents into the default source directory:

```bash
cp ~/my-docs/*.md ~/.local/share/mcp-hybrid-search/
cp ~/reports/*.pdf ~/.local/share/mcp-hybrid-search/
cp ~/data/*.xlsx ~/.local/share/mcp-hybrid-search/
```

### 6. Ingest documents

```bash
# Use default source directory (~/.local/share/mcp-hybrid-search)
./target/release/ragctl ingest

# Or specify source directories explicitly
./target/release/ragctl ingest \
  --source /path/to/your/docs \
  --source /path/to/more/docs
```

### 7. Start MCP Server

```bash
./target/release/mcp-server-hybrid-search
```

The server will listen on `http://localhost:7070`.

### 8. Connect from Claude Code

Add to your Claude Code MCP configuration:

```json
{
  "mcpServers": {
    "hybrid-search": {
      "url": "http://localhost:7070/sse"
    }
  }
}
```

## CLI Usage

### Initialize directories

```bash
ragctl init
```

Creates the default source directory (`~/.local/share/mcp-hybrid-search/`) and the Tantivy index directory. Run this once before first use.

### Ingest documents

```bash
# Default source directory
ragctl ingest

# Custom source directories
ragctl ingest \
  --source /path/to/docs \
  --source /path/to/converted \
  --qdrant http://localhost:6334 \
  --index-dir ~/.mcp-hybrid-search/tantivy \
  --chunk-size 1000 \
  --chunk-overlap 200
```

Supported file types:
- **Direct**: `.md`, `.txt`
- **Via markitdown**: `.pdf`, `.xlsx`, `.xls`, `.docx`, `.pptx`, `.csv`, `.html`

### Check status

```bash
ragctl status
```

### Export data

Export all indexed chunks (with embeddings) to a JSON file for sharing with other engineers:

```bash
ragctl export --output ./exported-data.json
```

The exported file contains all chunk payloads and their embedding vectors. Other engineers can import this without needing an OpenAI API key.

### Import data

Import previously exported data into Qdrant and Tantivy:

```bash
ragctl import --input ./exported-data.json
```

This populates both Qdrant (vectors) and Tantivy (BM25 index) from the export file.

### Search (debug)

```bash
ragctl search --query "your search query" --top-k 10
```

## MCP Tools

### search

Hybrid search across indexed documents using vector similarity + BM25 ranking with RRF fusion.

**Input:**
- `query` (string, required): Search query
- `top_k` (number, optional): Number of results (default: 10)
- `filters` (object, optional):
  - `source_type` (string): Filter by file type (md/txt/pdf/xlsx)
  - `path_prefix` (string): Filter by path prefix

### get

Retrieve full content of a document chunk.

**Input:**
- `chunk_id` (string, required): Chunk identifier

## Configuration

Edit `config.toml`:

| Key | Default | Description |
|-----|---------|-------------|
| `qdrant_url` | `http://localhost:6334` | Qdrant gRPC URL |
| `collection_name` | `docs` | Qdrant collection name |
| `tantivy_index_dir` | `~/.mcp-hybrid-search/tantivy` | Tantivy index directory |
| `chunk_size` | `1000` | Chunk size in characters |
| `chunk_overlap` | `200` | Chunk overlap in characters |
| `listen_port` | `7070` | MCP server port |
| `embedding_provider` | `openai` | Embedding provider (see below) |
| `embedding_model` | `text-embedding-3-small` | OpenAI embedding model |
| `embedding_dimension` | `1536` | Embedding vector dimension |
| `tokenizer` | `default` | BM25 tokenizer (see below) |

Default source directory: `~/.local/share/mcp-hybrid-search/`

### Tokenizer

The `tokenizer` config controls how Tantivy splits text for BM25 full-text search. The default tokenizer is whitespace-based, which works well for English but poorly for CJK languages (Japanese, Korean, Chinese) where words are not separated by spaces.

| Value | Feature flag | Dictionary |
|-------|-------------|------------|
| `default` | *(none)* | Whitespace-based (built-in) |
| `japanese` | `--features ja` | IPADIC |
| `korean` | `--features ko` | ko-dic |
| `chinese` | `--features zh` | CC-CEDICT |

Language dictionaries are embedded into the binary at build time via [Lindera](https://github.com/lindera/lindera). Only enable the features you need — each adds ~50MB to the binary.

> **Note:** Changing the tokenizer requires rebuilding the Tantivy index. Run `ragctl reset` then `ragctl ingest` after switching tokenizers.

### Embedding Provider

| Provider | Feature flag | Model | Dimension | API key required |
|----------|-------------|-------|-----------|-----------------|
| `openai` | *(none)* | `text-embedding-3-small` | 1536 | Yes (`OPENAI_API_KEY`) |
| `local` | `--features local-embed` | `intfloat/multilingual-e5-small` | 384 | No |
| `local` | `--features local-embed` | `intfloat/multilingual-e5-base` | 768 | No |

The local provider uses [fastembed](https://github.com/Anush008/fastembed-rs) with ONNX Runtime. Models are automatically downloaded and cached on first use.

To use local embeddings:

```toml
# config.toml — multilingual-e5-small (lighter, faster)
embedding_provider = "local"
embedding_model = "multilingual-e5-small"
embedding_dimension = 384
```

```toml
# config.toml — multilingual-e5-base (better accuracy)
embedding_provider = "local"
embedding_model = "multilingual-e5-base"
embedding_dimension = 768
```

```bash
cargo build --release --features local-embed
```

> **Note:** Switching embedding provider changes the vector dimension. Run `ragctl reset` then `ragctl ingest` after switching.

## Search Algorithm

1. Query is embedded using the configured embedding provider
2. Qdrant vector search returns top 30 candidates
3. Tantivy BM25 search returns top 30 candidates
4. Results are merged using Reciprocal Rank Fusion (RRF) with k=60
5. Top N results are returned (default 10)
