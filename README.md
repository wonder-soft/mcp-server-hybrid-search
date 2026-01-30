# mcp-server-hybrid-search

MCP Server for hybrid document search (Qdrant vector search + Tantivy BM25) with SSE transport.

## Architecture

- **MCP Server** (`mcp-server-hybrid-search`): SSE-based MCP server on port 7070 providing `search` and `get` tools
- **CLI** (`ragctl`): Document indexer that ingests md/txt files into Qdrant and Tantivy
- **Qdrant**: Vector database for semantic search
- **Tantivy**: Full-text search engine for BM25 ranking

## Prerequisites

- Rust toolchain (1.75+)
- Docker (for Qdrant)
- OpenAI API key (for embeddings)

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
cargo build --release
```

### 4. Ingest documents

```bash
./target/release/ragctl ingest \
  --source /path/to/your/docs \
  --source /path/to/more/docs
```

### 5. Start MCP Server

```bash
./target/release/mcp-server-hybrid-search
```

The server will listen on `http://localhost:7070`.

### 6. Connect from Claude Code

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

### Ingest documents

```bash
ragctl ingest \
  --source /path/to/docs \
  --source /path/to/converted \
  --qdrant http://localhost:6334 \
  --index-dir ~/.mcp-hybrid-search/tantivy \
  --chunk-size 1000 \
  --chunk-overlap 200
```

### Check status

```bash
ragctl status
```

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
  - `source_type` (string): Filter by file type (md/txt)
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
| `embedding_model` | `text-embedding-3-small` | OpenAI embedding model |
| `embedding_dimension` | `1536` | Embedding vector dimension |

## Search Algorithm

1. Query is embedded using OpenAI embedding API
2. Qdrant vector search returns top 30 candidates
3. Tantivy BM25 search returns top 30 candidates
4. Results are merged using Reciprocal Rank Fusion (RRF) with k=60
5. Top N results are returned (default 10)
