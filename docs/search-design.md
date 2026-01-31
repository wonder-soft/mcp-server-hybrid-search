# Search Design

This document describes the search architecture, chunking strategy, and ranking algorithm used in mcp-server-hybrid-search.

## Overview

Queries go through a **hybrid search** pipeline that combines two independent retrieval engines and fuses their results:

```
Query
  ├─ Embedding API ─→ Qdrant (vector search)  ─→ top 30 candidates
  ├─ Raw text       ─→ Tantivy (BM25 search)  ─→ top 30 candidates
  └─ RRF Fusion (k=60) ─→ top_k results returned to client
```

This two-stage design — broad internal retrieval followed by rank fusion — is central to result quality.

## Chunking Strategy

Documents are split into fixed-size character chunks with overlap before indexing.

| Parameter | Default | Config key |
|-----------|---------|------------|
| Chunk size | 1000 characters | `chunk_size` |
| Chunk overlap | 200 characters | `chunk_overlap` |

### How chunking works

1. The document text is split into chunks of `chunk_size` characters
2. Each subsequent chunk starts `chunk_size - chunk_overlap` characters after the previous one
3. The overlap ensures that sentences or concepts spanning a chunk boundary appear fully in at least one chunk
4. Chunking is character-based (not token-based) so it works correctly with multi-byte characters (Japanese, Chinese, Korean, etc.)

### Why these defaults

- **1000 characters** is roughly 150–250 English words or 300–500 Japanese characters. This is large enough to carry meaningful context for a single topic, while small enough that the embedding vector represents a focused concept rather than a mixture of topics.
- **200 characters of overlap** (~20% of chunk size) prevents information loss at boundaries. Sentence-level chunking was considered but adds complexity without significant quality improvement for general-purpose document search.

### Trade-offs

| Smaller chunks | Larger chunks |
|----------------|---------------|
| More precise retrieval | More context per result |
| More chunks to store/search | Fewer chunks to store/search |
| Embeddings represent narrower topics | Embeddings may blend multiple topics |

For domain-specific tuning, adjust `chunk_size` and `chunk_overlap` in `config.toml` and re-index with `ragctl reset && ragctl ingest`.

## Internal Candidate Count

Each search engine independently retrieves **30 candidates** per query. This is a hardcoded constant in `server/src/search/hybrid.rs` and `cli/src/main.rs`.

### Why 30?

- RRF quality depends on having enough candidates from each source. With too few candidates (e.g., 5), a relevant document that ranks #8 in one engine would be lost entirely.
- 30 candidates per engine means up to 60 unique chunks enter the fusion stage (fewer if there is overlap between the two result sets).
- Empirically, going beyond 30 per engine yields diminishing returns — the additional candidates rarely affect the final top 10–20.

### Why not expose it?

The internal candidate count is intentionally **not exposed as a parameter** to the MCP client. The reasoning:

1. **LLMs are not good at tuning retrieval parameters.** Claude may request `top_k=20` results, but cannot reliably reason about how many internal candidates are needed to produce good results.
2. **The two numbers serve different purposes.** Internal candidates determine retrieval recall (quality). `top_k` determines how many results the client wants to see. These should be independently controllable.
3. **A safe, fixed value is simpler.** 30 candidates × 2 engines is a reasonable default for most document collections.

If you need to adjust the internal candidate count (e.g., for very large collections), modify the `30` constants in `hybrid.rs`.

## Reciprocal Rank Fusion (RRF)

RRF is the algorithm that combines rankings from vector search and BM25 search into a single result list.

### Formula

For each document `d`, the fused score is:

```
score(d) = Σ  1 / (k + rank_i(d))
           i
```

Where:
- `k` = 60 (smoothing constant)
- `rank_i(d)` = the 1-based rank of document `d` in result list `i` (vector or BM25)
- The sum is over all result lists that contain `d`

### Why k=60?

The constant `k = 60` comes from the original RRF paper (Cormack et al., 2009). It controls how much the fusion favors top-ranked results:

- **Smaller k** (e.g., 1): Heavily favors the top result; lower-ranked results contribute almost nothing.
- **Larger k** (e.g., 1000): All ranks contribute nearly equally; fusion becomes rank-insensitive.
- **k=60**: A balanced default that gives meaningful weight to results in the top 30 while still distinguishing rank positions.

### Behavior

- A document found in **both** engines gets contributions from both rankings, boosting it above documents found by only one engine. This is the key advantage of hybrid search.
- A document found at rank #1 in both engines gets: `1/(60+1) + 1/(60+1) ≈ 0.0328`
- A document found at rank #1 in one engine only gets: `1/(60+1) ≈ 0.0164`
- Documents are sorted by fused score and the top `top_k` are returned.

## top_k Parameter

The `top_k` parameter controls how many final results are returned to the MCP client.

| Layer | Value | Configurable? |
|-------|-------|---------------|
| Internal candidates (per engine) | 30 | No (hardcoded) |
| Final results (`top_k`) | Default 10 | Yes (MCP tool parameter) |

### How top_k is determined

1. The MCP tool schema exposes `top_k` as an optional parameter with description `"Number of results to return (default: 10)"`
2. The MCP client (e.g., Claude) decides the value at call time based on the query context
3. If not specified, the server uses the default of 10

The client may choose values like 5 for specific lookups or 20–30 for exploratory queries. The internal candidate pool (30 per engine, up to 60 total) is always large enough to support these values.

## Embedding

| Provider | Model | Dimension | Notes |
|----------|-------|-----------|-------|
| `openai` | `text-embedding-3-small` | 1536 | Requires `OPENAI_API_KEY` |
| `local` | `intfloat/multilingual-e5-base` | 768 | `--features local-embed` |
| `local` | `intfloat/multilingual-e5-small` | 384 | `--features local-embed` |

Switching embedding provider changes the vector dimension. Run `ragctl reset && ragctl ingest` after switching.

## BM25 Tokenizer

The Tantivy BM25 search uses a configurable tokenizer:

| Value | Feature flag | Best for |
|-------|-------------|----------|
| `default` | *(none)* | English, space-separated languages |
| `japanese` | `--features ja` | Japanese text (IPADIC dictionary) |
| `korean` | `--features ko` | Korean text (ko-dic dictionary) |
| `chinese` | `--features zh` | Chinese text (CC-CEDICT dictionary) |

The tokenizer affects BM25 only. Vector search uses the embedding model's own tokenization.
