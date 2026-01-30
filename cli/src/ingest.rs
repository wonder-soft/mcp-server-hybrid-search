use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use mcp_hybrid_search_common::config::AppConfig;
use mcp_hybrid_search_common::types::{ChunkPayload, SearchResult};
use walkdir::WalkDir;

use crate::chunker;
use crate::embedding;
use crate::qdrant_client;
use crate::tantivy_index;

const SUPPORTED_EXTENSIONS: &[&str] = &["md", "txt"];

/// Run the ingest pipeline for the given source directories.
pub async fn run_ingest(config: &AppConfig, sources: &[String]) -> Result<()> {
    // Ensure Qdrant collection exists
    qdrant_client::ensure_collection(config).await?;

    // Collect files
    let files = collect_files(sources);
    tracing::info!("Found {} files to process", files.len());

    if files.is_empty() {
        tracing::warn!("No files found in the specified source directories");
        return Ok(());
    }

    let mut total_chunks = 0;
    let mut total_errors = 0;

    // Process files in batches
    let batch_size = 10;
    for batch in files.chunks(batch_size) {
        let mut all_chunks = Vec::new();

        for file_path in batch {
            match process_file(config, file_path) {
                Ok(chunks) => {
                    all_chunks.extend(chunks);
                }
                Err(e) => {
                    tracing::error!("Error processing {}: {}", file_path, e);
                    total_errors += 1;
                }
            }
        }

        if all_chunks.is_empty() {
            continue;
        }

        // Get embeddings for all chunks in this batch
        let texts: Vec<String> = all_chunks.iter().map(|c| c.text.clone()).collect();

        // Embed in sub-batches of 20 (OpenAI limit considerations)
        let embed_batch_size = 20;
        let mut all_embeddings = Vec::new();

        for text_batch in texts.chunks(embed_batch_size) {
            match embedding::get_embeddings(config, text_batch).await {
                Ok(embeddings) => {
                    all_embeddings.extend(embeddings);
                }
                Err(e) => {
                    tracing::error!("Embedding error: {}", e);
                    // Skip this entire sub-batch
                    for _ in 0..text_batch.len() {
                        all_embeddings.push(vec![0.0; config.embedding_dimension]);
                    }
                    total_errors += 1;
                }
            }
        }

        // Upsert to Qdrant
        if let Err(e) = qdrant_client::upsert_chunks(config, &all_chunks, &all_embeddings).await {
            tracing::error!("Qdrant upsert error: {}", e);
            total_errors += 1;
        }

        // Index in Tantivy
        if let Err(e) = tantivy_index::index_chunks(config, &all_chunks) {
            tracing::error!("Tantivy index error: {}", e);
            total_errors += 1;
        }

        total_chunks += all_chunks.len();
        tracing::info!(
            "Processed batch: {} chunks (total: {})",
            all_chunks.len(),
            total_chunks
        );
    }

    tracing::info!(
        "Ingest complete: {} chunks indexed, {} errors",
        total_chunks,
        total_errors
    );
    Ok(())
}

/// Collect all supported files from source directories.
fn collect_files(sources: &[String]) -> Vec<String> {
    let mut files = Vec::new();

    for source in sources {
        let path = Path::new(source);
        if !path.exists() {
            tracing::warn!("Source path does not exist: {}", source);
            continue;
        }

        for entry in WalkDir::new(path).follow_links(true) {
            match entry {
                Ok(entry) => {
                    if entry.file_type().is_file() {
                        if let Some(ext) = entry.path().extension() {
                            let ext_str = ext.to_string_lossy().to_lowercase();
                            if SUPPORTED_EXTENSIONS.contains(&ext_str.as_str()) {
                                files.push(entry.path().to_string_lossy().to_string());
                            }
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("Error walking directory: {}", e);
                }
            }
        }
    }

    files
}

/// Process a single file into chunks.
fn process_file(config: &AppConfig, file_path: &str) -> Result<Vec<ChunkPayload>> {
    let content = std::fs::read_to_string(file_path)?;
    let path = Path::new(file_path);

    let file_name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();

    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().to_string())
        .unwrap_or_else(|| "txt".to_string());

    let title = chunker::extract_title(&content, &file_name);
    let chunks = chunker::chunk_text(&content, config.chunk_size, config.chunk_overlap);

    let now = chrono::Utc::now().to_rfc3339();

    let payloads: Vec<ChunkPayload> = chunks
        .iter()
        .enumerate()
        .map(|(i, chunk_text)| {
            let chunk_id = uuid::Uuid::new_v4().to_string();
            ChunkPayload {
                chunk_id,
                source_path: file_path.to_string(),
                source_type: ext.clone(),
                title: title.clone(),
                chunk_index: i as u32,
                text: chunk_text.clone(),
                updated_at: now.clone(),
            }
        })
        .collect();

    Ok(payloads)
}

/// Reciprocal Rank Fusion (RRF) merge of vector and BM25 results.
pub fn rrf_merge(
    vector_results: &[SearchResult],
    bm25_results: &[SearchResult],
    top_k: usize,
) -> Vec<SearchResult> {
    let k = 60.0; // RRF constant

    let mut scores: HashMap<String, f64> = HashMap::new();
    let mut result_map: HashMap<String, SearchResult> = HashMap::new();

    // Score vector results
    for (rank, result) in vector_results.iter().enumerate() {
        let rrf_score = 1.0 / (k + rank as f64 + 1.0);
        *scores.entry(result.chunk_id.clone()).or_insert(0.0) += rrf_score;
        result_map
            .entry(result.chunk_id.clone())
            .or_insert_with(|| result.clone());
    }

    // Score BM25 results
    for (rank, result) in bm25_results.iter().enumerate() {
        let rrf_score = 1.0 / (k + rank as f64 + 1.0);
        *scores.entry(result.chunk_id.clone()).or_insert(0.0) += rrf_score;
        result_map
            .entry(result.chunk_id.clone())
            .or_insert_with(|| result.clone());
    }

    // Sort by fused score
    let mut scored: Vec<(String, f64)> = scores.into_iter().collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    scored
        .into_iter()
        .take(top_k)
        .filter_map(|(id, score)| {
            result_map.remove(&id).map(|mut r| {
                r.score = score;
                r
            })
        })
        .collect()
}
