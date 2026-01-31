use std::collections::HashMap;
use std::path::Path;
use std::process::Command;

use anyhow::Result;
use mcp_hybrid_search_common::config::AppConfig;
use mcp_hybrid_search_common::types::{ChunkPayload, SearchResult};
use walkdir::WalkDir;

use crate::chunker;
use crate::embedding;
use crate::qdrant_client;
use crate::tantivy_index;

/// Text files that can be read directly.
const TEXT_EXTENSIONS: &[&str] = &["md", "txt"];

/// Binary/rich files that require markitdown conversion.
const MARKITDOWN_EXTENSIONS: &[&str] = &["pdf", "xlsx", "xls", "docx", "pptx", "csv", "html"];

/// Ingest state file: tracks which files have been ingested and when.
fn state_file_path(config: &AppConfig) -> std::path::PathBuf {
    let tantivy_parent = Path::new(&config.tantivy_index_dir)
        .parent()
        .unwrap_or_else(|| Path::new("."));
    tantivy_parent.join("ingest_state.json")
}

/// State: maps file path -> last modified timestamp (as string).
type IngestState = HashMap<String, String>;

fn load_state(config: &AppConfig) -> IngestState {
    let path = state_file_path(config);
    if path.exists() {
        match std::fs::read_to_string(&path) {
            Ok(content) => serde_json::from_str(&content).unwrap_or_default(),
            Err(_) => HashMap::new(),
        }
    } else {
        HashMap::new()
    }
}

fn save_state(config: &AppConfig, state: &IngestState) -> Result<()> {
    let path = state_file_path(config);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(state)?;
    std::fs::write(path, json)?;
    Ok(())
}

/// Get the modified time of a file as an RFC3339 string.
fn file_modified_time(path: &str) -> Option<String> {
    std::fs::metadata(path)
        .ok()
        .and_then(|m| m.modified().ok())
        .map(|t| {
            let datetime: chrono::DateTime<chrono::Utc> = t.into();
            datetime.to_rfc3339()
        })
}

/// Run the ingest pipeline for the given source directories.
pub async fn run_ingest(config: &AppConfig, sources: &[String]) -> Result<()> {
    // Ensure Qdrant collection exists
    qdrant_client::ensure_collection(config).await?;

    // Check markitdown availability
    let markitdown_available = check_markitdown();
    if !markitdown_available {
        tracing::warn!(
            "markitdown not found in PATH. PDF/Excel/Word files will be skipped. \
             Install with: pip install markitdown"
        );
    }

    // Load previous ingest state for diff detection
    let mut state = load_state(config);

    // Collect files
    let files = collect_files(sources, markitdown_available);
    tracing::info!("Found {} candidate files", files.len());

    if files.is_empty() {
        tracing::warn!("No files found in the specified source directories");
        return Ok(());
    }

    // Filter to only changed/new files
    let total_candidates = files.len();
    let files_to_process: Vec<String> = files
        .into_iter()
        .filter(|f| {
            let current_mtime = file_modified_time(f).unwrap_or_default();
            match state.get(f) {
                Some(prev_mtime) if prev_mtime == &current_mtime => {
                    tracing::debug!("Skipping unchanged file: {}", f);
                    false
                }
                _ => true,
            }
        })
        .collect();

    let skipped = total_candidates - files_to_process.len();
    tracing::info!(
        "{} files need processing ({} unchanged, skipped)",
        files_to_process.len(),
        skipped
    );

    if files_to_process.is_empty() {
        tracing::info!("All files are up to date. Nothing to ingest.");
        return Ok(());
    }

    let mut total_chunks = 0;
    let mut total_errors = 0;
    let mut processed_files: Vec<String> = Vec::new();

    // Process files in batches
    let batch_size = 10;
    for batch in files_to_process.chunks(batch_size) {
        let mut all_chunks = Vec::new();

        for file_path in batch {
            match process_file(config, file_path) {
                Ok(chunks) => {
                    all_chunks.extend(chunks);
                    processed_files.push(file_path.clone());
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

        // Get embeddings for all chunks in this batch.
        // We embed in sub-batches and collect only successfully embedded chunks.
        let embed_batch_size = 20;
        let mut embedded_chunks: Vec<ChunkPayload> = Vec::new();
        let mut all_embeddings: Vec<Vec<f32>> = Vec::new();

        for (sub_chunks, text_batch) in chunk_sub_batches(&all_chunks, embed_batch_size) {
            match embedding::get_embeddings(config, &text_batch).await {
                Ok(embeddings) => {
                    embedded_chunks.extend(sub_chunks.to_vec());
                    all_embeddings.extend(embeddings);
                }
                Err(e) => {
                    tracing::error!(
                        "Embedding error (skipping {} chunks): {}",
                        sub_chunks.len(),
                        e
                    );
                    total_errors += 1;
                }
            }
        }

        if embedded_chunks.is_empty() {
            tracing::warn!("No chunks were successfully embedded in this batch");
            continue;
        }

        // Upsert to Qdrant
        if let Err(e) =
            qdrant_client::upsert_chunks(config, &embedded_chunks, &all_embeddings).await
        {
            tracing::error!("Qdrant upsert error: {}", e);
            total_errors += 1;
        }

        // Index in Tantivy (all chunks, not just embedded — BM25 doesn't need vectors)
        if let Err(e) = tantivy_index::index_chunks(config, &all_chunks) {
            tracing::error!("Tantivy index error: {}", e);
            total_errors += 1;
        }

        total_chunks += embedded_chunks.len();
        tracing::info!(
            "Processed batch: {} chunks (total: {})",
            all_chunks.len(),
            total_chunks
        );
    }

    // Update state for successfully processed files
    for file_path in &processed_files {
        if let Some(mtime) = file_modified_time(file_path) {
            state.insert(file_path.clone(), mtime);
        }
    }
    save_state(config, &state)?;

    tracing::info!(
        "Ingest complete: {} files processed, {} chunks indexed, {} errors",
        processed_files.len(),
        total_chunks,
        total_errors
    );
    Ok(())
}

/// Split chunks into sub-batches for embedding, returning (chunk_slice, texts) pairs.
fn chunk_sub_batches(
    chunks: &[ChunkPayload],
    batch_size: usize,
) -> Vec<(&[ChunkPayload], Vec<String>)> {
    chunks
        .chunks(batch_size)
        .map(|sub| {
            let texts: Vec<String> = sub.iter().map(|c| c.text.clone()).collect();
            (sub, texts)
        })
        .collect()
}

/// Check if markitdown CLI is available.
fn check_markitdown() -> bool {
    Command::new("markitdown").arg("--help").output().is_ok()
}

/// Convert a file to markdown text using markitdown CLI.
fn convert_with_markitdown(file_path: &str) -> Result<String> {
    tracing::info!("Converting with markitdown: {}", file_path);
    let output = Command::new("markitdown")
        .arg(file_path)
        .output()
        .map_err(|e| anyhow::anyhow!("Failed to run markitdown: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("markitdown failed for {}: {}", file_path, stderr);
    }

    let text = String::from_utf8_lossy(&output.stdout).to_string();
    if text.trim().is_empty() {
        anyhow::bail!("markitdown returned empty output for {}", file_path);
    }

    Ok(text)
}

/// Collect all supported files from source directories.
fn collect_files(sources: &[String], markitdown_available: bool) -> Vec<String> {
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
                            if TEXT_EXTENSIONS.contains(&ext_str.as_str()) {
                                files.push(entry.path().to_string_lossy().to_string());
                            } else if MARKITDOWN_EXTENSIONS.contains(&ext_str.as_str()) {
                                if markitdown_available {
                                    files.push(entry.path().to_string_lossy().to_string());
                                } else {
                                    tracing::warn!(
                                        "Skipping {} (markitdown not available)",
                                        entry.path().display()
                                    );
                                }
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
    let path = Path::new(file_path);
    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_else(|| "txt".to_string());

    // Read or convert file content
    let content = if MARKITDOWN_EXTENSIONS.contains(&ext.as_str()) {
        convert_with_markitdown(file_path)?
    } else {
        std::fs::read_to_string(file_path)?
    };

    let file_name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();

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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_result(id: &str) -> SearchResult {
        SearchResult {
            chunk_id: id.to_string(),
            score: 0.0,
            title: format!("Title {}", id),
            source_path: "/test".to_string(),
            source_type: "md".to_string(),
            snippet: "snippet".to_string(),
        }
    }

    #[test]
    fn test_rrf_merge_empty() {
        let result = rrf_merge(&[], &[], 10);
        assert!(result.is_empty());
    }

    #[test]
    fn test_rrf_merge_single_source() {
        let vec_results = vec![make_result("a"), make_result("b")];
        let result = rrf_merge(&vec_results, &[], 10);
        assert_eq!(result.len(), 2);
        // First rank should have higher score
        assert!(result[0].score > result[1].score);
    }

    #[test]
    fn test_rrf_merge_overlap_boosted() {
        // "a" appears in both -> should be ranked highest
        let vec_results = vec![make_result("a"), make_result("b")];
        let bm25_results = vec![make_result("a"), make_result("c")];
        let result = rrf_merge(&vec_results, &bm25_results, 10);
        assert_eq!(result[0].chunk_id, "a");
        // "a" should have roughly double the score of "b" or "c"
        assert!(result[0].score > result[1].score);
    }

    #[test]
    fn test_rrf_merge_top_k_limit() {
        let vec_results: Vec<SearchResult> =
            (0..20).map(|i| make_result(&format!("v{}", i))).collect();
        let bm25_results: Vec<SearchResult> =
            (0..20).map(|i| make_result(&format!("b{}", i))).collect();
        let result = rrf_merge(&vec_results, &bm25_results, 5);
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn test_rrf_merge_scores_are_positive() {
        let vec_results = vec![make_result("a")];
        let bm25_results = vec![make_result("b")];
        let result = rrf_merge(&vec_results, &bm25_results, 10);
        for r in &result {
            assert!(r.score > 0.0);
        }
    }
}
