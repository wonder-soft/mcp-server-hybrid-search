use std::collections::HashMap;

use anyhow::Result;
use mcp_hybrid_search_common::config::AppConfig;
use mcp_hybrid_search_common::types::{ChunkDetail, SearchFilters, SearchResult};

use super::embedding;
use super::qdrant_search;
use super::tantivy_search;

pub struct HybridSearcher {
    // Stateless for now; could cache index readers in the future
}

impl HybridSearcher {
    pub fn new(_config: &AppConfig) -> Result<Self> {
        Ok(Self {})
    }

    pub async fn search(
        &self,
        config: &AppConfig,
        query: &str,
        top_k: usize,
        filters: &SearchFilters,
    ) -> Result<Vec<SearchResult>> {
        // Get query embedding
        let query_embedding = embedding::get_embedding(config, query).await?;

        // Vector search (top 30)
        let vector_results =
            qdrant_search::search(config, &query_embedding, 30, filters).await?;

        // BM25 search (top 30)
        let bm25_results = tantivy_search::search(config, query, 30, filters)?;

        // RRF merge
        let merged = rrf_merge(&vector_results, &bm25_results, top_k);

        Ok(merged)
    }

    pub async fn get_chunk(
        &self,
        config: &AppConfig,
        chunk_id: &str,
    ) -> Result<Option<ChunkDetail>> {
        qdrant_search::get_chunk(config, chunk_id).await
    }
}

/// Reciprocal Rank Fusion
fn rrf_merge(
    vector_results: &[SearchResult],
    bm25_results: &[SearchResult],
    top_k: usize,
) -> Vec<SearchResult> {
    let k = 60.0;

    let mut scores: HashMap<String, f64> = HashMap::new();
    let mut result_map: HashMap<String, SearchResult> = HashMap::new();

    for (rank, result) in vector_results.iter().enumerate() {
        let rrf_score = 1.0 / (k + rank as f64 + 1.0);
        *scores.entry(result.chunk_id.clone()).or_insert(0.0) += rrf_score;
        result_map
            .entry(result.chunk_id.clone())
            .or_insert_with(|| result.clone());
    }

    for (rank, result) in bm25_results.iter().enumerate() {
        let rrf_score = 1.0 / (k + rank as f64 + 1.0);
        *scores.entry(result.chunk_id.clone()).or_insert(0.0) += rrf_score;
        result_map
            .entry(result.chunk_id.clone())
            .or_insert_with(|| result.clone());
    }

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
