use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkPayload {
    pub chunk_id: String,
    pub source_path: String,
    pub source_type: String,
    pub title: String,
    pub chunk_index: u32,
    pub text: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub chunk_id: String,
    pub score: f64,
    pub title: String,
    pub source_path: String,
    pub source_type: String,
    pub snippet: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkDetail {
    pub chunk_id: String,
    pub text: String,
    pub metadata: ChunkMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkMetadata {
    pub title: String,
    pub source_path: String,
    pub source_type: String,
    pub chunk_index: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SearchFilters {
    pub source_type: Option<String>,
    pub path_prefix: Option<String>,
}
