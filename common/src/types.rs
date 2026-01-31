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

/// Truncate a string to at most `max_chars` characters (UTF-8 safe).
pub fn truncate_snippet(text: &str, max_chars: usize) -> String {
    let char_count = text.chars().count();
    if char_count <= max_chars {
        text.to_string()
    } else {
        let truncated: String = text.chars().take(max_chars).collect();
        format!("{}...", truncated)
    }
}
