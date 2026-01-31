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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_short_text() {
        assert_eq!(truncate_snippet("hello", 10), "hello");
    }

    #[test]
    fn test_truncate_exact_length() {
        let text = "a".repeat(10);
        assert_eq!(truncate_snippet(&text, 10), text);
    }

    #[test]
    fn test_truncate_long_text() {
        let text = "a".repeat(20);
        let result = truncate_snippet(&text, 10);
        assert_eq!(result, format!("{}...", "a".repeat(10)));
    }

    #[test]
    fn test_truncate_japanese() {
        let text = "あいうえおかきくけこさしすせそ"; // 15 chars
        let result = truncate_snippet(&text, 5);
        assert_eq!(result, "あいうえお...");
    }

    #[test]
    fn test_truncate_mixed_multibyte() {
        let text = "Hello世界！こんにちは";
        let result = truncate_snippet(&text, 8);
        assert_eq!(result, "Hello世界！...");
    }

    #[test]
    fn test_truncate_empty() {
        assert_eq!(truncate_snippet("", 10), "");
    }

    #[test]
    fn test_search_filters_default() {
        let f = SearchFilters::default();
        assert!(f.source_type.is_none());
        assert!(f.path_prefix.is_none());
    }

    #[test]
    fn test_chunk_payload_serialization() {
        let payload = ChunkPayload {
            chunk_id: "test-id".to_string(),
            source_path: "/path/to/file.md".to_string(),
            source_type: "md".to_string(),
            title: "Test".to_string(),
            chunk_index: 0,
            text: "content".to_string(),
            updated_at: "2026-01-01T00:00:00Z".to_string(),
        };
        let json = serde_json::to_string(&payload).unwrap();
        let deserialized: ChunkPayload = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.chunk_id, "test-id");
        assert_eq!(deserialized.chunk_index, 0);
    }
}
