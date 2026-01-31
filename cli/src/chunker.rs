/// Split text into chunks with overlap.
pub fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    if text.is_empty() {
        return vec![];
    }

    let chars: Vec<char> = text.chars().collect();
    let total = chars.len();

    if total <= chunk_size {
        return vec![text.to_string()];
    }

    let mut chunks = Vec::new();
    let mut start = 0;

    while start < total {
        let end = (start + chunk_size).min(total);
        let chunk: String = chars[start..end].iter().collect();
        chunks.push(chunk.trim().to_string());

        if end >= total {
            break;
        }

        // Move forward by (chunk_size - overlap)
        let step = if chunk_size > overlap {
            chunk_size - overlap
        } else {
            chunk_size
        };
        start += step;
    }

    // Remove empty chunks
    chunks.retain(|c| !c.is_empty());
    chunks
}

/// Extract a title from a markdown/text document.
/// Uses the first heading or first non-empty line.
pub fn extract_title(text: &str, file_name: &str) -> String {
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('#') {
            return trimmed.trim_start_matches('#').trim().to_string();
        }
        if !trimmed.is_empty() {
            return mcp_hybrid_search_common::types::truncate_snippet(trimmed, 100);
        }
    }
    file_name.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- chunk_text ---

    #[test]
    fn test_chunk_empty_text() {
        let chunks = chunk_text("", 100, 20);
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_chunk_short_text() {
        let chunks = chunk_text("hello world", 100, 20);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "hello world");
    }

    #[test]
    fn test_chunk_exact_size() {
        let text = "a".repeat(100);
        let chunks = chunk_text(&text, 100, 20);
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn test_chunk_with_overlap() {
        let text = "a".repeat(300);
        let chunks = chunk_text(&text, 100, 20);
        assert!(chunks.len() >= 3);
        // Each chunk should be at most chunk_size characters
        for chunk in &chunks {
            assert!(chunk.chars().count() <= 100);
        }
    }

    #[test]
    fn test_chunk_overlap_larger_than_size() {
        // overlap >= chunk_size should not cause infinite loop
        let text = "a".repeat(300);
        let chunks = chunk_text(&text, 100, 150);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_chunk_japanese_text() {
        let text = "あ".repeat(300);
        let chunks = chunk_text(&text, 100, 20);
        assert!(chunks.len() >= 3);
        for chunk in &chunks {
            assert!(chunk.chars().count() <= 100);
        }
    }

    // --- extract_title ---

    #[test]
    fn test_extract_title_heading() {
        let text = "# My Title\n\nSome content";
        assert_eq!(extract_title(text, "file.md"), "My Title");
    }

    #[test]
    fn test_extract_title_h2() {
        let text = "## Section Title\n\nContent";
        assert_eq!(extract_title(text, "file.md"), "Section Title");
    }

    #[test]
    fn test_extract_title_first_line() {
        let text = "First line content\n\nMore content";
        assert_eq!(extract_title(text, "file.md"), "First line content");
    }

    #[test]
    fn test_extract_title_empty_lines_before_content() {
        let text = "\n\n\nActual content";
        assert_eq!(extract_title(text, "file.md"), "Actual content");
    }

    #[test]
    fn test_extract_title_empty_text() {
        assert_eq!(extract_title("", "fallback.md"), "fallback.md");
    }

    #[test]
    fn test_extract_title_long_line_truncated() {
        let long_line = "あ".repeat(200);
        let title = extract_title(&long_line, "file.md");
        assert!(title.chars().count() <= 104); // 100 chars + "..."
    }
}
