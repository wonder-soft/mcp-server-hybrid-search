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
            let title = if trimmed.len() > 100 {
                format!("{}...", &trimmed[..100])
            } else {
                trimmed.to_string()
            };
            return title;
        }
    }
    file_name.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_short_text() {
        let chunks = chunk_text("hello world", 100, 20);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "hello world");
    }

    #[test]
    fn test_chunk_with_overlap() {
        let text = "a".repeat(300);
        let chunks = chunk_text(&text, 100, 20);
        assert!(chunks.len() >= 3);
    }

    #[test]
    fn test_extract_title_heading() {
        let text = "# My Title\n\nSome content";
        assert_eq!(extract_title(text, "file.md"), "My Title");
    }

    #[test]
    fn test_extract_title_first_line() {
        let text = "First line content\n\nMore content";
        assert_eq!(extract_title(text, "file.md"), "First line content");
    }
}
