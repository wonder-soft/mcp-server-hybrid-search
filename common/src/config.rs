use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    #[serde(default = "default_qdrant_url")]
    pub qdrant_url: String,

    #[serde(default = "default_collection_name")]
    pub collection_name: String,

    #[serde(default = "default_tantivy_index_dir")]
    pub tantivy_index_dir: String,

    #[serde(default = "default_chunk_size")]
    pub chunk_size: usize,

    #[serde(default = "default_chunk_overlap")]
    pub chunk_overlap: usize,

    #[serde(default = "default_listen_port")]
    pub listen_port: u16,

    #[serde(default = "default_embedding_provider")]
    pub embedding_provider: String,

    #[serde(default = "default_embedding_model")]
    pub embedding_model: String,

    #[serde(default = "default_embedding_dimension")]
    pub embedding_dimension: usize,

    #[serde(default = "default_tokenizer")]
    pub tokenizer: String,
}

fn default_qdrant_url() -> String {
    "http://localhost:6334".to_string()
}

fn default_collection_name() -> String {
    "docs".to_string()
}

fn default_tantivy_index_dir() -> String {
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    home.join(".mcp-hybrid-search")
        .join("tantivy")
        .to_string_lossy()
        .to_string()
}

fn default_chunk_size() -> usize {
    1000
}

fn default_chunk_overlap() -> usize {
    200
}

fn default_listen_port() -> u16 {
    7070
}

fn default_embedding_provider() -> String {
    "openai".to_string()
}

fn default_embedding_model() -> String {
    "text-embedding-3-small".to_string()
}

fn default_embedding_dimension() -> usize {
    1536
}

fn default_tokenizer() -> String {
    "default".to_string()
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            qdrant_url: default_qdrant_url(),
            collection_name: default_collection_name(),
            tantivy_index_dir: default_tantivy_index_dir(),
            chunk_size: default_chunk_size(),
            chunk_overlap: default_chunk_overlap(),
            listen_port: default_listen_port(),
            embedding_provider: default_embedding_provider(),
            embedding_model: default_embedding_model(),
            embedding_dimension: default_embedding_dimension(),
            tokenizer: default_tokenizer(),
        }
    }
}

impl AppConfig {
    /// Default source directory for documents (~/.local/share/mcp-hybrid-search).
    pub fn default_source_dir() -> PathBuf {
        let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        home.join(".local").join("share").join("mcp-hybrid-search")
    }

    pub fn load(path: Option<&str>) -> anyhow::Result<Self> {
        let config_path = if let Some(p) = path {
            PathBuf::from(p)
        } else {
            // Try current directory, then home directory
            let cwd = std::env::current_dir()?;
            let cwd_config = cwd.join("config.toml");
            if cwd_config.exists() {
                cwd_config
            } else {
                let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
                home.join(".mcp-hybrid-search").join("config.toml")
            }
        };

        if config_path.exists() {
            let content = std::fs::read_to_string(&config_path)?;
            let config: AppConfig = toml::from_str(&content)?;
            Ok(config)
        } else {
            Ok(AppConfig::default())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = AppConfig::default();
        assert_eq!(config.qdrant_url, "http://localhost:6334");
        assert_eq!(config.collection_name, "docs");
        assert_eq!(config.chunk_size, 1000);
        assert_eq!(config.chunk_overlap, 200);
        assert_eq!(config.listen_port, 7070);
        assert_eq!(config.embedding_model, "text-embedding-3-small");
        assert_eq!(config.embedding_dimension, 1536);
    }

    #[test]
    fn test_load_nonexistent_returns_default() {
        let config = AppConfig::load(Some("/nonexistent/path/config.toml")).unwrap();
        assert_eq!(config.listen_port, 7070);
    }

    #[test]
    fn test_parse_partial_toml() {
        let toml_str = r#"
            listen_port = 8080
            chunk_size = 500
        "#;
        let config: AppConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.listen_port, 8080);
        assert_eq!(config.chunk_size, 500);
        // Defaults for unspecified fields
        assert_eq!(config.collection_name, "docs");
        assert_eq!(config.embedding_dimension, 1536);
    }

    #[test]
    fn test_default_source_dir() {
        let dir = AppConfig::default_source_dir();
        let dir_str = dir.to_string_lossy();
        assert!(dir_str.contains("mcp-hybrid-search"));
        assert!(dir_str.contains(".local/share"));
    }
}
