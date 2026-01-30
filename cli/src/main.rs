mod chunker;
mod embedding;
mod ingest;
mod qdrant_client;
mod tantivy_index;

use clap::{Parser, Subcommand};
use mcp_hybrid_search_common::config::AppConfig;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "ragctl", about = "CLI indexer for mcp-server-hybrid-search")]
struct Cli {
    /// Path to config.toml
    #[arg(long, global = true)]
    config: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Ingest documents from source directories
    Ingest {
        /// Source directories (can be specified multiple times)
        #[arg(long = "source", required = true)]
        sources: Vec<String>,

        /// Qdrant URL (overrides config)
        #[arg(long)]
        qdrant: Option<String>,

        /// Tantivy index directory (overrides config)
        #[arg(long)]
        index_dir: Option<String>,

        /// Chunk size in characters
        #[arg(long)]
        chunk_size: Option<usize>,

        /// Chunk overlap in characters
        #[arg(long)]
        chunk_overlap: Option<usize>,
    },
    /// Show index status
    Status {
        /// Qdrant URL (overrides config)
        #[arg(long)]
        qdrant: Option<String>,

        /// Tantivy index directory (overrides config)
        #[arg(long)]
        index_dir: Option<String>,
    },
    /// Search documents (debug/testing)
    Search {
        /// Search query
        #[arg(long)]
        query: String,

        /// Number of results
        #[arg(long, default_value = "10")]
        top_k: usize,

        /// Qdrant URL (overrides config)
        #[arg(long)]
        qdrant: Option<String>,

        /// Tantivy index directory (overrides config)
        #[arg(long)]
        index_dir: Option<String>,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();
    let mut config = AppConfig::load(cli.config.as_deref())?;

    match cli.command {
        Commands::Ingest {
            sources,
            qdrant,
            index_dir,
            chunk_size,
            chunk_overlap,
        } => {
            if let Some(url) = qdrant {
                config.qdrant_url = url;
            }
            if let Some(dir) = index_dir {
                config.tantivy_index_dir = dir;
            }
            if let Some(size) = chunk_size {
                config.chunk_size = size;
            }
            if let Some(overlap) = chunk_overlap {
                config.chunk_overlap = overlap;
            }
            ingest::run_ingest(&config, &sources).await?;
        }
        Commands::Status { qdrant, index_dir } => {
            if let Some(url) = qdrant {
                config.qdrant_url = url;
            }
            if let Some(dir) = index_dir {
                config.tantivy_index_dir = dir;
            }
            run_status(&config).await?;
        }
        Commands::Search {
            query,
            top_k,
            qdrant,
            index_dir,
        } => {
            if let Some(url) = qdrant {
                config.qdrant_url = url;
            }
            if let Some(dir) = index_dir {
                config.tantivy_index_dir = dir;
            }
            run_search(&config, &query, top_k).await?;
        }
    }

    Ok(())
}

async fn run_status(config: &AppConfig) -> anyhow::Result<()> {
    println!("=== Index Status ===");

    // Qdrant status
    match qdrant_client::get_collection_info(config).await {
        Ok(info) => {
            println!("Qdrant collection '{}': {} points", config.collection_name, info);
        }
        Err(e) => {
            println!("Qdrant: error - {}", e);
        }
    }

    // Tantivy status
    match tantivy_index::get_index_count(config) {
        Ok(count) => {
            println!("Tantivy index: {} documents", count);
        }
        Err(e) => {
            println!("Tantivy: error - {}", e);
        }
    }

    Ok(())
}

async fn run_search(config: &AppConfig, query: &str, top_k: usize) -> anyhow::Result<()> {
    use mcp_hybrid_search_common::types::SearchFilters;

    // Get embedding for query
    let query_embedding = embedding::get_embedding(config, query).await?;

    // Vector search
    let vector_results = qdrant_client::search(config, &query_embedding, 30, &SearchFilters::default()).await?;

    // BM25 search
    let bm25_results = tantivy_index::search(config, query, 30, &SearchFilters::default())?;

    // RRF fusion
    let merged = crate::ingest::rrf_merge(&vector_results, &bm25_results, top_k);

    println!("=== Search Results ({} hits) ===", merged.len());
    for (i, r) in merged.iter().enumerate() {
        println!(
            "\n[{}] score={:.4}  {}\n    {}\n    {}",
            i + 1,
            r.score,
            r.title,
            r.source_path,
            r.snippet
        );
    }

    Ok(())
}
