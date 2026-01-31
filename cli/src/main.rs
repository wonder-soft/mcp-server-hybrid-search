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

    /// Project name for collection isolation
    #[arg(long, global = true)]
    project: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize the default source directory and data directories
    Init,

    /// Ingest documents from source directories
    Ingest {
        /// Source directories (can be specified multiple times).
        /// Defaults to ~/.local/share/mcp-hybrid-search if omitted.
        #[arg(long = "source")]
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
    /// Reset all indexes (Qdrant collection, Tantivy index, ingest state)
    Reset {
        /// Qdrant URL (overrides config)
        #[arg(long)]
        qdrant: Option<String>,

        /// Tantivy index directory (overrides config)
        #[arg(long)]
        index_dir: Option<String>,

        /// Skip confirmation prompt
        #[arg(long)]
        force: bool,
    },
    /// Export all indexed data (chunks + embeddings) to a JSON file
    Export {
        /// Output file path
        #[arg(long, short)]
        output: String,

        /// Qdrant URL (overrides config)
        #[arg(long)]
        qdrant: Option<String>,
    },
    /// Import data from an exported JSON file
    Import {
        /// Input file path
        #[arg(long, short)]
        input: String,

        /// Qdrant URL (overrides config)
        #[arg(long)]
        qdrant: Option<String>,

        /// Tantivy index directory (overrides config)
        #[arg(long)]
        index_dir: Option<String>,
    },
    /// List all projects (Qdrant collections)
    ListProjects {
        /// Qdrant URL (overrides config)
        #[arg(long)]
        qdrant: Option<String>,
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
    config = config.with_project(cli.project.as_deref());

    match cli.command {
        Commands::Init => {
            run_init(&config)?;
        }
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

            let sources = resolve_sources(sources);
            ingest::run_ingest(&config, &sources).await?;
        }
        Commands::Reset {
            qdrant,
            index_dir,
            force,
        } => {
            if let Some(url) = qdrant {
                config.qdrant_url = url;
            }
            if let Some(dir) = index_dir {
                config.tantivy_index_dir = dir;
            }
            run_reset(&config, force).await?;
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
        Commands::Export { output, qdrant } => {
            if let Some(url) = qdrant {
                config.qdrant_url = url;
            }
            run_export(&config, &output).await?;
        }
        Commands::Import {
            input,
            qdrant,
            index_dir,
        } => {
            if let Some(url) = qdrant {
                config.qdrant_url = url;
            }
            if let Some(dir) = index_dir {
                config.tantivy_index_dir = dir;
            }
            run_import(&config, &input).await?;
        }
        Commands::ListProjects { qdrant } => {
            if let Some(url) = qdrant {
                config.qdrant_url = url;
            }
            run_list_projects(&config).await?;
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

/// Reset all indexes and ingest state.
async fn run_reset(config: &AppConfig, force: bool) -> anyhow::Result<()> {
    if !force {
        println!("This will delete:");
        println!("  - Qdrant collection '{}'", config.collection_name);
        println!("  - Tantivy index at {}", config.tantivy_index_dir);
        println!("  - Ingest state file");
        println!();
        print!("Are you sure? [y/N] ");
        use std::io::Write;
        std::io::stdout().flush()?;
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        if !input.trim().eq_ignore_ascii_case("y") {
            println!("Cancelled.");
            return Ok(());
        }
    }

    // Delete Qdrant collection
    match qdrant_client::delete_collection(config).await {
        Ok(()) => println!("Deleted Qdrant collection '{}'", config.collection_name),
        Err(e) => println!("Qdrant: {}", e),
    }

    // Delete Tantivy index directory
    let tantivy_path = std::path::Path::new(&config.tantivy_index_dir);
    if tantivy_path.exists() {
        std::fs::remove_dir_all(tantivy_path)?;
        println!("Deleted Tantivy index at {}", config.tantivy_index_dir);
    } else {
        println!("Tantivy index not found (already clean)");
    }

    // Delete ingest state file
    let state_parent = tantivy_path
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."));
    let state_file = state_parent.join("ingest_state.json");
    if state_file.exists() {
        std::fs::remove_file(&state_file)?;
        println!("Deleted ingest state file");
    }

    println!("\nReset complete. Run `ragctl ingest` to re-index.");
    Ok(())
}

/// Resolve source directories. If none specified, use the default.
fn resolve_sources(sources: Vec<String>) -> Vec<String> {
    if !sources.is_empty() {
        return sources;
    }

    let default_dir = AppConfig::default_source_dir();
    let default_str = default_dir.to_string_lossy().to_string();

    if !default_dir.exists() {
        tracing::warn!("Default source directory does not exist: {}", default_str);
        tracing::info!("Run `ragctl init` to create it, or specify --source explicitly.");
    } else {
        tracing::info!("Using default source directory: {}", default_str);
    }

    vec![default_str]
}

/// Initialize directories.
fn run_init(config: &AppConfig) -> anyhow::Result<()> {
    // Create default source directory
    let source_dir = AppConfig::default_source_dir();
    if source_dir.exists() {
        println!("Source directory already exists: {}", source_dir.display());
    } else {
        std::fs::create_dir_all(&source_dir)?;
        println!("Created source directory: {}", source_dir.display());
    }

    // Create tantivy index directory
    let tantivy_dir = std::path::Path::new(&config.tantivy_index_dir);
    if tantivy_dir.exists() {
        println!(
            "Tantivy index directory already exists: {}",
            tantivy_dir.display()
        );
    } else {
        std::fs::create_dir_all(tantivy_dir)?;
        println!("Created Tantivy index directory: {}", tantivy_dir.display());
    }

    println!();
    println!("Setup complete. Place documents (md, txt, pdf, xlsx, docx, ...) in:");
    println!("  {}", source_dir.display());
    println!();
    println!("Then run:");
    println!("  ragctl ingest");

    Ok(())
}

async fn run_status(config: &AppConfig) -> anyhow::Result<()> {
    println!("=== Index Status ===");
    println!(
        "Source directory: {}",
        AppConfig::default_source_dir().display()
    );

    // Qdrant status
    match qdrant_client::get_collection_info(config).await {
        Ok(info) => {
            println!(
                "Qdrant collection '{}': {} points",
                config.collection_name, info
            );
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

async fn run_export(config: &AppConfig, output_path: &str) -> anyhow::Result<()> {
    println!("Exporting data from Qdrant...");

    let chunks = qdrant_client::export_all_chunks(config).await?;

    if chunks.is_empty() {
        println!("No data to export. Is the collection empty?");
        return Ok(());
    }

    let json = serde_json::to_string_pretty(&chunks)?;
    std::fs::write(output_path, json)?;

    println!("Exported {} chunks to {}", chunks.len(), output_path);
    Ok(())
}

async fn run_import(config: &AppConfig, input_path: &str) -> anyhow::Result<()> {
    use mcp_hybrid_search_common::types::ExportedChunk;

    let content = std::fs::read_to_string(input_path)?;
    let chunks: Vec<ExportedChunk> = serde_json::from_str(&content)?;

    if chunks.is_empty() {
        println!("No data in the export file.");
        return Ok(());
    }

    println!("Importing {} chunks from {}...", chunks.len(), input_path);

    // Ensure Qdrant collection exists
    qdrant_client::ensure_collection(config).await?;

    // Import in batches
    let batch_size = 10;
    let mut total_imported = 0;

    for batch in chunks.chunks(batch_size) {
        let payloads: Vec<_> = batch.iter().map(|c| c.payload.clone()).collect();
        let embeddings: Vec<_> = batch.iter().map(|c| c.embedding.clone()).collect();

        // Upsert to Qdrant
        qdrant_client::upsert_chunks(config, &payloads, &embeddings).await?;

        // Index in Tantivy
        tantivy_index::index_chunks(config, &payloads)?;

        total_imported += batch.len();
        tracing::info!("Imported {}/{} chunks", total_imported, chunks.len());
    }

    println!(
        "Import complete: {} chunks imported to Qdrant and Tantivy",
        total_imported
    );
    Ok(())
}

async fn run_list_projects(config: &AppConfig) -> anyhow::Result<()> {
    let collections = qdrant_client::list_collections(config).await?;

    if collections.is_empty() {
        println!("No collections found.");
        return Ok(());
    }

    println!("=== Collections (Projects) ===");
    for (name, count) in &collections {
        println!("  {} — {} points", name, count);
    }

    Ok(())
}

async fn run_search(config: &AppConfig, query: &str, top_k: usize) -> anyhow::Result<()> {
    use mcp_hybrid_search_common::types::SearchFilters;

    // Get embedding for query
    let query_embedding = embedding::get_embedding(config, query).await?;

    // Vector search
    let vector_results =
        qdrant_client::search(config, &query_embedding, 30, &SearchFilters::default()).await?;

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
