use anyhow::Result;
use mcp_hybrid_search_common::config::AppConfig;
use mcp_hybrid_search_common::types::{ChunkPayload, SearchFilters, SearchResult};
use std::path::Path;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::*;
use tantivy::{doc, Index, IndexWriter, ReloadPolicy};

/// Name used for the custom tokenizer when configured.
const CUSTOM_TOKENIZER_NAME: &str = "custom_tokenizer";

/// Build the Tantivy schema.
/// When a non-default tokenizer is configured, text fields use it.
fn build_schema(tokenizer_name: &str) -> Schema {
    let mut schema_builder = Schema::builder();
    schema_builder.add_text_field("chunk_id", STRING | STORED);
    schema_builder.add_text_field("source_path", STRING | STORED);

    let text_options = TextOptions::default()
        .set_indexing_options(
            TextFieldIndexing::default()
                .set_tokenizer(tokenizer_name)
                .set_index_option(IndexRecordOption::WithFreqsAndPositions),
        )
        .set_stored();

    schema_builder.add_text_field("title", text_options.clone());
    schema_builder.add_text_field("body", text_options);
    schema_builder.add_text_field("source_type", STRING | STORED);
    schema_builder.build()
}

/// Resolve the tokenizer name to use based on config.
fn resolve_tokenizer_name(config: &AppConfig) -> &str {
    match config.tokenizer.as_str() {
        "default" | "" => "default",
        _ => CUSTOM_TOKENIZER_NAME,
    }
}

/// Register the appropriate tokenizer on the index based on config.
fn register_tokenizer(index: &Index, config: &AppConfig) -> Result<()> {
    match config.tokenizer.as_str() {
        "default" | "" => {
            // Use tantivy's built-in default tokenizer; nothing to register.
            Ok(())
        }
        "japanese" | "korean" | "chinese" => register_lindera_tokenizer(index, &config.tokenizer),
        other => {
            anyhow::bail!(
                "Unknown tokenizer '{}'. Supported values: default, japanese, korean, chinese",
                other
            )
        }
    }
}

#[cfg(any(feature = "ja", feature = "ko", feature = "zh"))]
fn register_lindera_tokenizer(index: &Index, lang: &str) -> Result<()> {
    use lindera::mode::Mode;
    use lindera::segmenter::Segmenter;
    use lindera_tantivy::tokenizer::LinderaTokenizer;

    let dict_uri = match lang {
        "japanese" => {
            #[cfg(not(feature = "ja"))]
            anyhow::bail!(
                "tokenizer = \"japanese\" requires the 'ja' feature. \
                 Build with: cargo build --features ja"
            );
            #[cfg(feature = "ja")]
            "embedded://ipadic"
        }
        "korean" => {
            #[cfg(not(feature = "ko"))]
            anyhow::bail!(
                "tokenizer = \"korean\" requires the 'ko' feature. \
                 Build with: cargo build --features ko"
            );
            #[cfg(feature = "ko")]
            "embedded://ko-dic"
        }
        "chinese" => {
            #[cfg(not(feature = "zh"))]
            anyhow::bail!(
                "tokenizer = \"chinese\" requires the 'zh' feature. \
                 Build with: cargo build --features zh"
            );
            #[cfg(feature = "zh")]
            "embedded://cc-cedict"
        }
        _ => anyhow::bail!("Unsupported language for lindera: {}", lang),
    };

    let dictionary =
        lindera::dictionary::load_dictionary(dict_uri).map_err(|e| anyhow::anyhow!("{}", e))?;
    let segmenter = Segmenter::new(Mode::Normal, dictionary, None);
    let tokenizer = LinderaTokenizer::from_segmenter(segmenter);
    index
        .tokenizers()
        .register(CUSTOM_TOKENIZER_NAME, tokenizer);
    tracing::info!("Registered lindera tokenizer for '{}'", lang);
    Ok(())
}

#[cfg(not(any(feature = "ja", feature = "ko", feature = "zh")))]
fn register_lindera_tokenizer(_index: &Index, lang: &str) -> Result<()> {
    anyhow::bail!(
        "tokenizer = \"{}\" requires a language feature to be enabled at build time. \
         Available features: ja, ko, zh. \
         Example: cargo build --features ja",
        lang
    )
}

/// Open or create the Tantivy index.
fn open_or_create_index(config: &AppConfig) -> Result<Index> {
    let tokenizer_name = resolve_tokenizer_name(config);
    let schema = build_schema(tokenizer_name);
    let path = Path::new(&config.tantivy_index_dir);

    if path.exists() {
        match Index::open_in_dir(path) {
            Ok(index) => {
                register_tokenizer(&index, config)?;
                return Ok(index);
            }
            Err(e) => {
                tracing::warn!("Failed to open existing index, recreating: {}", e);
            }
        }
    }

    std::fs::create_dir_all(path)?;
    let index = Index::create_in_dir(path, schema)?;
    register_tokenizer(&index, config)?;
    Ok(index)
}

/// Index chunks into Tantivy.
pub fn index_chunks(config: &AppConfig, chunks: &[ChunkPayload]) -> Result<()> {
    let index = open_or_create_index(config)?;
    let schema = index.schema();

    let chunk_id_field = schema.get_field("chunk_id").unwrap();
    let source_path_field = schema.get_field("source_path").unwrap();
    let title_field = schema.get_field("title").unwrap();
    let body_field = schema.get_field("body").unwrap();
    let source_type_field = schema.get_field("source_type").unwrap();

    let mut writer: IndexWriter = index.writer(50_000_000)?;

    for chunk in chunks {
        // Delete existing document with same chunk_id
        let term = tantivy::Term::from_field_text(chunk_id_field, &chunk.chunk_id);
        writer.delete_term(term);

        writer.add_document(doc!(
            chunk_id_field => chunk.chunk_id.clone(),
            source_path_field => chunk.source_path.clone(),
            title_field => chunk.title.clone(),
            body_field => chunk.text.clone(),
            source_type_field => chunk.source_type.clone(),
        ))?;
    }

    writer.commit()?;
    Ok(())
}

/// Search Tantivy index with BM25.
pub fn search(
    config: &AppConfig,
    query_str: &str,
    top_k: usize,
    filters: &SearchFilters,
) -> Result<Vec<SearchResult>> {
    let index = open_or_create_index(config)?;
    let schema = index.schema();

    let chunk_id_field = schema.get_field("chunk_id").unwrap();
    let source_path_field = schema.get_field("source_path").unwrap();
    let title_field = schema.get_field("title").unwrap();
    let body_field = schema.get_field("body").unwrap();
    let source_type_field = schema.get_field("source_type").unwrap();

    let reader = index
        .reader_builder()
        .reload_policy(ReloadPolicy::OnCommitWithDelay)
        .try_into()?;
    let searcher = reader.searcher();

    let query_parser = QueryParser::for_index(&index, vec![title_field, body_field]);
    let query = query_parser.parse_query(query_str)?;

    let top_docs = searcher.search(&query, &TopDocs::with_limit(top_k))?;

    let mut results = Vec::new();

    for (score, doc_address) in top_docs {
        let retrieved_doc: tantivy::TantivyDocument = searcher.doc(doc_address)?;

        let chunk_id = get_field_text(&retrieved_doc, chunk_id_field);
        let source_path = get_field_text(&retrieved_doc, source_path_field);
        let title = get_field_text(&retrieved_doc, title_field);
        let body = get_field_text(&retrieved_doc, body_field);
        let source_type = get_field_text(&retrieved_doc, source_type_field);

        // Apply filters
        if let Some(ref ft) = filters.source_type {
            if &source_type != ft {
                continue;
            }
        }
        if let Some(ref prefix) = filters.path_prefix {
            if !source_path.starts_with(prefix) {
                continue;
            }
        }

        let snippet = mcp_hybrid_search_common::types::truncate_snippet(&body, 200);

        results.push(SearchResult {
            chunk_id,
            score: score as f64,
            title,
            source_path,
            source_type,
            snippet,
        });
    }

    Ok(results)
}

/// Get total document count.
pub fn get_index_count(config: &AppConfig) -> Result<u64> {
    let index = open_or_create_index(config)?;
    let reader = index
        .reader_builder()
        .reload_policy(ReloadPolicy::OnCommitWithDelay)
        .try_into()?;
    let searcher = reader.searcher();
    Ok(searcher.num_docs())
}

fn get_field_text(doc: &tantivy::TantivyDocument, field: tantivy::schema::Field) -> String {
    doc.get_first(field)
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string()
}
