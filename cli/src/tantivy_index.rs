use anyhow::Result;
use mcp_hybrid_search_common::config::AppConfig;
use mcp_hybrid_search_common::types::{ChunkPayload, SearchFilters, SearchResult};
use std::path::Path;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::*;
use tantivy::{doc, Index, IndexWriter, ReloadPolicy};

/// Build the Tantivy schema.
fn build_schema() -> Schema {
    let mut schema_builder = Schema::builder();
    schema_builder.add_text_field("chunk_id", STRING | STORED);
    schema_builder.add_text_field("source_path", STRING | STORED);
    schema_builder.add_text_field("title", TEXT | STORED);
    schema_builder.add_text_field("body", TEXT | STORED);
    schema_builder.add_text_field("source_type", STRING | STORED);
    schema_builder.build()
}

/// Open or create the Tantivy index.
fn open_or_create_index(index_dir: &str) -> Result<Index> {
    let schema = build_schema();
    let path = Path::new(index_dir);

    if path.exists() {
        match Index::open_in_dir(path) {
            Ok(index) => return Ok(index),
            Err(e) => {
                tracing::warn!("Failed to open existing index, recreating: {}", e);
            }
        }
    }

    std::fs::create_dir_all(path)?;
    let index = Index::create_in_dir(path, schema)?;
    Ok(index)
}

/// Index chunks into Tantivy.
pub fn index_chunks(config: &AppConfig, chunks: &[ChunkPayload]) -> Result<()> {
    let index = open_or_create_index(&config.tantivy_index_dir)?;
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
    let index = open_or_create_index(&config.tantivy_index_dir)?;
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
    let index = open_or_create_index(&config.tantivy_index_dir)?;
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
