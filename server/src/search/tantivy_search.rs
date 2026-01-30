use anyhow::Result;
use mcp_hybrid_search_common::config::AppConfig;
use mcp_hybrid_search_common::types::{SearchFilters, SearchResult};
use std::path::Path;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::*;
use tantivy::{Index, ReloadPolicy};

fn build_schema() -> Schema {
    let mut schema_builder = Schema::builder();
    schema_builder.add_text_field("chunk_id", STRING | STORED);
    schema_builder.add_text_field("source_path", STRING | STORED);
    schema_builder.add_text_field("title", TEXT | STORED);
    schema_builder.add_text_field("body", TEXT | STORED);
    schema_builder.add_text_field("source_type", STRING | STORED);
    schema_builder.build()
}

fn open_index(index_dir: &str) -> Result<Index> {
    let path = Path::new(index_dir);
    if path.exists() {
        Ok(Index::open_in_dir(path)?)
    } else {
        std::fs::create_dir_all(path)?;
        Ok(Index::create_in_dir(path, build_schema())?)
    }
}

pub fn search(
    config: &AppConfig,
    query_str: &str,
    top_k: usize,
    filters: &SearchFilters,
) -> Result<Vec<SearchResult>> {
    let index = open_index(&config.tantivy_index_dir)?;
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
        let doc: tantivy::TantivyDocument = searcher.doc(doc_address)?;

        let chunk_id = get_text(&doc, chunk_id_field);
        let source_path = get_text(&doc, source_path_field);
        let title = get_text(&doc, title_field);
        let body = get_text(&doc, body_field);
        let source_type = get_text(&doc, source_type_field);

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

        let snippet = if body.len() > 200 {
            format!("{}...", &body[..200])
        } else {
            body
        };

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

fn get_text(doc: &tantivy::TantivyDocument, field: Field) -> String {
    doc.get_first(field)
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string()
}
