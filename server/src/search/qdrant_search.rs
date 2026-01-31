use anyhow::Result;
use mcp_hybrid_search_common::config::AppConfig;
use mcp_hybrid_search_common::types::{ChunkDetail, ChunkMetadata, SearchFilters, SearchResult};
use qdrant_client::qdrant::{Condition, Filter, GetPointsBuilder, PointId, SearchPointsBuilder};
use qdrant_client::Qdrant;

pub async fn search(
    config: &AppConfig,
    query_embedding: &[f32],
    top_k: usize,
    filters: &SearchFilters,
) -> Result<Vec<SearchResult>> {
    let client = Qdrant::from_url(&config.qdrant_url).build()?;

    let mut conditions = Vec::new();

    if let Some(ref source_type) = filters.source_type {
        conditions.push(Condition::matches("source_type", source_type.clone()));
    }

    let mut builder = SearchPointsBuilder::new(
        &config.collection_name,
        query_embedding.to_vec(),
        top_k as u64,
    )
    .with_payload(true);

    if !conditions.is_empty() {
        builder = builder.filter(Filter::must(conditions));
    }

    let results = client.search_points(builder).await?;

    let search_results: Vec<SearchResult> = results
        .result
        .iter()
        .map(|point| {
            let payload = &point.payload;
            let chunk_id = get_str(payload, "chunk_id");
            let title = get_str(payload, "title");
            let source_path = get_str(payload, "source_path");
            let source_type = get_str(payload, "source_type");
            let text = get_str(payload, "text");
            let snippet = mcp_hybrid_search_common::types::truncate_snippet(&text, 200);

            SearchResult {
                chunk_id,
                score: point.score as f64,
                title,
                source_path,
                source_type,
                snippet,
            }
        })
        .collect();

    Ok(search_results)
}

pub async fn get_chunk(config: &AppConfig, chunk_id: &str) -> Result<Option<ChunkDetail>> {
    let client = Qdrant::from_url(&config.qdrant_url).build()?;
    let point_id: PointId = chunk_id.to_string().into();

    let response = client
        .get_points(GetPointsBuilder::new(&config.collection_name, &[point_id]).with_payload(true))
        .await?;

    if let Some(point) = response.result.first() {
        let payload = &point.payload;
        Ok(Some(ChunkDetail {
            chunk_id: get_str(payload, "chunk_id"),
            text: get_str(payload, "text"),
            metadata: ChunkMetadata {
                title: get_str(payload, "title"),
                source_path: get_str(payload, "source_path"),
                source_type: get_str(payload, "source_type"),
                chunk_index: get_str(payload, "chunk_index").parse().unwrap_or(0),
            },
        }))
    } else {
        Ok(None)
    }
}

/// Get the number of points in the collection.
pub async fn get_collection_count(config: &AppConfig) -> Result<u64> {
    let client = Qdrant::from_url(&config.qdrant_url).build()?;
    let info = client.collection_info(&config.collection_name).await?;
    Ok(info
        .result
        .map(|r| r.points_count.unwrap_or(0))
        .unwrap_or(0))
}

fn get_str(
    payload: &std::collections::HashMap<String, qdrant_client::qdrant::Value>,
    key: &str,
) -> String {
    payload
        .get(key)
        .and_then(|v| {
            if let Some(ref kind) = v.kind {
                match kind {
                    qdrant_client::qdrant::value::Kind::StringValue(s) => Some(s.clone()),
                    qdrant_client::qdrant::value::Kind::IntegerValue(i) => Some(i.to_string()),
                    qdrant_client::qdrant::value::Kind::DoubleValue(d) => Some(d.to_string()),
                    _ => None,
                }
            } else {
                None
            }
        })
        .unwrap_or_default()
}
