use anyhow::Result;
use mcp_hybrid_search_common::config::AppConfig;
use mcp_hybrid_search_common::types::{ChunkPayload, ExportedChunk, SearchFilters, SearchResult};
use qdrant_client::qdrant::{
    Condition, CreateCollectionBuilder, Distance, Filter, GetPointsBuilder, PointId, PointStruct,
    ScalarQuantizationBuilder, ScrollPointsBuilder, SearchPointsBuilder, UpsertPointsBuilder,
    VectorParamsBuilder,
};
use qdrant_client::Qdrant;
use serde_json::Value;
use uuid::Uuid;

/// Ensure the collection exists, creating it if necessary.
pub async fn ensure_collection(config: &AppConfig) -> Result<()> {
    let client = Qdrant::from_url(&config.qdrant_url).build()?;

    let exists = client.collection_exists(&config.collection_name).await?;

    if !exists {
        client
            .create_collection(
                CreateCollectionBuilder::new(&config.collection_name)
                    .vectors_config(VectorParamsBuilder::new(
                        config.embedding_dimension as u64,
                        Distance::Cosine,
                    ))
                    .quantization_config(ScalarQuantizationBuilder::default()),
            )
            .await?;
        tracing::info!("Created Qdrant collection '{}'", config.collection_name);
    } else {
        tracing::info!(
            "Qdrant collection '{}' already exists",
            config.collection_name
        );
    }

    Ok(())
}

/// Upsert chunks with their embeddings into Qdrant.
pub async fn upsert_chunks(
    config: &AppConfig,
    chunks: &[ChunkPayload],
    embeddings: &[Vec<f32>],
) -> Result<()> {
    let client = Qdrant::from_url(&config.qdrant_url).build()?;

    let points: Vec<PointStruct> = chunks
        .iter()
        .zip(embeddings.iter())
        .map(|(chunk, emb)| {
            let payload = serde_json::to_value(chunk).unwrap();
            let payload_map: std::collections::HashMap<String, Value> =
                serde_json::from_value(payload).unwrap();

            let id = Uuid::parse_str(&chunk.chunk_id).unwrap_or_else(|_| Uuid::new_v4());

            PointStruct::new(id.to_string(), emb.clone(), payload_map)
        })
        .collect();

    // Upsert in batches of 100
    for batch in points.chunks(100) {
        client
            .upsert_points(UpsertPointsBuilder::new(
                &config.collection_name,
                batch.to_vec(),
            ))
            .await?;
    }

    Ok(())
}

/// Search Qdrant for similar vectors.
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
            let chunk_id = get_payload_str(payload, "chunk_id");
            let title = get_payload_str(payload, "title");
            let source_path = get_payload_str(payload, "source_path");
            let source_type = get_payload_str(payload, "source_type");
            let text = get_payload_str(payload, "text");
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

/// Get a chunk by its chunk_id from Qdrant.
#[allow(dead_code)]
pub async fn get_chunk(config: &AppConfig, chunk_id: &str) -> Result<Option<ChunkPayload>> {
    let client = Qdrant::from_url(&config.qdrant_url).build()?;

    let point_id: PointId = chunk_id.to_string().into();

    let response = client
        .get_points(GetPointsBuilder::new(&config.collection_name, &[point_id]).with_payload(true))
        .await?;

    if let Some(point) = response.result.first() {
        let payload = &point.payload;
        let chunk = ChunkPayload {
            chunk_id: get_payload_str(payload, "chunk_id"),
            source_path: get_payload_str(payload, "source_path"),
            source_type: get_payload_str(payload, "source_type"),
            title: get_payload_str(payload, "title"),
            chunk_index: get_payload_str(payload, "chunk_index").parse().unwrap_or(0),
            text: get_payload_str(payload, "text"),
            updated_at: get_payload_str(payload, "updated_at"),
        };
        Ok(Some(chunk))
    } else {
        Ok(None)
    }
}

/// Delete the Qdrant collection.
pub async fn delete_collection(config: &AppConfig) -> Result<()> {
    let client = Qdrant::from_url(&config.qdrant_url).build()?;
    client.delete_collection(&config.collection_name).await?;
    Ok(())
}

/// Get collection point count.
pub async fn get_collection_info(config: &AppConfig) -> Result<u64> {
    let client = Qdrant::from_url(&config.qdrant_url).build()?;

    let info = client.collection_info(&config.collection_name).await?;
    Ok(info
        .result
        .map(|r| r.points_count.unwrap_or(0))
        .unwrap_or(0))
}

/// Scroll all points from the collection, returning chunks with their embeddings.
pub async fn export_all_chunks(config: &AppConfig) -> Result<Vec<ExportedChunk>> {
    let client = Qdrant::from_url(&config.qdrant_url).build()?;

    let mut all_chunks = Vec::new();
    let mut offset: Option<PointId> = None;
    let batch_size = 100u32;

    loop {
        let mut builder = ScrollPointsBuilder::new(&config.collection_name)
            .with_payload(true)
            .with_vectors(true)
            .limit(batch_size);

        if let Some(ref next_offset) = offset {
            builder = builder.offset(next_offset.clone());
        }

        let response = client.scroll(builder).await?;

        if response.result.is_empty() {
            break;
        }

        for point in &response.result {
            let payload = &point.payload;
            let chunk_payload = ChunkPayload {
                chunk_id: get_payload_str(payload, "chunk_id"),
                source_path: get_payload_str(payload, "source_path"),
                source_type: get_payload_str(payload, "source_type"),
                title: get_payload_str(payload, "title"),
                chunk_index: get_payload_str(payload, "chunk_index").parse().unwrap_or(0),
                text: get_payload_str(payload, "text"),
                updated_at: get_payload_str(payload, "updated_at"),
            };

            #[allow(deprecated)]
            let embedding = point
                .vectors
                .as_ref()
                .and_then(|v| {
                    use qdrant_client::qdrant::vectors_output::VectorsOptions;
                    match &v.vectors_options {
                        Some(VectorsOptions::Vector(vec_output)) => Some(vec_output.data.clone()),
                        _ => None,
                    }
                })
                .unwrap_or_default();

            all_chunks.push(ExportedChunk {
                payload: chunk_payload,
                embedding,
            });
        }

        tracing::info!("Exported {} chunks so far...", all_chunks.len());

        offset = response.next_page_offset;
        if offset.is_none() {
            break;
        }
    }

    Ok(all_chunks)
}

fn get_payload_str(
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
