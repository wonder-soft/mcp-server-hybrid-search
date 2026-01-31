use anyhow::Result;
use mcp_hybrid_search_common::config::AppConfig;
use serde::{Deserialize, Serialize};

/// Get embedding for a single text, dispatching based on config.embedding_provider.
pub async fn get_embedding(config: &AppConfig, text: &str) -> Result<Vec<f32>> {
    let embeddings = get_embeddings(config, &[text.to_string()]).await?;
    embeddings
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("No embedding returned"))
}

/// Get embeddings for multiple texts, dispatching based on config.embedding_provider.
pub async fn get_embeddings(config: &AppConfig, texts: &[String]) -> Result<Vec<Vec<f32>>> {
    match config.embedding_provider.as_str() {
        "openai" => get_embeddings_openai(config, texts).await,
        "local" => get_embeddings_local(config, texts),
        other => anyhow::bail!(
            "Unknown embedding_provider '{}'. Supported: openai, local",
            other
        ),
    }
}

// --- OpenAI provider ---

#[derive(Serialize)]
struct EmbeddingRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

async fn get_embeddings_openai(config: &AppConfig, texts: &[String]) -> Result<Vec<Vec<f32>>> {
    let api_key = std::env::var("OPENAI_API_KEY")
        .map_err(|_| anyhow::anyhow!("OPENAI_API_KEY environment variable not set"))?;

    let base_url =
        std::env::var("OPENAI_API_BASE").unwrap_or_else(|_| "https://api.openai.com/v1".into());

    let client = reqwest::Client::new();
    let request = EmbeddingRequest {
        model: config.embedding_model.clone(),
        input: texts.to_vec(),
    };

    let response = client
        .post(format!("{}/embeddings", base_url))
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .await?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!("OpenAI API error ({}): {}", status, body);
    }

    let resp: EmbeddingResponse = response.json().await?;
    Ok(resp.data.into_iter().map(|d| d.embedding).collect())
}

// --- Local provider (fastembed) ---

#[cfg(feature = "local-embed")]
fn get_embeddings_local(config: &AppConfig, texts: &[String]) -> Result<Vec<Vec<f32>>> {
    use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

    let model_type = resolve_local_model(&config.embedding_model)?;
    let mut model = TextEmbedding::try_new(InitOptions::new(model_type))?;

    // E5 models expect "passage: " prefix for documents and "query: " for queries.
    // During ingest we treat all texts as passages.
    let prefixed: Vec<String> = texts.iter().map(|t| format!("passage: {}", t)).collect();

    let embeddings = model.embed(prefixed, None)?;
    Ok(embeddings)
}

#[cfg(feature = "local-embed")]
fn resolve_local_model(model_name: &str) -> Result<fastembed::EmbeddingModel> {
    use fastembed::EmbeddingModel;
    match model_name {
        "multilingual-e5-small" => Ok(EmbeddingModel::MultilingualE5Small),
        "multilingual-e5-base" => Ok(EmbeddingModel::MultilingualE5Base),
        _ => anyhow::bail!(
            "Unknown local embedding model '{}'. Supported: multilingual-e5-small, multilingual-e5-base",
            model_name
        ),
    }
}

#[cfg(not(feature = "local-embed"))]
fn get_embeddings_local(_config: &AppConfig, _texts: &[String]) -> Result<Vec<Vec<f32>>> {
    anyhow::bail!(
        "embedding_provider = \"local\" requires the 'local-embed' feature. \
         Build with: cargo build --features local-embed"
    )
}
