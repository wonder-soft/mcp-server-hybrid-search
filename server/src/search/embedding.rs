use anyhow::Result;
use mcp_hybrid_search_common::config::AppConfig;
use serde::{Deserialize, Serialize};

/// Get embedding for a query string, dispatching based on config.embedding_provider.
pub async fn get_embedding(config: &AppConfig, text: &str) -> Result<Vec<f32>> {
    match config.embedding_provider.as_str() {
        "openai" => get_embedding_openai(config, text).await,
        "gemini" => get_embedding_gemini(config, text).await,
        "local" => get_embedding_local(config, text),
        other => anyhow::bail!(
            "Unknown embedding_provider '{}'. Supported: openai, gemini, local",
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

async fn get_embedding_openai(config: &AppConfig, text: &str) -> Result<Vec<f32>> {
    let api_key = std::env::var("OPENAI_API_KEY")
        .map_err(|_| anyhow::anyhow!("OPENAI_API_KEY environment variable not set"))?;

    let base_url =
        std::env::var("OPENAI_API_BASE").unwrap_or_else(|_| "https://api.openai.com/v1".into());

    let client = reqwest::Client::new();
    let request = EmbeddingRequest {
        model: config.embedding_model.clone(),
        input: vec![text.to_string()],
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
    resp.data
        .into_iter()
        .next()
        .map(|d| d.embedding)
        .ok_or_else(|| anyhow::anyhow!("No embedding returned"))
}

// --- Gemini provider ---

#[derive(Serialize)]
struct GeminiEmbedRequest {
    model: String,
    content: GeminiContent,
}

#[derive(Serialize)]
struct GeminiContent {
    parts: Vec<GeminiPart>,
}

#[derive(Serialize)]
struct GeminiPart {
    text: String,
}

#[derive(Deserialize)]
struct GeminiEmbedResponse {
    embedding: GeminiEmbeddingValues,
}

#[derive(Deserialize)]
struct GeminiEmbeddingValues {
    values: Vec<f32>,
}

async fn get_embedding_gemini(config: &AppConfig, text: &str) -> Result<Vec<f32>> {
    let api_key = std::env::var("GEMINI_API_KEY")
        .map_err(|_| anyhow::anyhow!("GEMINI_API_KEY environment variable not set"))?;

    let base_url = std::env::var("GEMINI_API_BASE")
        .unwrap_or_else(|_| "https://generativelanguage.googleapis.com/v1beta".into());

    let model = &config.embedding_model;
    let model_path = if model.starts_with("models/") {
        model.clone()
    } else {
        format!("models/{}", model)
    };

    let request = GeminiEmbedRequest {
        model: model_path.clone(),
        content: GeminiContent {
            parts: vec![GeminiPart {
                text: text.to_string(),
            }],
        },
    };

    let client = reqwest::Client::new();
    let url = format!("{}/{}:embedContent?key={}", base_url, model_path, api_key);

    let response = client
        .post(&url)
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .await?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!("Gemini API error ({}): {}", status, body);
    }

    let resp: GeminiEmbedResponse = response.json().await?;
    Ok(resp.embedding.values)
}

// --- Local provider (fastembed) ---

#[cfg(feature = "local-embed")]
fn get_embedding_local(config: &AppConfig, text: &str) -> Result<Vec<f32>> {
    use fastembed::{InitOptions, TextEmbedding};

    let model_type = resolve_local_model(&config.embedding_model)?;
    let mut model = TextEmbedding::try_new(InitOptions::new(model_type))?;

    // E5 models expect "query: " prefix for search queries
    let prefixed = format!("query: {}", text);
    let embeddings = model.embed(vec![prefixed], None)?;

    embeddings
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("No embedding returned"))
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
fn get_embedding_local(_config: &AppConfig, _text: &str) -> Result<Vec<f32>> {
    anyhow::bail!(
        "embedding_provider = \"local\" requires the 'local-embed' feature. \
         Build with: cargo build --features local-embed"
    )
}
