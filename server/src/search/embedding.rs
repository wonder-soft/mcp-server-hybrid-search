use anyhow::Result;
use mcp_hybrid_search_common::config::AppConfig;
use serde::{Deserialize, Serialize};

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

pub async fn get_embedding(config: &AppConfig, text: &str) -> Result<Vec<f32>> {
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
