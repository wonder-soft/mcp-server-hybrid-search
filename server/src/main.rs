mod mcp;
mod search;

use clap::Parser;

use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive},
        Sse,
    },
    routing::{get, post},
    Json, Router,
};
use futures::stream::Stream;
use mcp_hybrid_search_common::config::AppConfig;
use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(
    name = "mcp-server-hybrid-search",
    about = "MCP server for hybrid document search"
)]
struct Args {
    /// Path to config.toml
    #[arg(long)]
    config: Option<String>,

    /// Project name for collection isolation
    #[arg(long)]
    project: Option<String>,
}

type SessionId = String;
type Sessions = Arc<RwLock<HashMap<SessionId, mpsc::Sender<String>>>>;

pub struct AppState {
    pub config: AppConfig,
    pub mcp_server: Arc<RwLock<mcp::server::McpServer>>,
    pub sessions: Sessions,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info,mcp_server_hybrid_search=debug")),
        )
        .init();

    let args = Args::parse();
    let config = AppConfig::load(args.config.as_deref())?;
    let config = config.with_project(args.project.as_deref());
    let listen_port = config.listen_port;

    if let Some(ref proj) = args.project {
        tracing::info!("Project: {}", proj);
    }

    let mcp_server = mcp::server::McpServer::new(config.clone()).await?;

    let state = Arc::new(AppState {
        config,
        mcp_server: Arc::new(RwLock::new(mcp_server)),
        sessions: Arc::new(RwLock::new(HashMap::new())),
    });

    let app = Router::new()
        .route("/sse", get(sse_handler))
        .route("/message", post(message_handler))
        .route("/health", get(health_handler))
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = format!("0.0.0.0:{}", listen_port);
    tracing::info!("MCP server starting on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn sse_handler(
    State(state): State<Arc<AppState>>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let session_id = uuid::Uuid::new_v4().to_string();
    let (tx, mut rx) = mpsc::channel::<String>(100);

    state.sessions.write().await.insert(session_id.clone(), tx);

    tracing::info!("SSE connection established: {}", session_id);

    let sessions = state.sessions.clone();
    let sid = session_id.clone();

    let stream = async_stream::stream! {
        // Send the endpoint URL
        let endpoint = format!("/message?sessionId={}", sid);
        yield Ok(Event::default().event("endpoint").data(endpoint));

        // Stream messages
        loop {
            match rx.recv().await {
                Some(msg) => {
                    yield Ok(Event::default().event("message").data(msg));
                }
                None => {
                    tracing::info!("SSE session closed: {}", sid);
                    break;
                }
            }
        }

        // Cleanup
        sessions.write().await.remove(&sid);
    };

    Sse::new(stream).keep_alive(KeepAlive::default())
}

#[derive(serde::Deserialize)]
struct MessageQuery {
    #[serde(rename = "sessionId")]
    session_id: String,
}

async fn message_handler(
    State(state): State<Arc<AppState>>,
    Query(query): Query<MessageQuery>,
    Json(request): Json<mcp::protocol::JsonRpcRequest>,
) -> StatusCode {
    tracing::debug!(
        "Received message for session {}: method={}",
        query.session_id,
        request.method
    );

    let response = {
        let server = state.mcp_server.read().await;
        server.handle_request(request).await
    };

    let response_json = match serde_json::to_string(&response) {
        Ok(json) => json,
        Err(e) => {
            tracing::error!("Failed to serialize response: {}", e);
            return StatusCode::INTERNAL_SERVER_ERROR;
        }
    };

    let sessions = state.sessions.read().await;
    if let Some(tx) = sessions.get(&query.session_id) {
        if tx.send(response_json).await.is_err() {
            tracing::warn!("Failed to send response to session {}", query.session_id);
            return StatusCode::GONE;
        }
    } else {
        tracing::warn!("Session not found: {}", query.session_id);
        return StatusCode::NOT_FOUND;
    }

    StatusCode::ACCEPTED
}

async fn health_handler() -> &'static str {
    "ok"
}
