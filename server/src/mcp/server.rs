use mcp_hybrid_search_common::config::AppConfig;
use mcp_hybrid_search_common::types::SearchFilters;
use serde_json::{json, Value};

use super::protocol::*;
use super::tools::*;
use crate::search;

pub struct McpServer {
    config: AppConfig,
    searcher: search::HybridSearcher,
}

impl McpServer {
    pub async fn new(config: AppConfig) -> anyhow::Result<Self> {
        let searcher = search::HybridSearcher::new(&config)?;
        Ok(Self { config, searcher })
    }

    pub async fn handle_request(&self, request: JsonRpcRequest) -> JsonRpcResponse {
        tracing::debug!("Handling method: {}", request.method);

        match request.method.as_str() {
            "initialize" => self.handle_initialize(request.id),
            "initialized" => JsonRpcResponse::success(request.id, json!({})),
            "notifications/initialized" => JsonRpcResponse::success(request.id, json!({})),
            "tools/list" => self.handle_tools_list(request.id),
            "tools/call" => self.handle_tools_call(request.id, request.params).await,
            "ping" => JsonRpcResponse::success(request.id, json!({})),
            _ => {
                tracing::warn!("Unknown method: {}", request.method);
                JsonRpcResponse::error(
                    request.id,
                    METHOD_NOT_FOUND,
                    format!("Method not found: {}", request.method),
                )
            }
        }
    }

    fn handle_initialize(&self, id: Option<Value>) -> JsonRpcResponse {
        JsonRpcResponse::success(
            id,
            json!({
                "protocolVersion": "2024-11-05",
                "serverInfo": {
                    "name": "mcp-server-hybrid-search",
                    "version": env!("CARGO_PKG_VERSION")
                },
                "capabilities": {
                    "tools": {
                        "listChanged": false
                    }
                }
            }),
        )
    }

    fn handle_tools_list(&self, id: Option<Value>) -> JsonRpcResponse {
        let tools = list_tools();
        JsonRpcResponse::success(id, json!({ "tools": tools }))
    }

    async fn handle_tools_call(
        &self,
        id: Option<Value>,
        params: Option<Value>,
    ) -> JsonRpcResponse {
        let params = match params {
            Some(p) => p,
            None => {
                return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing params");
            }
        };

        let tool_name = params
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let arguments = params.get("arguments").cloned().unwrap_or(json!({}));

        let result = match ToolName::parse(tool_name) {
            Some(ToolName::Search) => self.execute_search(arguments).await,
            Some(ToolName::Get) => self.execute_get(arguments).await,
            None => {
                return JsonRpcResponse::error(
                    id,
                    METHOD_NOT_FOUND,
                    format!("Unknown tool: {}", tool_name),
                );
            }
        };

        match result {
            Ok(tool_result) => JsonRpcResponse::success(id, json!(tool_result)),
            Err(e) => {
                let error_result = ToolResult::error(format!("Error: {}", e));
                JsonRpcResponse::success(id, json!(error_result))
            }
        }
    }

    async fn execute_search(&self, arguments: Value) -> anyhow::Result<ToolResult> {
        let args: SearchArgs = serde_json::from_value(arguments)?;
        let top_k = args.top_k.unwrap_or(10);

        let filters = SearchFilters {
            source_type: args.filters.as_ref().and_then(|f| f.source_type.clone()),
            path_prefix: args.filters.as_ref().and_then(|f| f.path_prefix.clone()),
        };

        let results = self
            .searcher
            .search(&self.config, &args.query, top_k, &filters)
            .await?;

        let output = json!({
            "results": results
        });

        Ok(ToolResult::text(serde_json::to_string_pretty(&output)?))
    }

    async fn execute_get(&self, arguments: Value) -> anyhow::Result<ToolResult> {
        let args: GetArgs = serde_json::from_value(arguments)?;

        let chunk = self
            .searcher
            .get_chunk(&self.config, &args.chunk_id)
            .await?;

        match chunk {
            Some(detail) => {
                let output = json!(detail);
                Ok(ToolResult::text(serde_json::to_string_pretty(&output)?))
            }
            None => Ok(ToolResult::error(format!(
                "Chunk not found: {}",
                args.chunk_id
            ))),
        }
    }
}
