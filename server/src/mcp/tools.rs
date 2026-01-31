use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Debug, Clone, Serialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    #[serde(rename = "inputSchema")]
    pub input_schema: Value,
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolResult {
    pub content: Vec<ToolResultContent>,
    #[serde(rename = "isError")]
    pub is_error: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolResultContent {
    #[serde(rename = "type")]
    pub content_type: String,
    pub text: String,
}

impl ToolResult {
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: vec![ToolResultContent {
                content_type: "text".to_string(),
                text: text.into(),
            }],
            is_error: false,
        }
    }

    pub fn error(text: impl Into<String>) -> Self {
        Self {
            content: vec![ToolResultContent {
                content_type: "text".to_string(),
                text: text.into(),
            }],
            is_error: true,
        }
    }
}

pub enum ToolName {
    Search,
    Get,
    GetProjectInfo,
}

impl ToolName {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "search" => Some(Self::Search),
            "get" => Some(Self::Get),
            "get_project_info" => Some(Self::GetProjectInfo),
            _ => None,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct SearchArgs {
    pub query: String,
    pub top_k: Option<usize>,
    pub filters: Option<FilterArgs>,
}

#[derive(Debug, Deserialize)]
pub struct FilterArgs {
    pub source_type: Option<String>,
    pub path_prefix: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct GetArgs {
    pub chunk_id: String,
}

pub fn list_tools() -> Vec<Tool> {
    vec![
        Tool {
            name: "search".to_string(),
            description: "Search documents using hybrid search (vector + BM25). Returns ranked results from indexed documents.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query text"
                    },
                    "top_k": {
                        "type": "number",
                        "description": "Number of results to return (default: 10)"
                    },
                    "filters": {
                        "type": "object",
                        "properties": {
                            "source_type": {
                                "type": "string",
                                "description": "Filter by file type (md/txt/pdf/xlsx)"
                            },
                            "path_prefix": {
                                "type": "string",
                                "description": "Filter by path prefix"
                            }
                        }
                    }
                },
                "required": ["query"]
            }),
        },
        Tool {
            name: "get".to_string(),
            description: "Get the full content of a specific document chunk by its chunk_id.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "chunk_id": {
                        "type": "string",
                        "description": "The unique identifier of the chunk"
                    }
                },
                "required": ["chunk_id"]
            }),
        },
        Tool {
            name: "get_project_info".to_string(),
            description: "Get information about the current project: collection name, document count, tantivy index directory, and embedding settings.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        },
    ]
}
