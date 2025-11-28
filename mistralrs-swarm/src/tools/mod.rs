//! Tool system for worker agents
//!
//! Provides a registry of tools that workers can use during task execution.

mod builtins;

pub use builtins::*;

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

/// A tool that can be called by worker agents
#[async_trait]
pub trait Tool: Send + Sync {
    /// Tool name (must be unique)
    fn name(&self) -> &str;

    /// Human-readable description
    fn description(&self) -> &str;

    /// JSON Schema for parameters
    fn parameters_schema(&self) -> Value;

    /// Execute the tool with given arguments
    async fn execute(&self, arguments: Value) -> Result<ToolResult>;

    /// Whether this tool is async (may take time)
    fn is_async(&self) -> bool {
        true
    }

    /// Timeout in seconds (0 = no timeout)
    fn timeout_secs(&self) -> u64 {
        30
    }
}

/// Result from a tool execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// Whether execution succeeded
    pub success: bool,
    /// Output content
    pub output: String,
    /// Any error message
    pub error: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, Value>,
}

impl ToolResult {
    pub fn success(output: impl Into<String>) -> Self {
        Self {
            success: true,
            output: output.into(),
            error: None,
            metadata: HashMap::new(),
        }
    }

    pub fn error(error: impl Into<String>) -> Self {
        Self {
            success: false,
            output: String::new(),
            error: Some(error.into()),
            metadata: HashMap::new(),
        }
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

/// A tool call request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique call ID
    pub id: String,
    /// Tool name to invoke
    pub name: String,
    /// Arguments as JSON
    pub arguments: Value,
}

impl ToolCall {
    pub fn new(name: impl Into<String>, arguments: Value) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            name: name.into(),
            arguments,
        }
    }
}

/// Registry of available tools
#[derive(Default)]
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a registry with default built-in tools
    pub fn with_builtins() -> Self {
        let mut registry = Self::new();
        registry.register(Arc::new(ReadFileTool));
        registry.register(Arc::new(WriteFileTool));
        registry.register(Arc::new(ListDirectoryTool));
        registry.register(Arc::new(SearchFilesTool));
        registry.register(Arc::new(ExecuteBashTool::new()));
        registry.register(Arc::new(GrepTool));
        registry
    }

    /// Register a tool
    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    /// Get a tool by name
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    /// Execute a tool call
    pub async fn execute(&self, call: &ToolCall) -> Result<ToolResult> {
        let tool = self
            .get(&call.name)
            .ok_or_else(|| anyhow::anyhow!("Tool not found: {}", call.name))?;

        // Apply timeout if specified
        let timeout = tool.timeout_secs();
        if timeout > 0 {
            match tokio::time::timeout(
                std::time::Duration::from_secs(timeout),
                tool.execute(call.arguments.clone()),
            )
            .await
            {
                Ok(result) => result,
                Err(_) => Ok(ToolResult::error(format!(
                    "Tool '{}' timed out after {}s",
                    call.name, timeout
                ))),
            }
        } else {
            tool.execute(call.arguments.clone()).await
        }
    }

    /// Get all tools as OpenAI-compatible tool definitions
    pub fn to_openai_tools(&self) -> Vec<Value> {
        self.tools
            .values()
            .map(|tool| {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": tool.name(),
                        "description": tool.description(),
                        "parameters": tool.parameters_schema()
                    }
                })
            })
            .collect()
    }

    /// Get tool definitions for mistral.rs format
    pub fn to_mistralrs_tools(&self) -> Vec<mistralrs::Tool> {
        self.tools
            .values()
            .map(|tool| {
                let params: HashMap<String, Value> =
                    serde_json::from_value(tool.parameters_schema()).unwrap_or_default();
                mistralrs::Tool {
                    tp: mistralrs::ToolType::Function,
                    function: mistralrs::Function {
                        name: tool.name().to_string(),
                        description: Some(tool.description().to_string()),
                        parameters: Some(params),
                    },
                }
            })
            .collect()
    }

    /// List all registered tool names
    pub fn list_tools(&self) -> Vec<&str> {
        self.tools.keys().map(|s| s.as_str()).collect()
    }
}

impl Clone for ToolRegistry {
    fn clone(&self) -> Self {
        Self {
            tools: self.tools.clone(),
        }
    }
}
