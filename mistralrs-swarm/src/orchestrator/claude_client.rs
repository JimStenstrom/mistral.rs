//! Claude API client for orchestrator communication

use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};

const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Client for the Claude/Anthropic API
pub struct ClaudeClient {
    client: Client,
    api_key: String,
    base_url: String,
}

impl ClaudeClient {
    pub fn new(api_key: &str, base_url: &str) -> Result<Self> {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .build()?;

        Ok(Self {
            client,
            api_key: api_key.to_string(),
            base_url: base_url.to_string(),
        })
    }

    /// Send a message to Claude and get a response
    pub async fn send_message(
        &self,
        model: &str,
        system: &str,
        content: &str,
        max_tokens: usize,
    ) -> Result<String> {
        let request = MessagesRequest {
            model: model.to_string(),
            max_tokens,
            system: Some(system.to_string()),
            messages: vec![Message {
                role: "user".to_string(),
                content: MessageContent::Text(content.to_string()),
            }],
        };

        let response = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!(
                "Claude API error: {} - {}",
                status,
                body
            ));
        }

        let response: MessagesResponse = response.json().await?;

        // Extract text from response
        let text = response
            .content
            .into_iter()
            .filter_map(|block| {
                if let ContentBlock::Text { text } = block {
                    Some(text)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("");

        Ok(text)
    }

    /// Send a message with tool use enabled
    pub async fn send_message_with_tools(
        &self,
        model: &str,
        system: &str,
        messages: Vec<Message>,
        tools: Vec<ToolDefinition>,
        max_tokens: usize,
    ) -> Result<MessagesResponse> {
        let request = MessagesRequestWithTools {
            model: model.to_string(),
            max_tokens,
            system: Some(system.to_string()),
            messages,
            tools: Some(tools),
        };

        let response = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!(
                "Claude API error: {} - {}",
                status,
                body
            ));
        }

        let response: MessagesResponse = response.json().await?;
        Ok(response)
    }

    /// Stream a response from Claude
    pub async fn stream_message(
        &self,
        model: &str,
        system: &str,
        content: &str,
        max_tokens: usize,
    ) -> Result<impl futures::Stream<Item = Result<StreamEvent>>> {
        use futures::StreamExt;

        let request = MessagesRequest {
            model: model.to_string(),
            max_tokens,
            system: Some(system.to_string()),
            messages: vec![Message {
                role: "user".to_string(),
                content: MessageContent::Text(content.to_string()),
            }],
        };

        let response = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("content-type", "application/json")
            .header("accept", "text/event-stream")
            .json(&serde_json::json!({
                "model": request.model,
                "max_tokens": request.max_tokens,
                "system": request.system,
                "messages": request.messages,
                "stream": true
            }))
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!(
                "Claude API error: {} - {}",
                status,
                body
            ));
        }

        let stream = response.bytes_stream().map(|result| {
            result
                .map_err(|e| anyhow::anyhow!("Stream error: {}", e))
                .and_then(|bytes| {
                    let text = String::from_utf8_lossy(&bytes);
                    // Parse SSE events
                    for line in text.lines() {
                        if line.starts_with("data: ") {
                            let data = &line[6..];
                            if data == "[DONE]" {
                                return Ok(StreamEvent::Done);
                            }
                            if let Ok(event) = serde_json::from_str::<StreamEventData>(data) {
                                return Ok(StreamEvent::Data(event));
                            }
                        }
                    }
                    Ok(StreamEvent::Ping)
                })
        });

        Ok(stream)
    }
}

/// Messages API request
#[derive(Debug, Serialize)]
struct MessagesRequest {
    model: String,
    max_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    messages: Vec<Message>,
}

/// Messages API request with tools
#[derive(Debug, Serialize)]
struct MessagesRequestWithTools {
    model: String,
    max_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ToolDefinition>>,
}

/// A message in the conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: MessageContent,
}

/// Message content (text or structured)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

/// Content block in a message
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
    },
}

/// Tool definition for Claude
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

/// Messages API response
#[derive(Debug, Deserialize)]
pub struct MessagesResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub response_type: String,
    pub role: String,
    pub content: Vec<ContentBlock>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: Usage,
}

/// Token usage
#[derive(Debug, Deserialize)]
pub struct Usage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

/// Streaming event types
#[derive(Debug)]
pub enum StreamEvent {
    Data(StreamEventData),
    Ping,
    Done,
}

/// Streaming event data
#[derive(Debug, Deserialize)]
pub struct StreamEventData {
    #[serde(rename = "type")]
    pub event_type: String,
    pub index: Option<usize>,
    pub delta: Option<DeltaContent>,
}

#[derive(Debug, Deserialize)]
pub struct DeltaContent {
    #[serde(rename = "type")]
    pub content_type: Option<String>,
    pub text: Option<String>,
}
