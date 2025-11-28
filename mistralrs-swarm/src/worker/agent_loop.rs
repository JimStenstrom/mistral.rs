//! Agent loop implementation for worker agents

use crate::protocol::{Task, TokenUsage, ToolCallRecord};
use crate::tools::{ToolCall, ToolRegistry};
use anyhow::Result;
use mistralrs::{RequestBuilder, TextMessageRole, ToolChoice};
use std::sync::Arc;
use tracing::{debug, info};

/// The core agent loop that drives worker reasoning
pub struct AgentLoop {
    model: Arc<mistralrs::Model>,
    tools: Arc<ToolRegistry>,
    system_prompt: String,
    max_steps: usize,
}

impl AgentLoop {
    pub fn new(
        model: Arc<mistralrs::Model>,
        tools: Arc<ToolRegistry>,
        system_prompt: &str,
        max_steps: usize,
    ) -> Self {
        Self {
            model,
            tools,
            system_prompt: system_prompt.to_string(),
            max_steps,
        }
    }

    /// Run the agent loop for a task
    pub async fn run(
        &self,
        task: &Task,
    ) -> Result<(String, Vec<ToolCallRecord>, TokenUsage)> {
        let mut tool_call_records = Vec::new();
        let mut total_tokens = TokenUsage::default();

        // Build initial prompt with task context
        let task_prompt = self.format_task_prompt(task);

        // Initialize conversation
        let mut messages = RequestBuilder::new()
            .add_message(TextMessageRole::System, &self.system_prompt)
            .add_message(TextMessageRole::User, &task_prompt)
            .set_tools(self.tools.to_mistralrs_tools())
            .set_tool_choice(ToolChoice::Auto);

        for step in 0..self.max_steps {
            debug!(step, "Agent loop step");

            // Get model response
            let response = self.model.send_chat_request(messages.clone()).await?;

            // Update token counts
            total_tokens.prompt_tokens += response.usage.prompt_tokens;
            total_tokens.completion_tokens += response.usage.completion_tokens;
            total_tokens.total_tokens += response.usage.total_tokens;

            let choice = &response.choices[0];
            let message = &choice.message;

            // Check for tool calls
            if let Some(tool_calls) = &message.tool_calls {
                // Process each tool call
                for tc in tool_calls {
                    info!(tool = %tc.function.name, "Executing tool call");

                    let call = ToolCall {
                        id: tc.id.clone(),
                        name: tc.function.name.clone(),
                        arguments: serde_json::from_str(&tc.function.arguments)
                            .unwrap_or(serde_json::Value::Object(Default::default())),
                    };

                    let start = std::time::Instant::now();
                    let result = self.tools.execute(&call).await?;
                    let duration_ms = start.elapsed().as_millis() as u64;

                    // Record the tool call
                    tool_call_records.push(ToolCallRecord {
                        tool: call.name.clone(),
                        arguments: call.arguments.clone(),
                        result: result.output.clone(),
                        duration_ms,
                        success: result.success,
                    });

                    // Add tool call and result to conversation
                    messages = messages
                        .add_message_with_tool_call(
                            TextMessageRole::Assistant,
                            String::new(),
                            vec![tc.clone()],
                        )
                        .add_tool_message(
                            if result.success {
                                result.output
                            } else {
                                format!("Error: {}", result.error.unwrap_or_default())
                            },
                            tc.id.clone(),
                        );
                }

                // Continue the loop to let model process tool results
                continue;
            }

            // No tool calls - check if we have a final response
            if let Some(content) = &message.content {
                if !content.trim().is_empty() {
                    // Check if this looks like a completion
                    if self.is_completion_response(content) {
                        info!(step, "Task completed with final response");
                        return Ok((content.clone(), tool_call_records, total_tokens));
                    }
                }
            }

            // Check stop reason
            if choice.finish_reason == "stop" {
                let output = message.content.clone().unwrap_or_default();
                info!(step, "Model stopped, returning response");
                return Ok((output, tool_call_records, total_tokens));
            }
        }

        // Max steps reached
        let final_output = format!(
            "Task did not complete within {} steps. Last progress: {:?}",
            self.max_steps,
            tool_call_records.last().map(|r| &r.tool)
        );

        Ok((final_output, tool_call_records, total_tokens))
    }

    /// Format the task into a prompt for the agent
    fn format_task_prompt(&self, task: &Task) -> String {
        let mut prompt = format!(
            "## Task\n\n{}\n\n## Instructions\n\n{}\n",
            task.description, task.instructions
        );

        // Add context if available
        if !task.context.relevant_files.is_empty() {
            prompt.push_str("\n## Relevant Files\n\n");
            for file in &task.context.relevant_files {
                prompt.push_str(&format!("- {}\n", file));
            }
        }

        if !task.context.constraints.is_empty() {
            prompt.push_str("\n## Constraints\n\n");
            for constraint in &task.context.constraints {
                prompt.push_str(&format!("- {}\n", constraint));
            }
        }

        if !task.context.success_criteria.is_empty() {
            prompt.push_str("\n## Success Criteria\n\n");
            for criterion in &task.context.success_criteria {
                prompt.push_str(&format!("- {}\n", criterion));
            }
        }

        prompt.push_str("\n## Instructions\n\nComplete this task using the available tools. When finished, provide a clear summary of what you accomplished.\n");

        prompt
    }

    /// Check if the response indicates task completion
    fn is_completion_response(&self, content: &str) -> bool {
        let lower = content.to_lowercase();

        // Look for completion indicators
        let completion_phrases = [
            "task complete",
            "task is complete",
            "completed the task",
            "have completed",
            "successfully completed",
            "finished",
            "done",
            "## summary",
            "## result",
            "## output",
            "in conclusion",
            "to summarize",
        ];

        completion_phrases.iter().any(|phrase| lower.contains(phrase))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_completion_detection() {
        let loop_ = AgentLoop::new(
            Arc::new(unimplemented!()), // Would need mock
            Arc::new(ToolRegistry::new()),
            "",
            10,
        );

        assert!(loop_.is_completion_response("Task complete. The file was created."));
        assert!(loop_.is_completion_response("## Summary\n\nI have finished the analysis."));
        assert!(!loop_.is_completion_response("Let me search for more files..."));
    }
}
