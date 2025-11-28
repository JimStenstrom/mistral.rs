//! Worker factory for spawning and managing worker agents
//!
//! Workers execute delegated tasks from the todo list in parallel.

use crate::todo::{TodoItem, TodoList, TodoStatus};
use anyhow::Result;
use colored::Colorize;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use mistralrs::{RequestBuilder, TextMessageRole, Tool, ToolChoice, ToolType, Function};
use mistralrs_swarm::tools::{ToolRegistry, ToolCall};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};

/// Configuration for workers
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    pub max_steps: usize,
    pub timeout_secs: u64,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            max_steps: 15,
            timeout_secs: 300,
        }
    }
}

/// A single worker that executes tasks
pub struct Worker {
    id: usize,
    model: Arc<mistralrs::Model>,
    tools: Arc<ToolRegistry>,
    config: WorkerConfig,
}

impl Worker {
    pub fn new(
        id: usize,
        model: Arc<mistralrs::Model>,
        tools: Arc<ToolRegistry>,
        config: WorkerConfig,
    ) -> Self {
        Self { id, model, tools, config }
    }

    /// Execute a task and return the result
    pub async fn execute(&self, task: &TodoItem, progress: Option<&ProgressBar>) -> WorkerResult {
        let start = std::time::Instant::now();

        if let Some(pb) = progress {
            pb.set_message(format!("Starting: {}", task.description));
        }

        // Build the task prompt
        let task_prompt = format!(
            "## Task\n\n{}\n\n## Details\n\n{}\n\nComplete this task using the available tools. When done, provide a clear summary of what you accomplished.",
            task.description,
            task.details.as_deref().unwrap_or("No additional details.")
        );

        let system_prompt = r#"You are a worker agent executing a specific task. Use the available tools to complete the task.

Guidelines:
- Use tools to interact with files, run commands, search, etc.
- Be efficient - don't call tools unnecessarily
- Provide a clear summary when done
- If you encounter errors, try to recover or report clearly"#;

        // Convert tools to mistralrs format
        let tools = self.tools.to_mistralrs_tools();

        let mut messages = RequestBuilder::new()
            .add_message(TextMessageRole::System, system_prompt)
            .add_message(TextMessageRole::User, &task_prompt)
            .set_tools(tools)
            .set_tool_choice(ToolChoice::Auto);

        let mut steps = 0;
        let mut tool_calls_made = Vec::new();

        // Agent loop
        loop {
            if steps >= self.config.max_steps {
                return WorkerResult {
                    task_id: task.id,
                    success: false,
                    output: format!("Task did not complete within {} steps", self.config.max_steps),
                    tool_calls: tool_calls_made,
                    duration_ms: start.elapsed().as_millis() as u64,
                };
            }

            steps += 1;

            if let Some(pb) = progress {
                pb.set_message(format!("Step {}: {}", steps, task.description));
            }

            // Get model response
            let response = match self.model.send_chat_request(messages.clone()).await {
                Ok(r) => r,
                Err(e) => {
                    return WorkerResult {
                        task_id: task.id,
                        success: false,
                        output: format!("Model error: {}", e),
                        tool_calls: tool_calls_made,
                        duration_ms: start.elapsed().as_millis() as u64,
                    };
                }
            };

            let choice = &response.choices[0];
            let message = &choice.message;

            // Check for tool calls
            if let Some(tool_calls) = &message.tool_calls {
                for tc in tool_calls {
                    if let Some(pb) = progress {
                        pb.set_message(format!("Tool: {}", tc.function.name));
                    }

                    let call = ToolCall {
                        id: tc.id.clone(),
                        name: tc.function.name.clone(),
                        arguments: serde_json::from_str(&tc.function.arguments)
                            .unwrap_or(serde_json::Value::Object(Default::default())),
                    };

                    let result = match self.tools.execute(&call).await {
                        Ok(r) => r,
                        Err(e) => {
                            mistralrs_swarm::tools::ToolResult::error(format!("Tool error: {}", e))
                        }
                    };

                    tool_calls_made.push((tc.function.name.clone(), result.success));

                    // Add to conversation
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
                continue;
            }

            // No tool calls - check for completion
            if let Some(content) = &message.content {
                if !content.trim().is_empty() {
                    if let Some(pb) = progress {
                        pb.finish_with_message(format!("Done: {}", task.description));
                    }

                    return WorkerResult {
                        task_id: task.id,
                        success: true,
                        output: content.clone(),
                        tool_calls: tool_calls_made,
                        duration_ms: start.elapsed().as_millis() as u64,
                    };
                }
            }

            // Check stop reason
            if choice.finish_reason == "stop" {
                let output = message.content.clone().unwrap_or_else(|| "Task completed.".to_string());

                if let Some(pb) = progress {
                    pb.finish_with_message(format!("Done: {}", task.description));
                }

                return WorkerResult {
                    task_id: task.id,
                    success: true,
                    output,
                    tool_calls: tool_calls_made,
                    duration_ms: start.elapsed().as_millis() as u64,
                };
            }
        }
    }
}

/// Result from a worker execution
#[derive(Debug)]
pub struct WorkerResult {
    pub task_id: usize,
    pub success: bool,
    pub output: String,
    pub tool_calls: Vec<(String, bool)>, // (tool_name, success)
    pub duration_ms: u64,
}

/// Factory for creating and managing workers
pub struct WorkerFactory {
    model: Arc<mistralrs::Model>,
    tools: Arc<ToolRegistry>,
    num_workers: usize,
    config: WorkerConfig,
}

impl WorkerFactory {
    pub fn new(
        model: Arc<mistralrs::Model>,
        tools: ToolRegistry,
        num_workers: usize,
    ) -> Self {
        Self {
            model,
            tools: Arc::new(tools),
            num_workers,
            config: WorkerConfig::default(),
        }
    }

    pub fn with_config(mut self, config: WorkerConfig) -> Self {
        self.config = config;
        self
    }

    /// Create a new worker
    pub fn create_worker(&self, id: usize) -> Worker {
        Worker::new(
            id,
            self.model.clone(),
            self.tools.clone(),
            self.config.clone(),
        )
    }

    /// Execute all delegated tasks from a todo list
    pub async fn execute_todo_list(&self, todo_list: Arc<Mutex<TodoList>>) -> Vec<WorkerResult> {
        let multi_progress = MultiProgress::new();
        let style = ProgressStyle::default_spinner()
            .template("{spinner:.cyan} [{elapsed_precise}] {msg}")
            .unwrap();

        // Get delegated tasks
        let tasks: Vec<TodoItem> = {
            let list = todo_list.lock().await;
            list.pending_delegated().into_iter().cloned().collect()
        };

        if tasks.is_empty() {
            println!("{}", "No delegated tasks to execute.".dimmed());
            return Vec::new();
        }

        println!("\n{} {} tasks to {} workers",
            "Dispatching".yellow(),
            tasks.len(),
            self.num_workers
        );

        // Create progress bars and channels
        let (result_tx, mut result_rx) = mpsc::channel::<WorkerResult>(tasks.len());

        // Spawn workers
        let mut handles = Vec::new();

        for (i, task) in tasks.into_iter().enumerate() {
            let worker = self.create_worker(i % self.num_workers);
            let result_tx = result_tx.clone();
            let todo_list = todo_list.clone();

            let pb = multi_progress.add(ProgressBar::new_spinner());
            pb.set_style(style.clone());
            pb.set_message(format!("Queued: {}", task.description));

            // Mark as in progress
            {
                let mut list = todo_list.lock().await;
                list.start(task.id);
            }

            let handle = tokio::spawn(async move {
                let result = worker.execute(&task, Some(&pb)).await;

                // Update todo list
                {
                    let mut list = todo_list.lock().await;
                    if result.success {
                        list.complete(task.id, Some(result.output.clone()));
                    } else {
                        list.fail(task.id, result.output.clone());
                    }
                }

                let _ = result_tx.send(result).await;
            });

            handles.push(handle);
        }

        // Drop sender so receiver knows when all are done
        drop(result_tx);

        // Collect results
        let mut results = Vec::new();
        while let Some(result) = result_rx.recv().await {
            results.push(result);
        }

        // Wait for all handles
        for handle in handles {
            let _ = handle.await;
        }

        multi_progress.clear().ok();

        results
    }
}
