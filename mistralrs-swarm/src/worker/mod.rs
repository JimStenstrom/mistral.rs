//! Worker agents using mistral.rs for local inference
//!
//! Workers run autonomous agent loops, using the local LLM for reasoning
//! and the tool registry for actions.

mod agent_loop;

pub use agent_loop::AgentLoop;

use crate::protocol::{Task, TaskResult, TaskStatus, TokenUsage};
use crate::tools::ToolRegistry;
use anyhow::Result;
use mistralrs::{
    IsqType, MemoryGpuConfig, PagedAttentionMetaBuilder, TextModelBuilder,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{info, warn};

/// Configuration for worker agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerConfig {
    /// Model ID to load (HuggingFace format)
    pub model_id: String,
    /// Number of worker agents to spawn
    #[serde(default = "default_num_workers")]
    pub num_workers: usize,
    /// Quantization level (e.g., "Q4K", "Q8_0")
    pub quantization: Option<String>,
    /// GPU memory to allocate for KV cache (in MB)
    #[serde(default = "default_gpu_memory")]
    pub gpu_memory_mb: usize,
    /// Maximum steps per task
    #[serde(default = "default_max_steps")]
    pub max_steps: usize,
    /// System prompt for workers
    #[serde(default = "default_system_prompt")]
    pub system_prompt: String,
}

fn default_num_workers() -> usize {
    8
}

fn default_gpu_memory() -> usize {
    16000 // 16GB default
}

fn default_max_steps() -> usize {
    15
}

fn default_system_prompt() -> String {
    r#"You are an autonomous worker agent in a swarm system. Your task is to complete assignments given to you by the orchestrator.

## Guidelines

1. ANALYZE the task carefully before acting
2. Use TOOLS when needed to gather information or make changes
3. THINK step by step through complex problems
4. REPORT clearly what you accomplished or why you couldn't complete the task

## Available Actions

You can:
- Call tools using the provided tool interface
- Ask for clarification if the task is ambiguous
- Report completion with your findings/output
- Report failure if the task cannot be completed

## Tool Calling

When you need to use a tool, respond with a tool call in this format. The system will execute the tool and provide the result.

Be efficient - don't call tools unnecessarily. Think about what information you need before making calls.

## Completion

When the task is complete, provide a clear summary of:
- What you accomplished
- Any files created/modified
- Any issues encountered
- The final answer/output"#
        .to_string()
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            model_id: "Qwen/Qwen3-4B".to_string(),
            num_workers: default_num_workers(),
            quantization: Some("Q4K".to_string()),
            gpu_memory_mb: default_gpu_memory(),
            max_steps: default_max_steps(),
            system_prompt: default_system_prompt(),
        }
    }
}

/// A single worker agent
pub struct Worker {
    id: usize,
    config: WorkerConfig,
    model: Arc<mistralrs::Model>,
    tools: Arc<ToolRegistry>,
}

impl Worker {
    /// Create a new worker (shares model with pool)
    pub fn new(
        id: usize,
        config: WorkerConfig,
        model: Arc<mistralrs::Model>,
        tools: Arc<ToolRegistry>,
    ) -> Self {
        Self {
            id,
            config,
            model,
            tools,
        }
    }

    /// Execute a task
    pub async fn execute(&self, task: Task) -> TaskResult {
        let start = std::time::Instant::now();
        info!(worker = self.id, task_id = %task.id, "Starting task: {}", task.description);

        let agent_loop = AgentLoop::new(
            self.model.clone(),
            self.tools.clone(),
            &self.config.system_prompt,
            task.max_steps.min(self.config.max_steps),
        );

        match agent_loop.run(&task).await {
            Ok((output, tool_calls, tokens)) => {
                info!(worker = self.id, task_id = %task.id, "Task completed successfully");
                let steps_taken = tool_calls.len();
                TaskResult {
                    task_id: task.id,
                    status: TaskStatus::Completed,
                    output,
                    artifacts: Vec::new(),
                    tool_calls,
                    steps_taken,
                    tokens_used: tokens,
                    duration_ms: start.elapsed().as_millis() as u64,
                    errors: Vec::new(),
                    completed_at: chrono::Utc::now(),
                }
            }
            Err(e) => {
                warn!(worker = self.id, task_id = %task.id, "Task failed: {}", e);
                TaskResult {
                    task_id: task.id,
                    status: TaskStatus::Failed,
                    output: String::new(),
                    artifacts: Vec::new(),
                    tool_calls: Vec::new(),
                    steps_taken: 0,
                    tokens_used: TokenUsage::default(),
                    duration_ms: start.elapsed().as_millis() as u64,
                    errors: vec![e.to_string()],
                    completed_at: chrono::Utc::now(),
                }
            }
        }
    }
}

/// Pool of worker agents sharing a single model
pub struct WorkerPool {
    workers: Vec<Arc<Worker>>,
    model: Arc<mistralrs::Model>,
    config: WorkerConfig,
    /// Handle to the worker tasks
    _handles: Vec<tokio::task::JoinHandle<()>>,
}

impl WorkerPool {
    /// Create a new worker pool
    pub async fn new(
        config: WorkerConfig,
        tools: ToolRegistry,
        task_rx: mpsc::Receiver<Task>,
        result_tx: mpsc::Sender<TaskResult>,
    ) -> Result<Self> {
        info!(
            "Initializing worker pool with {} workers, model: {}",
            config.num_workers, config.model_id
        );

        // Build the shared model
        let model = Self::build_model(&config).await?;
        let model = Arc::new(model);
        let tools = Arc::new(tools);

        // Create workers
        let mut workers = Vec::with_capacity(config.num_workers);
        for i in 0..config.num_workers {
            workers.push(Arc::new(Worker::new(
                i,
                config.clone(),
                model.clone(),
                tools.clone(),
            )));
        }

        // Spawn worker tasks that pull from the task queue
        let mut handles = Vec::new();
        let task_rx = Arc::new(tokio::sync::Mutex::new(task_rx));

        for worker in &workers {
            let worker = worker.clone();
            let result_tx = result_tx.clone();
            let task_rx = task_rx.clone();

            let handle = tokio::spawn(async move {
                loop {
                    // Get next task from queue
                    let task = {
                        let mut rx = task_rx.lock().await;
                        rx.recv().await
                    };

                    match task {
                        Some(task) => {
                            let result = worker.execute(task).await;
                            if result_tx.send(result).await.is_err() {
                                break; // Result channel closed
                            }
                        }
                        None => break, // Task channel closed
                    }
                }
            });
            handles.push(handle);
        }

        info!("Worker pool initialized with {} workers", workers.len());

        Ok(Self {
            workers,
            model,
            config,
            _handles: handles,
        })
    }

    /// Build the mistral.rs model
    async fn build_model(config: &WorkerConfig) -> Result<mistralrs::Model> {
        let mut builder = TextModelBuilder::new(&config.model_id).with_logging();

        // Apply quantization
        if let Some(ref quant) = config.quantization {
            let isq = match quant.to_uppercase().as_str() {
                "Q2K" => IsqType::Q2K,
                "Q3K" => IsqType::Q3K,
                "Q4K" => IsqType::Q4K,
                "Q4_0" => IsqType::Q4_0,
                "Q4_1" => IsqType::Q4_1,
                "Q5K" => IsqType::Q5K,
                "Q5_0" => IsqType::Q5_0,
                "Q5_1" => IsqType::Q5_1,
                "Q6K" => IsqType::Q6K,
                "Q8_0" => IsqType::Q8_0,
                "Q8_1" => IsqType::Q8_1,
                _ => {
                    warn!("Unknown quantization '{}', defaulting to Q4K", quant);
                    IsqType::Q4K
                }
            };
            builder = builder.with_isq(isq);
        }

        // Configure paged attention for efficient batching
        builder = builder.with_paged_attn(|| {
            PagedAttentionMetaBuilder::default()
                .with_gpu_memory(MemoryGpuConfig::MbAmount(config.gpu_memory_mb))
                .with_block_size(32)
                .build()
        })?;

        let model = builder.build().await?;
        Ok(model)
    }

    /// Get number of workers
    pub fn num_workers(&self) -> usize {
        self.workers.len()
    }

    /// Get the shared model
    pub fn model(&self) -> Arc<mistralrs::Model> {
        self.model.clone()
    }
}
