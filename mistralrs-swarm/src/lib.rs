//! # mistralrs-swarm
//!
//! A high-throughput agentic swarm system that uses Claude (Opus 4.5) as an orchestrator
//! and mistral.rs for local LLM inference on worker agents.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    CLAUDE ORCHESTRATOR                          │
//! │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
//! │  │  Planner │→ │Decomposer│→ │Dispatcher│→ │Synthestic│        │
//! │  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
//! └─────────────────────────┬───────────────────────────────────────┘
//!                           │ Task Queue (async)
//!           ┌───────────────┼───────────────┐
//!           ▼               ▼               ▼
//!     ┌──────────┐    ┌──────────┐    ┌──────────┐
//!     │ Worker 1 │    │ Worker 2 │    │ Worker N │  (32+ workers)
//!     │  Agent   │    │  Agent   │    │  Agent   │
//!     └────┬─────┘    └────┬─────┘    └────┬─────┘
//!          │               │               │
//!          └───────────────┼───────────────┘
//!                          ▼
//!               ┌─────────────────────┐
//!               │   mistral.rs Engine │
//!               │  (Local Inference)  │
//!               │  Continuous Batch   │
//!               └─────────────────────┘
//!                          │
//!               ┌─────────────────────┐
//!               │   Learning System   │
//!               │  (Pattern Tracking) │
//!               └─────────────────────┘
//! ```
//!
//! ## Key Features
//!
//! - **Claude Orchestrator**: Uses Claude Opus 4.5 for high-level planning and task decomposition
//! - **Local Worker Agents**: Runs on mistral.rs with automatic batching for GPU saturation
//! - **Tool System**: Extensible tool registry for file operations, code execution, web search
//! - **Learning System**: Tracks what works locally, suggests task graduation
//! - **Async-first**: Built on Tokio for maximum concurrency
//!
//! ## Example
//!
//! ```rust,ignore
//! use mistralrs_swarm::{Swarm, SwarmConfig, ClaudeConfig, WorkerConfig};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let swarm = Swarm::builder()
//!         .with_claude_orchestrator(ClaudeConfig {
//!             api_key: std::env::var("ANTHROPIC_API_KEY")?,
//!             model: "claude-opus-4-5-20250929".to_string(),
//!         })
//!         .with_workers(WorkerConfig {
//!             model_id: "Qwen/Qwen3-8B".to_string(),
//!             num_workers: 32,
//!             quantization: Some("Q4K".to_string()),
//!         })
//!         .with_learning("./learning_data")  // Enable learning
//!         .build()
//!         .await?;
//!
//!     let result = swarm.execute("Analyze this codebase and find security issues").await?;
//!     println!("{}", result);
//!
//!     // Show what you've learned
//!     println!("{}", swarm.insights());
//!     Ok(())
//! }
//! ```

pub mod git_sandbox;
pub mod learning;
pub mod orchestrator;
pub mod protocol;
pub mod tools;
pub mod worker;
pub mod workflow;

// Re-exports
pub use learning::{
    ExecutionRecord, GraduationCandidate, Insights, LearningRecorder,
    LearningSystem, Pattern, PatternStore, TaskTemplate, TemplateStore,
};
pub use orchestrator::{ClaudeConfig, ClaudeOrchestrator, Orchestrator};
pub use protocol::{Task, TaskResult, TaskStatus, WorkerMessage};
pub use tools::{Tool, ToolCall, ToolRegistry, ToolResult};
pub use worker::{Worker, WorkerConfig, WorkerPool};

use anyhow::Result;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};

/// Main swarm coordinator that ties orchestrator and workers together
pub struct Swarm {
    orchestrator: Arc<ClaudeOrchestrator>,
    worker_pool: Arc<WorkerPool>,
    task_tx: mpsc::Sender<Task>,
    result_rx: mpsc::Receiver<TaskResult>,
    learning: Option<Arc<Mutex<LearningSystem>>>,
}

impl Swarm {
    /// Create a new swarm builder
    pub fn builder() -> SwarmBuilder {
        SwarmBuilder::default()
    }

    /// Execute a high-level goal using the swarm
    pub async fn execute(&mut self, goal: &str) -> Result<String> {
        // Start recording if learning is enabled
        let mut record = learning::ExecutionRecord::new(goal);

        // 1. Orchestrator plans and decomposes the goal
        let tasks = self.orchestrator.plan_and_decompose(goal).await?;

        // Record decomposition
        record.decomposition = Some(learning::DecompositionRecord {
            task_count: tasks.len(),
            local_tasks: tasks.len(), // All go to local workers
            claude_tasks: 0,
            reasoning: None,
            decomposition_time_ms: 0,
        });

        // 2. Dispatch tasks to worker pool
        for task in tasks {
            self.task_tx.send(task).await?;
        }

        // 3. Collect results
        let mut results = Vec::new();
        while let Ok(result) = self.result_rx.try_recv() {
            // Record task execution
            let task_record = learning::TaskExecutionRecord::new(&result.task_id, "")
                .local()
                .with_outcome(if result.status == TaskStatus::Completed {
                    learning::TaskOutcome::LocalSuccess {
                        worker_steps: result.steps_taken,
                        tools_used: result.tool_calls.iter().map(|t| t.tool.clone()).collect(),
                        duration_ms: result.duration_ms,
                    }
                } else {
                    learning::TaskOutcome::LocalFailure {
                        error: result.errors.join(", "),
                        partial_progress: Some(result.output.clone()),
                    }
                })
                .with_duration(result.duration_ms);

            record.add_task_execution(task_record);
            results.push(result);
        }

        // 4. Orchestrator synthesizes final result
        let synthesis = self.orchestrator.synthesize(&results).await?;

        // Record outcome
        let _local_successes = results.iter().filter(|r| r.status == TaskStatus::Completed).count();
        record.complete(learning::OverallOutcome::Success {
            summary: synthesis.clone(),
            local_percentage: 100.0,
        });

        // Save to learning system
        if let Some(ref learning) = self.learning {
            let mut learning = learning.lock().await;
            learning.record(record)?;
        }

        Ok(synthesis)
    }

    /// Execute with streaming updates
    pub async fn execute_streaming(
        &mut self,
        goal: &str,
        update_tx: mpsc::Sender<SwarmUpdate>,
    ) -> Result<String> {
        // Send planning update
        update_tx
            .send(SwarmUpdate::Planning(goal.to_string()))
            .await?;

        // Plan and decompose
        let tasks = self.orchestrator.plan_and_decompose(goal).await?;
        update_tx
            .send(SwarmUpdate::TasksCreated(tasks.len()))
            .await?;

        // Dispatch and track
        for task in tasks {
            update_tx
                .send(SwarmUpdate::TaskDispatched(task.id.clone()))
                .await?;
            self.task_tx.send(task).await?;
        }

        // Collect with updates
        let mut results = Vec::new();
        while let Some(result) = self.result_rx.recv().await {
            update_tx
                .send(SwarmUpdate::TaskCompleted(result.task_id.clone()))
                .await?;
            results.push(result);
        }

        // Synthesize
        update_tx.send(SwarmUpdate::Synthesizing).await?;
        let synthesis = self.orchestrator.synthesize(&results).await?;

        Ok(synthesis)
    }

    /// Get insights from the learning system
    pub async fn insights(&self) -> Option<Insights> {
        if let Some(ref learning) = self.learning {
            let learning = learning.lock().await;
            Some(learning.insights())
        } else {
            None
        }
    }

    /// Get tasks that might be ready for local-only execution
    pub async fn graduation_candidates(&self) -> Vec<GraduationCandidate> {
        if let Some(ref learning) = self.learning {
            let learning = learning.lock().await;
            learning.graduation_candidates()
        } else {
            Vec::new()
        }
    }

    /// Find a template for a task description
    pub async fn find_template(&self, description: &str) -> Option<TaskTemplate> {
        if let Some(ref learning) = self.learning {
            let learning = learning.lock().await;
            learning.find_template(description).cloned()
        } else {
            None
        }
    }
}

/// Updates from the swarm during execution
#[derive(Debug, Clone)]
pub enum SwarmUpdate {
    Planning(String),
    TasksCreated(usize),
    TaskDispatched(String),
    TaskInProgress { task_id: String, step: usize },
    TaskCompleted(String),
    Synthesizing,
    Complete,
    Error(String),
}

/// Builder for configuring a Swarm
#[derive(Default)]
pub struct SwarmBuilder {
    claude_config: Option<ClaudeConfig>,
    worker_config: Option<WorkerConfig>,
    tool_registry: Option<ToolRegistry>,
    learning_path: Option<PathBuf>,
}

impl SwarmBuilder {
    pub fn with_claude_orchestrator(mut self, config: ClaudeConfig) -> Self {
        self.claude_config = Some(config);
        self
    }

    pub fn with_workers(mut self, config: WorkerConfig) -> Self {
        self.worker_config = Some(config);
        self
    }

    pub fn with_tools(mut self, registry: ToolRegistry) -> Self {
        self.tool_registry = Some(registry);
        self
    }

    /// Enable learning system with storage at the given path
    pub fn with_learning(mut self, path: impl Into<PathBuf>) -> Self {
        self.learning_path = Some(path.into());
        self
    }

    pub async fn build(self) -> Result<Swarm> {
        let claude_config = self
            .claude_config
            .ok_or_else(|| anyhow::anyhow!("Claude config is required"))?;
        let worker_config = self
            .worker_config
            .ok_or_else(|| anyhow::anyhow!("Worker config is required"))?;
        let tool_registry = self.tool_registry.unwrap_or_default();

        // Create channels
        let (task_tx, task_rx) = mpsc::channel(1000);
        let (result_tx, result_rx) = mpsc::channel(1000);

        // Build orchestrator
        let orchestrator = Arc::new(ClaudeOrchestrator::new(claude_config).await?);

        // Build worker pool
        let worker_pool = Arc::new(
            WorkerPool::new(worker_config, tool_registry, task_rx, result_tx).await?,
        );

        // Build learning system if enabled
        let learning = if let Some(path) = self.learning_path {
            Some(Arc::new(Mutex::new(LearningSystem::new(path)?)))
        } else {
            None
        };

        Ok(Swarm {
            orchestrator,
            worker_pool,
            task_tx,
            result_rx,
            learning,
        })
    }
}
