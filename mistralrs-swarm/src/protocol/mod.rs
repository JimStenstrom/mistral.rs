//! Protocol definitions for communication between orchestrator and workers

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A task to be executed by a worker agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    /// Unique task identifier
    pub id: String,
    /// Human-readable task description
    pub description: String,
    /// Detailed instructions for the worker
    pub instructions: String,
    /// Task priority (higher = more urgent)
    pub priority: i32,
    /// Expected complexity (affects timeout and resource allocation)
    pub complexity: TaskComplexity,
    /// Dependencies on other task IDs (must complete first)
    pub dependencies: Vec<String>,
    /// Context from orchestrator (relevant information, constraints)
    pub context: TaskContext,
    /// Maximum steps the worker should take
    pub max_steps: usize,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
}

impl Task {
    pub fn new(description: impl Into<String>, instructions: impl Into<String>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            description: description.into(),
            instructions: instructions.into(),
            priority: 0,
            complexity: TaskComplexity::Medium,
            dependencies: Vec::new(),
            context: TaskContext::default(),
            max_steps: 10,
            created_at: Utc::now(),
        }
    }

    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    pub fn with_complexity(mut self, complexity: TaskComplexity) -> Self {
        self.complexity = complexity;
        self
    }

    pub fn with_dependencies(mut self, deps: Vec<String>) -> Self {
        self.dependencies = deps;
        self
    }

    pub fn with_context(mut self, context: TaskContext) -> Self {
        self.context = context;
        self
    }

    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }
}

/// Task complexity level
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TaskComplexity {
    /// Simple task, few steps expected
    Simple,
    /// Medium complexity, typical task
    #[default]
    Medium,
    /// Complex task, many steps or reasoning required
    Complex,
    /// Very complex, may need multiple attempts
    VeryComplex,
}

impl TaskComplexity {
    /// Get suggested timeout in seconds
    pub fn timeout_secs(&self) -> u64 {
        match self {
            TaskComplexity::Simple => 30,
            TaskComplexity::Medium => 120,
            TaskComplexity::Complex => 300,
            TaskComplexity::VeryComplex => 600,
        }
    }

    /// Get suggested max steps
    pub fn max_steps(&self) -> usize {
        match self {
            TaskComplexity::Simple => 5,
            TaskComplexity::Medium => 10,
            TaskComplexity::Complex => 20,
            TaskComplexity::VeryComplex => 50,
        }
    }
}

/// Context provided by orchestrator to help worker
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TaskContext {
    /// Relevant file paths
    pub relevant_files: Vec<String>,
    /// Key information from other tasks
    pub shared_knowledge: HashMap<String, String>,
    /// Constraints the worker must follow
    pub constraints: Vec<String>,
    /// Success criteria
    pub success_criteria: Vec<String>,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Result from a completed task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    /// The task ID this result belongs to
    pub task_id: String,
    /// Final status
    pub status: TaskStatus,
    /// The output/answer from the worker
    pub output: String,
    /// Artifacts produced (files created, etc.)
    pub artifacts: Vec<Artifact>,
    /// Tool calls made during execution
    pub tool_calls: Vec<ToolCallRecord>,
    /// Number of reasoning steps taken
    pub steps_taken: usize,
    /// Total tokens used
    pub tokens_used: TokenUsage,
    /// Execution duration in milliseconds
    pub duration_ms: u64,
    /// Any errors encountered
    pub errors: Vec<String>,
    /// Timestamp of completion
    pub completed_at: DateTime<Utc>,
}

impl TaskResult {
    pub fn success(task_id: String, output: String) -> Self {
        Self {
            task_id,
            status: TaskStatus::Completed,
            output,
            artifacts: Vec::new(),
            tool_calls: Vec::new(),
            steps_taken: 0,
            tokens_used: TokenUsage::default(),
            duration_ms: 0,
            errors: Vec::new(),
            completed_at: Utc::now(),
        }
    }

    pub fn failure(task_id: String, error: String) -> Self {
        Self {
            task_id,
            status: TaskStatus::Failed,
            output: String::new(),
            artifacts: Vec::new(),
            tool_calls: Vec::new(),
            steps_taken: 0,
            tokens_used: TokenUsage::default(),
            duration_ms: 0,
            errors: vec![error],
            completed_at: Utc::now(),
        }
    }
}

/// Status of a task
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    /// Waiting to be picked up
    Pending,
    /// Currently being executed
    InProgress,
    /// Successfully completed
    Completed,
    /// Failed to complete
    Failed,
    /// Cancelled by orchestrator
    Cancelled,
    /// Timed out
    TimedOut,
}

/// An artifact produced by a task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Artifact {
    /// Artifact type
    pub kind: ArtifactKind,
    /// Name/path
    pub name: String,
    /// Content (may be truncated for large files)
    pub content: Option<String>,
    /// Size in bytes
    pub size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    File,
    CodeChange,
    TestResult,
    Log,
    Report,
    Other(String),
}

/// Record of a tool call made during task execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallRecord {
    /// Tool name
    pub tool: String,
    /// Arguments passed
    pub arguments: serde_json::Value,
    /// Result returned
    pub result: String,
    /// Duration in milliseconds
    pub duration_ms: u64,
    /// Whether it succeeded
    pub success: bool,
}

/// Token usage statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// Messages between orchestrator and workers
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WorkerMessage {
    /// New task assignment
    TaskAssigned(Task),
    /// Progress update from worker
    Progress {
        task_id: String,
        step: usize,
        message: String,
    },
    /// Task completed
    TaskCompleted(TaskResult),
    /// Worker requesting help/clarification
    RequestClarification {
        task_id: String,
        question: String,
    },
    /// Orchestrator providing clarification
    Clarification {
        task_id: String,
        answer: String,
    },
    /// Cancel a task
    CancelTask { task_id: String },
    /// Worker heartbeat
    Heartbeat { worker_id: String },
    /// Shutdown signal
    Shutdown,
}
