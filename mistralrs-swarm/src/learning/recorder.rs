//! Execution recorder - captures every task execution for learning

use super::{ComplexityLevel, TaskOutcome};
use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// A record of a single task execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionRecord {
    /// Unique ID for this execution
    pub id: String,
    /// When this was executed
    pub timestamp: DateTime<Utc>,
    /// Original goal/request from user
    pub original_goal: String,
    /// How Claude decomposed it (if applicable)
    pub decomposition: Option<DecompositionRecord>,
    /// Individual task executions
    pub task_executions: Vec<TaskExecutionRecord>,
    /// Overall outcome
    pub outcome: OverallOutcome,
    /// Tags for categorization
    pub tags: Vec<String>,
    /// User notes (can be added later)
    pub notes: Option<String>,
}

impl ExecutionRecord {
    pub fn new(original_goal: impl Into<String>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            original_goal: original_goal.into(),
            decomposition: None,
            task_executions: Vec::new(),
            outcome: OverallOutcome::InProgress,
            tags: Vec::new(),
            notes: None,
        }
    }

    pub fn with_decomposition(mut self, decomp: DecompositionRecord) -> Self {
        self.decomposition = Some(decomp);
        self
    }

    pub fn add_task_execution(&mut self, exec: TaskExecutionRecord) {
        self.task_executions.push(exec);
    }

    pub fn complete(&mut self, outcome: OverallOutcome) {
        self.outcome = outcome;
    }

    pub fn add_tag(&mut self, tag: impl Into<String>) {
        self.tags.push(tag.into());
    }

    /// Calculate success rate of local tasks
    pub fn local_success_rate(&self) -> f32 {
        let local_tasks: Vec<_> = self.task_executions.iter()
            .filter(|t| t.executed_locally)
            .collect();

        if local_tasks.is_empty() {
            return 0.0;
        }

        let successes = local_tasks.iter()
            .filter(|t| t.outcome.is_local_success())
            .count();

        successes as f32 / local_tasks.len() as f32
    }

    /// Get task types that succeeded locally
    pub fn successful_local_patterns(&self) -> Vec<String> {
        self.task_executions.iter()
            .filter(|t| t.executed_locally && t.outcome.is_local_success())
            .map(|t| t.task_type.clone())
            .collect()
    }

    /// Get task types that failed locally
    pub fn failed_local_patterns(&self) -> Vec<String> {
        self.task_executions.iter()
            .filter(|t| t.executed_locally && t.outcome.is_local_failure())
            .map(|t| t.task_type.clone())
            .collect()
    }
}

/// Record of how Claude decomposed a goal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecompositionRecord {
    /// Number of tasks created
    pub task_count: usize,
    /// Tasks that were marked for local execution
    pub local_tasks: usize,
    /// Tasks Claude kept for itself
    pub claude_tasks: usize,
    /// The reasoning Claude provided
    pub reasoning: Option<String>,
    /// Time taken to decompose (ms)
    pub decomposition_time_ms: u64,
}

/// Record of a single task execution within a larger goal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskExecutionRecord {
    /// Task ID
    pub task_id: String,
    /// Short description
    pub description: String,
    /// Categorized task type (e.g., "file_read", "code_refactor", "search")
    pub task_type: String,
    /// Was this executed locally or by Claude?
    pub executed_locally: bool,
    /// The outcome
    pub outcome: TaskOutcome,
    /// Complexity assessment
    pub complexity: ComplexityLevel,
    /// Tools that were used
    pub tools_used: Vec<String>,
    /// Time taken (ms)
    pub duration_ms: u64,
    /// Number of LLM calls made
    pub llm_calls: usize,
}

impl TaskExecutionRecord {
    pub fn new(task_id: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            task_id: task_id.into(),
            description: description.into(),
            task_type: "unknown".to_string(),
            executed_locally: false,
            outcome: TaskOutcome::LocalFailure {
                error: "Not executed".to_string(),
                partial_progress: None
            },
            complexity: ComplexityLevel::Medium,
            tools_used: Vec::new(),
            duration_ms: 0,
            llm_calls: 0,
        }
    }

    pub fn local(mut self) -> Self {
        self.executed_locally = true;
        self
    }

    pub fn with_type(mut self, task_type: impl Into<String>) -> Self {
        self.task_type = task_type.into();
        self
    }

    pub fn with_outcome(mut self, outcome: TaskOutcome) -> Self {
        self.outcome = outcome;
        self
    }

    pub fn with_complexity(mut self, complexity: ComplexityLevel) -> Self {
        self.complexity = complexity;
        self
    }

    pub fn with_tools(mut self, tools: Vec<String>) -> Self {
        self.tools_used = tools;
        self
    }

    pub fn with_duration(mut self, duration_ms: u64) -> Self {
        self.duration_ms = duration_ms;
        self
    }

    pub fn with_llm_calls(mut self, calls: usize) -> Self {
        self.llm_calls = calls;
        self
    }
}

/// Overall outcome of a goal execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OverallOutcome {
    /// Still in progress
    InProgress,
    /// Completed successfully
    Success {
        /// Summary of what was accomplished
        summary: String,
        /// Percentage done by local models
        local_percentage: f32,
    },
    /// Partially completed
    PartialSuccess {
        completed_tasks: usize,
        failed_tasks: usize,
        summary: String,
    },
    /// Failed
    Failure {
        reason: String,
    },
    /// Cancelled by user
    Cancelled,
}

/// Manages recording and storage of execution records
pub struct LearningRecorder {
    records: Vec<ExecutionRecord>,
    storage_path: PathBuf,
}

impl LearningRecorder {
    pub fn load_or_create(path: PathBuf) -> Result<Self> {
        let records = if path.exists() {
            let data = std::fs::read_to_string(&path)?;
            serde_json::from_str(&data).unwrap_or_default()
        } else {
            Vec::new()
        };

        Ok(Self {
            records,
            storage_path: path,
        })
    }

    pub fn add(&mut self, record: ExecutionRecord) -> Result<()> {
        self.records.push(record);
        Ok(())
    }

    pub fn save(&self) -> Result<()> {
        let data = serde_json::to_string_pretty(&self.records)?;
        std::fs::write(&self.storage_path, data)?;
        Ok(())
    }

    pub fn records(&self) -> &[ExecutionRecord] {
        &self.records
    }

    /// Get records from the last N days
    pub fn recent(&self, days: i64) -> Vec<&ExecutionRecord> {
        let cutoff = Utc::now() - chrono::Duration::days(days);
        self.records.iter()
            .filter(|r| r.timestamp > cutoff)
            .collect()
    }

    /// Get records by tag
    pub fn by_tag(&self, tag: &str) -> Vec<&ExecutionRecord> {
        self.records.iter()
            .filter(|r| r.tags.iter().any(|t| t == tag))
            .collect()
    }

    /// Count total executions
    pub fn total_executions(&self) -> usize {
        self.records.len()
    }

    /// Count successful executions
    pub fn successful_executions(&self) -> usize {
        self.records.iter()
            .filter(|r| matches!(r.outcome, OverallOutcome::Success { .. }))
            .count()
    }
}
