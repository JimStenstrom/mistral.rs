//! Claude Orchestrator - Uses Claude Opus 4.5 for high-level planning and coordination

mod claude_client;

pub use claude_client::ClaudeClient;

use crate::protocol::{Task, TaskComplexity, TaskContext, TaskResult};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Configuration for Claude orchestrator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeConfig {
    /// Anthropic API key
    pub api_key: String,
    /// Model to use (default: claude-opus-4-5-20250929)
    #[serde(default = "default_model")]
    pub model: String,
    /// Maximum tokens for responses
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    /// API base URL (default: https://api.anthropic.com)
    #[serde(default = "default_base_url")]
    pub base_url: String,
}

fn default_model() -> String {
    "claude-opus-4-5-20250929".to_string()
}

fn default_max_tokens() -> usize {
    4096
}

fn default_base_url() -> String {
    "https://api.anthropic.com".to_string()
}

impl Default for ClaudeConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            model: default_model(),
            max_tokens: default_max_tokens(),
            base_url: default_base_url(),
        }
    }
}

/// Trait for orchestrator implementations
#[async_trait::async_trait]
pub trait Orchestrator: Send + Sync {
    /// Plan and decompose a high-level goal into tasks
    async fn plan_and_decompose(&self, goal: &str) -> Result<Vec<Task>>;

    /// Synthesize results from completed tasks into a final answer
    async fn synthesize(&self, results: &[TaskResult]) -> Result<String>;

    /// Handle a clarification request from a worker
    async fn handle_clarification(&self, task_id: &str, question: &str) -> Result<String>;

    /// Replan if a task fails
    async fn replan_on_failure(&self, failed_task: &Task, error: &str) -> Result<Vec<Task>>;
}

/// Claude-based orchestrator
pub struct ClaudeOrchestrator {
    client: Arc<ClaudeClient>,
    config: ClaudeConfig,
    system_prompt: String,
}

impl ClaudeOrchestrator {
    pub async fn new(config: ClaudeConfig) -> Result<Self> {
        let client = Arc::new(ClaudeClient::new(&config.api_key, &config.base_url)?);

        let system_prompt = r#"You are a master orchestrator for an agentic AI swarm system. Your role is to:

1. DECOMPOSE complex goals into independent, parallelizable tasks
2. DISPATCH tasks to worker agents who will execute them using local LLMs
3. SYNTHESIZE results from workers into coherent final outputs

## Task Decomposition Guidelines

When breaking down a goal:
- Create tasks that can run in PARALLEL where possible
- Each task should be SELF-CONTAINED with clear success criteria
- Tasks should be SPECIFIC enough for a worker to complete without clarification
- Include relevant CONTEXT (file paths, constraints, dependencies)
- Estimate COMPLEXITY accurately (simple, medium, complex, very_complex)

## Output Format

When decomposing tasks, respond with a JSON array of task objects:
```json
[
  {
    "description": "Short task description",
    "instructions": "Detailed step-by-step instructions for the worker",
    "priority": 0-10,
    "complexity": "simple|medium|complex|very_complex",
    "dependencies": ["task_id_1"],  // Optional: IDs of tasks that must complete first
    "context": {
      "relevant_files": ["path/to/file.rs"],
      "constraints": ["Do not modify X"],
      "success_criteria": ["File compiles", "Tests pass"]
    }
  }
]
```

## Synthesis Guidelines

When synthesizing results:
- Combine outputs logically
- Highlight key findings
- Note any failures or issues
- Provide actionable conclusions

Remember: Your workers are local LLMs (like Qwen 8B) - they're capable but not as powerful as you. Write clear, specific instructions."#.to_string();

        Ok(Self {
            client,
            config,
            system_prompt,
        })
    }
}

#[async_trait::async_trait]
impl Orchestrator for ClaudeOrchestrator {
    async fn plan_and_decompose(&self, goal: &str) -> Result<Vec<Task>> {
        let prompt = format!(
            r#"Decompose this goal into parallelizable tasks for worker agents:

GOAL: {}

Respond with ONLY a JSON array of tasks. No explanation, just the JSON."#,
            goal
        );

        let response = self
            .client
            .send_message(&self.config.model, &self.system_prompt, &prompt, self.config.max_tokens)
            .await?;

        // Parse JSON response
        let tasks_json: Vec<TaskJson> = serde_json::from_str(&response)
            .map_err(|e| anyhow::anyhow!("Failed to parse task decomposition: {}\nResponse: {}", e, response))?;

        // Convert to Task structs
        let tasks: Vec<Task> = tasks_json
            .into_iter()
            .map(|t| {
                Task::new(&t.description, &t.instructions)
                    .with_priority(t.priority.unwrap_or(0))
                    .with_complexity(match t.complexity.as_deref() {
                        Some("simple") => TaskComplexity::Simple,
                        Some("complex") => TaskComplexity::Complex,
                        Some("very_complex") => TaskComplexity::VeryComplex,
                        _ => TaskComplexity::Medium,
                    })
                    .with_context(TaskContext {
                        relevant_files: t.context.as_ref().map(|c| c.relevant_files.clone()).unwrap_or_default(),
                        constraints: t.context.as_ref().map(|c| c.constraints.clone()).unwrap_or_default(),
                        success_criteria: t.context.as_ref().map(|c| c.success_criteria.clone()).unwrap_or_default(),
                        ..Default::default()
                    })
            })
            .collect();

        Ok(tasks)
    }

    async fn synthesize(&self, results: &[TaskResult]) -> Result<String> {
        let results_summary: Vec<String> = results
            .iter()
            .map(|r| {
                format!(
                    "Task {}: {:?}\nOutput: {}\nErrors: {:?}",
                    r.task_id, r.status, r.output, r.errors
                )
            })
            .collect();

        let prompt = format!(
            r#"Synthesize these task results into a coherent final answer:

TASK RESULTS:
{}

Provide a clear, comprehensive synthesis that:
1. Summarizes key findings
2. Notes any failures or issues
3. Provides actionable conclusions"#,
            results_summary.join("\n\n---\n\n")
        );

        let response = self
            .client
            .send_message(&self.config.model, &self.system_prompt, &prompt, self.config.max_tokens)
            .await?;

        Ok(response)
    }

    async fn handle_clarification(&self, task_id: &str, question: &str) -> Result<String> {
        let prompt = format!(
            r#"A worker agent needs clarification for task {}:

QUESTION: {}

Provide a clear, specific answer that helps the worker proceed."#,
            task_id, question
        );

        let response = self
            .client
            .send_message(&self.config.model, &self.system_prompt, &prompt, self.config.max_tokens)
            .await?;

        Ok(response)
    }

    async fn replan_on_failure(&self, failed_task: &Task, error: &str) -> Result<Vec<Task>> {
        let prompt = format!(
            r#"A task failed and needs replanning:

FAILED TASK: {}
INSTRUCTIONS: {}
ERROR: {}

Either:
1. Create alternative tasks to achieve the same goal
2. Create simpler sub-tasks that might succeed
3. Return an empty array if the goal is not achievable

Respond with ONLY a JSON array of tasks."#,
            failed_task.description, failed_task.instructions, error
        );

        let response = self
            .client
            .send_message(&self.config.model, &self.system_prompt, &prompt, self.config.max_tokens)
            .await?;

        let tasks_json: Vec<TaskJson> = serde_json::from_str(&response).unwrap_or_default();

        let tasks: Vec<Task> = tasks_json
            .into_iter()
            .map(|t| {
                Task::new(&t.description, &t.instructions)
                    .with_priority(t.priority.unwrap_or(5)) // Higher priority for replanned tasks
                    .with_complexity(match t.complexity.as_deref() {
                        Some("simple") => TaskComplexity::Simple,
                        Some("complex") => TaskComplexity::Complex,
                        Some("very_complex") => TaskComplexity::VeryComplex,
                        _ => TaskComplexity::Medium,
                    })
            })
            .collect();

        Ok(tasks)
    }
}

/// JSON structure for task parsing
#[derive(Debug, Deserialize)]
struct TaskJson {
    description: String,
    instructions: String,
    priority: Option<i32>,
    complexity: Option<String>,
    dependencies: Option<Vec<String>>,
    context: Option<TaskContextJson>,
}

#[derive(Debug, Deserialize)]
struct TaskContextJson {
    #[serde(default)]
    relevant_files: Vec<String>,
    #[serde(default)]
    constraints: Vec<String>,
    #[serde(default)]
    success_criteria: Vec<String>,
}
