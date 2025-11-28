//! Local AI planner for task decomposition
//!
//! Uses the local LLM to plan and decompose goals into todo items.

use crate::todo::TodoList;
use anyhow::Result;
use mistralrs::{RequestBuilder, TextMessageRole};
use std::sync::Arc;

/// System prompt for the planner
const PLANNER_SYSTEM_PROMPT: &str = r#"You are a task planning assistant. Your role is to break down complex goals into actionable tasks.

When given a goal, create a structured plan with specific, actionable tasks. For each task, determine if it:
1. Can be done by you directly (simple queries, analysis, explanations)
2. Should be delegated to a worker agent (file operations, code changes, searches, execution)

## Output Format

Respond with a JSON array of tasks:
```json
[
  {
    "description": "Short task title",
    "details": "Specific instructions for completing this task",
    "delegate": true/false
  }
]
```

## Guidelines

- Break complex goals into 3-8 specific tasks
- Tasks should be independent when possible (can run in parallel)
- Delegate tasks that require: file I/O, code execution, searches, or multi-step tool use
- Keep tasks you'll handle directly: analysis, explanations, simple reasoning
- Be specific in details - workers need clear instructions
- Order tasks logically if there are dependencies

Respond ONLY with the JSON array. No explanations."#;

/// Local AI planner that uses the model for task decomposition
pub struct Planner {
    model: Arc<mistralrs::Model>,
}

impl Planner {
    pub fn new(model: Arc<mistralrs::Model>) -> Self {
        Self { model }
    }

    /// Plan tasks for a given goal
    pub async fn plan(&self, goal: &str) -> Result<Vec<PlannedTask>> {
        let prompt = format!(
            "Create a task plan for this goal:\n\n{}",
            goal
        );

        let messages = RequestBuilder::new()
            .add_message(TextMessageRole::System, PLANNER_SYSTEM_PROMPT)
            .add_message(TextMessageRole::User, &prompt);

        let response = self.model.send_chat_request(messages).await?;
        let content = response.choices[0]
            .message
            .content
            .clone()
            .unwrap_or_default();

        // Parse JSON response
        let tasks = self.parse_plan(&content)?;
        Ok(tasks)
    }

    /// Plan and populate a todo list
    pub async fn plan_to_todo(&self, goal: &str, todo_list: &mut TodoList) -> Result<usize> {
        let tasks = self.plan(goal).await?;
        let count = tasks.len();

        for task in tasks {
            if task.delegate {
                todo_list.add_delegated(&task.description, &task.details);
            } else {
                todo_list.add_with_details(&task.description, &task.details);
            }
        }

        Ok(count)
    }

    /// Parse the plan JSON from model output
    fn parse_plan(&self, content: &str) -> Result<Vec<PlannedTask>> {
        // Try to find JSON array in the response
        let json_str = if let Some(start) = content.find('[') {
            if let Some(end) = content.rfind(']') {
                &content[start..=end]
            } else {
                content
            }
        } else {
            content
        };

        // Parse JSON
        let tasks: Vec<PlannedTask> = serde_json::from_str(json_str)
            .map_err(|e| anyhow::anyhow!("Failed to parse plan: {}\nContent: {}", e, content))?;

        Ok(tasks)
    }

    /// Quick single-turn response (no planning)
    pub async fn quick_response(&self, prompt: &str) -> Result<String> {
        let messages = RequestBuilder::new()
            .add_message(TextMessageRole::User, prompt);

        let response = self.model.send_chat_request(messages).await?;
        let content = response.choices[0]
            .message
            .content
            .clone()
            .unwrap_or_default();

        Ok(content)
    }

    /// Analyze context and determine if planning is needed
    pub async fn should_plan(&self, input: &str) -> Result<bool> {
        // Simple heuristics first
        let lower = input.to_lowercase();

        // Keywords that suggest planning is needed
        let planning_keywords = [
            "implement", "create", "build", "refactor", "fix", "analyze",
            "review", "update", "add", "remove", "change", "modify",
            "write", "generate", "design", "setup", "configure"
        ];

        // If it looks like a complex task, plan it
        if planning_keywords.iter().any(|kw| lower.contains(kw)) {
            return Ok(true);
        }

        // Short questions don't need planning
        if input.len() < 50 && input.contains('?') {
            return Ok(false);
        }

        // Ask the model if unsure
        let prompt = format!(
            r#"Does this request require breaking down into multiple steps/tasks, or is it a simple question that can be answered directly?

Request: {}

Respond with only "PLAN" or "DIRECT"."#,
            input
        );

        let response = self.quick_response(&prompt).await?;
        Ok(response.to_uppercase().contains("PLAN"))
    }
}

/// A planned task from the AI
#[derive(Debug, Clone, serde::Deserialize)]
pub struct PlannedTask {
    pub description: String,
    pub details: String,
    #[serde(default)]
    pub delegate: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_plan() {
        let json = r#"[
            {"description": "Task 1", "details": "Do thing 1", "delegate": true},
            {"description": "Task 2", "details": "Do thing 2", "delegate": false}
        ]"#;

        // Would need model for full test
    }
}
