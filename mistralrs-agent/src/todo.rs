//! Todo list management for tracking and planning tasks

use colored::Colorize;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Status of a todo item
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TodoStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
}

impl fmt::Display for TodoStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TodoStatus::Pending => write!(f, "{}", "○".dimmed()),
            TodoStatus::InProgress => write!(f, "{}", "◐".yellow()),
            TodoStatus::Completed => write!(f, "{}", "●".green()),
            TodoStatus::Failed => write!(f, "{}", "✗".red()),
        }
    }
}

/// A single todo item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TodoItem {
    pub id: usize,
    pub description: String,
    pub details: Option<String>,
    pub status: TodoStatus,
    /// If true, this task should be run by a worker agent
    pub delegate_to_worker: bool,
    /// Result after completion
    pub result: Option<String>,
}

impl TodoItem {
    pub fn new(id: usize, description: impl Into<String>) -> Self {
        Self {
            id,
            description: description.into(),
            details: None,
            status: TodoStatus::Pending,
            delegate_to_worker: false,
            result: None,
        }
    }

    pub fn with_details(mut self, details: impl Into<String>) -> Self {
        self.details = Some(details.into());
        self
    }

    pub fn delegated(mut self) -> Self {
        self.delegate_to_worker = true;
        self
    }

    pub fn mark_in_progress(&mut self) {
        self.status = TodoStatus::InProgress;
    }

    pub fn mark_completed(&mut self, result: Option<String>) {
        self.status = TodoStatus::Completed;
        self.result = result;
    }

    pub fn mark_failed(&mut self, error: String) {
        self.status = TodoStatus::Failed;
        self.result = Some(error);
    }
}

impl fmt::Display for TodoItem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let worker_indicator = if self.delegate_to_worker {
            " [worker]".dimmed().to_string()
        } else {
            String::new()
        };

        write!(f, "{} {} {}{}",
            self.status,
            format!("[{}]", self.id).dimmed(),
            self.description,
            worker_indicator
        )?;

        if let Some(ref details) = self.details {
            write!(f, "\n     {}", details.dimmed())?;
        }

        Ok(())
    }
}

/// The todo list manager
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct TodoList {
    items: Vec<TodoItem>,
    next_id: usize,
}

impl TodoList {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a new todo item
    pub fn add(&mut self, description: impl Into<String>) -> usize {
        let id = self.next_id;
        self.items.push(TodoItem::new(id, description));
        self.next_id += 1;
        id
    }

    /// Add a todo item with details
    pub fn add_with_details(&mut self, description: impl Into<String>, details: impl Into<String>) -> usize {
        let id = self.next_id;
        self.items.push(TodoItem::new(id, description).with_details(details));
        self.next_id += 1;
        id
    }

    /// Add a delegated task (for worker execution)
    pub fn add_delegated(&mut self, description: impl Into<String>, details: impl Into<String>) -> usize {
        let id = self.next_id;
        self.items.push(
            TodoItem::new(id, description)
                .with_details(details)
                .delegated()
        );
        self.next_id += 1;
        id
    }

    /// Get a todo item by ID
    pub fn get(&self, id: usize) -> Option<&TodoItem> {
        self.items.iter().find(|item| item.id == id)
    }

    /// Get a mutable reference to a todo item by ID
    pub fn get_mut(&mut self, id: usize) -> Option<&mut TodoItem> {
        self.items.iter_mut().find(|item| item.id == id)
    }

    /// Mark a todo as in progress
    pub fn start(&mut self, id: usize) -> bool {
        if let Some(item) = self.get_mut(id) {
            item.mark_in_progress();
            true
        } else {
            false
        }
    }

    /// Mark a todo as completed
    pub fn complete(&mut self, id: usize, result: Option<String>) -> bool {
        if let Some(item) = self.get_mut(id) {
            item.mark_completed(result);
            true
        } else {
            false
        }
    }

    /// Mark a todo as failed
    pub fn fail(&mut self, id: usize, error: String) -> bool {
        if let Some(item) = self.get_mut(id) {
            item.mark_failed(error);
            true
        } else {
            false
        }
    }

    /// Get all pending items
    pub fn pending(&self) -> Vec<&TodoItem> {
        self.items
            .iter()
            .filter(|item| item.status == TodoStatus::Pending)
            .collect()
    }

    /// Get all delegated pending items
    pub fn pending_delegated(&self) -> Vec<&TodoItem> {
        self.items
            .iter()
            .filter(|item| item.status == TodoStatus::Pending && item.delegate_to_worker)
            .collect()
    }

    /// Get all items
    pub fn all(&self) -> &[TodoItem] {
        &self.items
    }

    /// Check if all items are completed or failed
    pub fn is_done(&self) -> bool {
        self.items.iter().all(|item| {
            matches!(item.status, TodoStatus::Completed | TodoStatus::Failed)
        })
    }

    /// Get progress stats
    pub fn stats(&self) -> TodoStats {
        let total = self.items.len();
        let completed = self.items.iter().filter(|i| i.status == TodoStatus::Completed).count();
        let failed = self.items.iter().filter(|i| i.status == TodoStatus::Failed).count();
        let in_progress = self.items.iter().filter(|i| i.status == TodoStatus::InProgress).count();
        let pending = self.items.iter().filter(|i| i.status == TodoStatus::Pending).count();

        TodoStats {
            total,
            completed,
            failed,
            in_progress,
            pending,
        }
    }

    /// Clear all items
    pub fn clear(&mut self) {
        self.items.clear();
        self.next_id = 0;
    }

    /// Remove completed items
    pub fn prune_completed(&mut self) {
        self.items.retain(|item| item.status != TodoStatus::Completed);
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Get count
    pub fn len(&self) -> usize {
        self.items.len()
    }
}

impl fmt::Display for TodoList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.items.is_empty() {
            writeln!(f, "{}", "No tasks in todo list.".dimmed())?;
            return Ok(());
        }

        let stats = self.stats();
        writeln!(f, "{}", "═══ Todo List ═══".bright_cyan())?;
        writeln!(f)?;

        for item in &self.items {
            writeln!(f, "  {}", item)?;
        }

        writeln!(f)?;
        writeln!(f, "{} {} total | {} completed | {} in progress | {} pending | {} failed",
            "Stats:".dimmed(),
            stats.total,
            stats.completed.to_string().green(),
            stats.in_progress.to_string().yellow(),
            stats.pending.to_string().dimmed(),
            stats.failed.to_string().red()
        )?;

        Ok(())
    }
}

/// Statistics about the todo list
#[derive(Debug, Clone)]
pub struct TodoStats {
    pub total: usize,
    pub completed: usize,
    pub failed: usize,
    pub in_progress: usize,
    pub pending: usize,
}
