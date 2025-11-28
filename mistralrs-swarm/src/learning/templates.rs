//! Task templates - reusable patterns for common workflows

use super::ComplexityLevel;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// A template for a commonly executed task type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskTemplate {
    /// Template name
    pub name: String,
    /// Human-readable description
    pub description: String,
    /// Pattern type this template matches
    pub pattern: String,
    /// Keywords that trigger this template
    pub keywords: Vec<String>,
    /// Default tools to enable
    pub default_tools: Vec<String>,
    /// Recommended complexity level
    pub recommended_complexity: ComplexityLevel,
    /// Template for task instructions
    pub instructions_template: String,
    /// How many successful executions created this template
    pub created_from_executions: usize,
}

impl TaskTemplate {
    /// Create a new template manually
    pub fn new(name: impl Into<String>, pattern: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            pattern: pattern.into(),
            keywords: Vec::new(),
            default_tools: Vec::new(),
            recommended_complexity: ComplexityLevel::Medium,
            instructions_template: String::new(),
            created_from_executions: 0,
        }
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    pub fn with_keywords(mut self, keywords: Vec<String>) -> Self {
        self.keywords = keywords;
        self
    }

    pub fn with_tools(mut self, tools: Vec<String>) -> Self {
        self.default_tools = tools;
        self
    }

    pub fn with_instructions(mut self, template: impl Into<String>) -> Self {
        self.instructions_template = template.into();
        self
    }

    /// Check if this template matches a description
    pub fn matches(&self, description: &str) -> bool {
        let lower = description.to_lowercase();
        self.keywords.iter().any(|kw| lower.contains(kw))
    }

    /// Generate instructions from this template
    pub fn generate_instructions(&self, context: &HashMap<String, String>) -> String {
        let mut instructions = self.instructions_template.clone();
        for (key, value) in context {
            instructions = instructions.replace(&format!("{{{}}}", key), value);
        }
        instructions
    }
}

/// Store of task templates
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TemplateStore {
    templates: Vec<TaskTemplate>,
    #[serde(skip)]
    storage_path: Option<PathBuf>,
}

impl TemplateStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with built-in templates
    pub fn with_defaults() -> Self {
        let mut store = Self::new();
        store.add_defaults();
        store
    }

    pub fn load_or_create(path: PathBuf) -> Result<Self> {
        let mut store = if path.exists() {
            let data = std::fs::read_to_string(&path)?;
            serde_json::from_str(&data).unwrap_or_default()
        } else {
            Self::with_defaults()
        };
        store.storage_path = Some(path);
        Ok(store)
    }

    pub fn save(&self) -> Result<()> {
        if let Some(ref path) = self.storage_path {
            let data = serde_json::to_string_pretty(&self.templates)?;
            std::fs::write(path, data)?;
        }
        Ok(())
    }

    /// Add a template
    pub fn add(&mut self, template: TaskTemplate) -> Result<()> {
        // Check for duplicates
        if !self.templates.iter().any(|t| t.name == template.name) {
            self.templates.push(template);
        }
        Ok(())
    }

    /// Find a template matching a description
    pub fn find_match(&self, description: &str) -> Option<&TaskTemplate> {
        // Score each template
        let lower = description.to_lowercase();
        let mut best_match: Option<(&TaskTemplate, usize)> = None;

        for template in &self.templates {
            let score = template.keywords.iter()
                .filter(|kw| lower.contains(&kw.to_lowercase()))
                .count();

            if score > 0 {
                match best_match {
                    Some((_, best_score)) if score > best_score => {
                        best_match = Some((template, score));
                    }
                    None => {
                        best_match = Some((template, score));
                    }
                    _ => {}
                }
            }
        }

        best_match.map(|(t, _)| t)
    }

    /// Get all templates
    pub fn all(&self) -> &[TaskTemplate] {
        &self.templates
    }

    /// Get template by name
    pub fn get(&self, name: &str) -> Option<&TaskTemplate> {
        self.templates.iter().find(|t| t.name == name)
    }

    /// Add default templates for common tasks
    fn add_defaults(&mut self) {
        // File reading template
        self.templates.push(TaskTemplate {
            name: "file_read".to_string(),
            description: "Read and analyze file contents".to_string(),
            pattern: "file_read".to_string(),
            keywords: vec!["read".into(), "file".into(), "contents".into(), "show".into(), "display".into()],
            default_tools: vec!["read_file".into()],
            recommended_complexity: ComplexityLevel::Simple,
            instructions_template: "Read the file at {path} and summarize its contents.".to_string(),
            created_from_executions: 0,
        });

        // File search template
        self.templates.push(TaskTemplate {
            name: "file_search".to_string(),
            description: "Search for files matching a pattern".to_string(),
            pattern: "file_search".to_string(),
            keywords: vec!["find".into(), "search".into(), "files".into(), "locate".into()],
            default_tools: vec!["search_files".into(), "grep".into()],
            recommended_complexity: ComplexityLevel::Simple,
            instructions_template: "Search for files matching {pattern} in {directory}.".to_string(),
            created_from_executions: 0,
        });

        // Code search template
        self.templates.push(TaskTemplate {
            name: "code_search".to_string(),
            description: "Search for code patterns".to_string(),
            pattern: "code_search".to_string(),
            keywords: vec!["grep".into(), "search".into(), "code".into(), "function".into(), "where".into()],
            default_tools: vec!["grep".into(), "read_file".into()],
            recommended_complexity: ComplexityLevel::Simple,
            instructions_template: "Search for {pattern} in the codebase and report findings.".to_string(),
            created_from_executions: 0,
        });

        // Simple edit template
        self.templates.push(TaskTemplate {
            name: "simple_edit".to_string(),
            description: "Make a simple edit to a file".to_string(),
            pattern: "file_edit".to_string(),
            keywords: vec!["edit".into(), "change".into(), "modify".into(), "update".into()],
            default_tools: vec!["read_file".into(), "write_file".into()],
            recommended_complexity: ComplexityLevel::Medium,
            instructions_template: "Read {path}, make the following change: {change}, and write the file.".to_string(),
            created_from_executions: 0,
        });

        // Test running template
        self.templates.push(TaskTemplate {
            name: "run_tests".to_string(),
            description: "Run tests and report results".to_string(),
            pattern: "test_run".to_string(),
            keywords: vec!["test".into(), "tests".into(), "run".into(), "check".into()],
            default_tools: vec!["execute_bash".into()],
            recommended_complexity: ComplexityLevel::Simple,
            instructions_template: "Run the test suite with {command} and report any failures.".to_string(),
            created_from_executions: 0,
        });

        // Git status template
        self.templates.push(TaskTemplate {
            name: "git_status".to_string(),
            description: "Check git status".to_string(),
            pattern: "git_status".to_string(),
            keywords: vec!["git".into(), "status".into(), "changes".into(), "modified".into()],
            default_tools: vec!["execute_bash".into()],
            recommended_complexity: ComplexityLevel::Trivial,
            instructions_template: "Run git status and summarize the current state.".to_string(),
            created_from_executions: 0,
        });

        // Directory listing template
        self.templates.push(TaskTemplate {
            name: "list_files".to_string(),
            description: "List files in a directory".to_string(),
            pattern: "dir_list".to_string(),
            keywords: vec!["list".into(), "directory".into(), "folder".into(), "ls".into()],
            default_tools: vec!["list_directory".into()],
            recommended_complexity: ComplexityLevel::Trivial,
            instructions_template: "List the contents of {path}.".to_string(),
            created_from_executions: 0,
        });
    }
}
