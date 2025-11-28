//! Learning System for Pattern Recognition and Task Graduation
//!
//! Records execution patterns to help you:
//! - See what works reliably with local models
//! - Identify tasks that always need Claude
//! - Build templates for common workflows
//! - Gradually expand local capabilities

mod patterns;
pub mod recorder;
mod templates;
mod insights;

pub use patterns::{GraduationCandidate, Pattern, PatternMatch, PatternStore};
pub use recorder::{DecompositionRecord, ExecutionRecord, LearningRecorder, OverallOutcome, TaskExecutionRecord};
pub use templates::{TaskTemplate, TemplateStore};
pub use insights::{Insights, TaskInsight};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// The main learning system that ties everything together
pub struct LearningSystem {
    recorder: LearningRecorder,
    patterns: PatternStore,
    templates: TemplateStore,
    storage_path: PathBuf,
}

impl LearningSystem {
    /// Create a new learning system with persistent storage
    pub fn new(storage_path: impl Into<PathBuf>) -> Result<Self> {
        let storage_path = storage_path.into();
        std::fs::create_dir_all(&storage_path)?;

        let recorder = LearningRecorder::load_or_create(storage_path.join("records.json"))?;
        let patterns = PatternStore::load_or_create(storage_path.join("patterns.json"))?;
        let templates = TemplateStore::load_or_create(storage_path.join("templates.json"))?;

        Ok(Self {
            recorder,
            patterns,
            templates,
            storage_path,
        })
    }

    /// Record a task execution
    pub fn record(&mut self, record: ExecutionRecord) -> Result<()> {
        // Store the record
        self.recorder.add(record.clone())?;

        // Update pattern statistics
        self.patterns.update_from_record(&record)?;

        // Check if this creates a template opportunity
        if let Some(template) = self.patterns.extract_template(&record) {
            self.templates.add(template)?;
        }

        self.save()?;
        Ok(())
    }

    /// Get insights about your usage patterns
    pub fn insights(&self) -> Insights {
        Insights::generate(&self.recorder, &self.patterns, &self.templates)
    }

    /// Get tasks that might be ready to "graduate" to local-only
    pub fn graduation_candidates(&self) -> Vec<GraduationCandidate> {
        self.patterns.find_graduation_candidates()
    }

    /// Find a template that matches a task description
    pub fn find_template(&self, description: &str) -> Option<&TaskTemplate> {
        self.templates.find_match(description)
    }

    /// Get all templates
    pub fn templates(&self) -> &TemplateStore {
        &self.templates
    }

    /// Save all data to disk
    pub fn save(&self) -> Result<()> {
        self.recorder.save()?;
        self.patterns.save()?;
        self.templates.save()?;
        Ok(())
    }

    /// Get execution history
    pub fn history(&self) -> &[ExecutionRecord] {
        self.recorder.records()
    }

    /// Get pattern statistics
    pub fn pattern_stats(&self) -> &PatternStore {
        &self.patterns
    }
}

/// Outcome of a task execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskOutcome {
    /// Task completed successfully by local worker
    LocalSuccess {
        worker_steps: usize,
        tools_used: Vec<String>,
        duration_ms: u64,
    },
    /// Task failed when attempted locally
    LocalFailure {
        error: String,
        partial_progress: Option<String>,
    },
    /// Task required Claude's intervention
    NeededClaude {
        reason: String,
    },
    /// Task was too complex, broken into subtasks
    Decomposed {
        subtask_count: usize,
    },
}

impl TaskOutcome {
    pub fn is_local_success(&self) -> bool {
        matches!(self, TaskOutcome::LocalSuccess { .. })
    }

    pub fn is_local_failure(&self) -> bool {
        matches!(self, TaskOutcome::LocalFailure { .. })
    }
}

/// Complexity assessment of a task
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ComplexityLevel {
    /// Simple, single-step task
    Trivial,
    /// Straightforward, few steps
    Simple,
    /// Moderate complexity
    Medium,
    /// Complex, many steps or reasoning
    Complex,
    /// Very complex, needs expert reasoning
    Expert,
}

impl ComplexityLevel {
    pub fn from_steps(steps: usize) -> Self {
        match steps {
            0..=1 => ComplexityLevel::Trivial,
            2..=3 => ComplexityLevel::Simple,
            4..=7 => ComplexityLevel::Medium,
            8..=15 => ComplexityLevel::Complex,
            _ => ComplexityLevel::Expert,
        }
    }
}
