//! Pattern recognition for task success/failure

use super::recorder::ExecutionRecord;
use super::templates::TaskTemplate;
use super::ComplexityLevel;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// A recognized pattern of task execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pattern {
    /// Pattern identifier (e.g., "file_read", "code_refactor")
    pub task_type: String,
    /// Keywords that identify this pattern
    pub keywords: Vec<String>,
    /// Total times this pattern was seen
    pub total_count: usize,
    /// Times it succeeded locally
    pub local_successes: usize,
    /// Times it failed locally
    pub local_failures: usize,
    /// Times Claude handled it
    pub claude_handled: usize,
    /// Average complexity when successful locally
    pub avg_local_complexity: f32,
    /// Common tools used for this pattern
    pub common_tools: HashMap<String, usize>,
    /// Average duration when successful (ms)
    pub avg_duration_ms: u64,
}

impl Pattern {
    pub fn new(task_type: impl Into<String>) -> Self {
        Self {
            task_type: task_type.into(),
            keywords: Vec::new(),
            total_count: 0,
            local_successes: 0,
            local_failures: 0,
            claude_handled: 0,
            avg_local_complexity: 0.0,
            common_tools: HashMap::new(),
            avg_duration_ms: 0,
        }
    }

    /// Calculate local success rate
    pub fn local_success_rate(&self) -> f32 {
        let local_total = self.local_successes + self.local_failures;
        if local_total == 0 {
            return 0.0;
        }
        self.local_successes as f32 / local_total as f32
    }

    /// Is this pattern reliable enough for local-only execution?
    pub fn is_graduation_ready(&self) -> bool {
        // Need at least 5 attempts
        let local_total = self.local_successes + self.local_failures;
        if local_total < 5 {
            return false;
        }

        // Need >80% success rate
        self.local_success_rate() > 0.8
    }

    /// Confidence score for local execution (0.0 - 1.0)
    pub fn local_confidence(&self) -> f32 {
        let total = self.local_successes + self.local_failures;
        if total == 0 {
            return 0.0;
        }

        // Base confidence from success rate
        let success_rate = self.local_success_rate();

        // Adjust for sample size (more samples = more confidence)
        let sample_factor = (total as f32 / 10.0).min(1.0);

        success_rate * sample_factor
    }

    /// Record a successful local execution
    pub fn record_local_success(&mut self, complexity: ComplexityLevel, tools: &[String], duration_ms: u64) {
        self.total_count += 1;
        self.local_successes += 1;

        // Update average complexity
        let n = self.local_successes as f32;
        let complexity_val = complexity as u8 as f32;
        self.avg_local_complexity = ((n - 1.0) * self.avg_local_complexity + complexity_val) / n;

        // Update tool counts
        for tool in tools {
            *self.common_tools.entry(tool.clone()).or_insert(0) += 1;
        }

        // Update average duration
        self.avg_duration_ms = ((n as u64 - 1) * self.avg_duration_ms + duration_ms) / n as u64;
    }

    /// Record a failed local execution
    pub fn record_local_failure(&mut self) {
        self.total_count += 1;
        self.local_failures += 1;
    }

    /// Record Claude handling this
    pub fn record_claude_handled(&mut self) {
        self.total_count += 1;
        self.claude_handled += 1;
    }
}

/// A potential match between input and a pattern
#[derive(Debug, Clone)]
pub struct PatternMatch {
    pub pattern: Pattern,
    pub confidence: f32,
    pub matched_keywords: Vec<String>,
}

/// Store of all recognized patterns
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PatternStore {
    patterns: HashMap<String, Pattern>,
    #[serde(skip)]
    storage_path: Option<PathBuf>,
}

impl PatternStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn load_or_create(path: PathBuf) -> Result<Self> {
        let mut store = if path.exists() {
            let data = std::fs::read_to_string(&path)?;
            serde_json::from_str(&data).unwrap_or_default()
        } else {
            Self::default()
        };
        store.storage_path = Some(path);
        Ok(store)
    }

    pub fn save(&self) -> Result<()> {
        if let Some(ref path) = self.storage_path {
            let data = serde_json::to_string_pretty(&self.patterns)?;
            std::fs::write(path, data)?;
        }
        Ok(())
    }

    /// Update patterns from an execution record
    pub fn update_from_record(&mut self, record: &ExecutionRecord) -> Result<()> {
        for task in &record.task_executions {
            let pattern = self.patterns
                .entry(task.task_type.clone())
                .or_insert_with(|| Pattern::new(&task.task_type));

            if task.executed_locally {
                if task.outcome.is_local_success() {
                    pattern.record_local_success(
                        task.complexity,
                        &task.tools_used,
                        task.duration_ms
                    );
                } else {
                    pattern.record_local_failure();
                }
            } else {
                pattern.record_claude_handled();
            }

            // Extract keywords from description
            let keywords = extract_keywords(&task.description);
            for kw in keywords {
                if !pattern.keywords.contains(&kw) {
                    pattern.keywords.push(kw);
                }
            }
        }

        Ok(())
    }

    /// Find patterns that match a task description
    pub fn find_matches(&self, description: &str) -> Vec<PatternMatch> {
        let input_keywords = extract_keywords(description);
        let mut matches = Vec::new();

        for pattern in self.patterns.values() {
            let matched: Vec<_> = pattern.keywords.iter()
                .filter(|kw| input_keywords.contains(kw))
                .cloned()
                .collect();

            if !matched.is_empty() {
                let keyword_match_rate = matched.len() as f32 / pattern.keywords.len().max(1) as f32;
                let confidence = keyword_match_rate * pattern.local_confidence();

                matches.push(PatternMatch {
                    pattern: pattern.clone(),
                    confidence,
                    matched_keywords: matched,
                });
            }
        }

        matches.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        matches
    }

    /// Get best pattern match for a description
    pub fn best_match(&self, description: &str) -> Option<PatternMatch> {
        self.find_matches(description).into_iter().next()
    }

    /// Find patterns ready for "graduation" to local-only
    pub fn find_graduation_candidates(&self) -> Vec<GraduationCandidate> {
        self.patterns.values()
            .filter(|p| p.is_graduation_ready())
            .map(|p| GraduationCandidate {
                task_type: p.task_type.clone(),
                success_rate: p.local_success_rate(),
                sample_count: p.local_successes + p.local_failures,
                confidence: p.local_confidence(),
                recommended_tools: p.common_tools.keys().cloned().collect(),
                avg_complexity: ComplexityLevel::from_steps(p.avg_local_complexity as usize),
            })
            .collect()
    }

    /// Check if a pattern might generate a template
    pub fn extract_template(&self, record: &ExecutionRecord) -> Option<TaskTemplate> {
        // Look for patterns that have been very successful
        for task in &record.task_executions {
            if let Some(pattern) = self.patterns.get(&task.task_type) {
                if pattern.local_success_rate() > 0.9 && pattern.local_successes >= 3 {
                    return Some(TaskTemplate {
                        name: format!("{}_template", task.task_type),
                        description: format!("Auto-generated template for {}", task.task_type),
                        pattern: task.task_type.clone(),
                        keywords: pattern.keywords.clone(),
                        default_tools: pattern.common_tools.keys().cloned().collect(),
                        recommended_complexity: ComplexityLevel::from_steps(pattern.avg_local_complexity as usize),
                        instructions_template: task.description.clone(),
                        created_from_executions: pattern.local_successes,
                    });
                }
            }
        }
        None
    }

    /// Get all patterns
    pub fn all(&self) -> impl Iterator<Item = &Pattern> {
        self.patterns.values()
    }

    /// Get pattern by type
    pub fn get(&self, task_type: &str) -> Option<&Pattern> {
        self.patterns.get(task_type)
    }
}

/// A task that might be ready to run locally without Claude
#[derive(Debug, Clone)]
pub struct GraduationCandidate {
    pub task_type: String,
    pub success_rate: f32,
    pub sample_count: usize,
    pub confidence: f32,
    pub recommended_tools: Vec<String>,
    pub avg_complexity: ComplexityLevel,
}

/// Extract keywords from a description
fn extract_keywords(text: &str) -> Vec<String> {
    // Common programming/task keywords
    let important_words = [
        "read", "write", "file", "create", "delete", "search", "find",
        "refactor", "test", "fix", "bug", "implement", "add", "remove",
        "update", "change", "modify", "analyze", "review", "check",
        "format", "lint", "build", "compile", "run", "execute",
        "function", "class", "module", "api", "endpoint", "database",
        "config", "setup", "install", "deploy", "git", "commit",
    ];

    text.to_lowercase()
        .split_whitespace()
        .filter(|w| important_words.contains(w))
        .map(|s| s.to_string())
        .collect()
}
