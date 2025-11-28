//! Enhanced 5-Stage Workflow
//!
//! Extends the typical Research → Planning → Implementation flow
//! with Validation and Cleanup stages for production-quality output.

use serde::{Deserialize, Serialize};

/// The five stages of a complete code generation workflow
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkflowStage {
    /// Stage 1: Research - Understand the codebase and requirements
    /// - Analyze existing code structure
    /// - Identify dependencies and patterns
    /// - Gather context needed for implementation
    Research,

    /// Stage 2: Planning - Break down into actionable steps
    /// - Decompose task into subtasks
    /// - Identify files to modify
    /// - Determine execution order
    Planning,

    /// Stage 3: Implementation - Execute the plan
    /// - Make code changes
    /// - Create new files as needed
    /// - Execute shell commands
    Implementation,

    /// Stage 4: Validation - Verify the changes work
    /// - Run existing tests
    /// - Run linter/formatter
    /// - Check for regressions
    /// - Self-review: re-read changes and verify correctness
    Validation,

    /// Stage 5: Cleanup - Polish and finalize
    /// - Remove debug statements
    /// - Update documentation
    /// - Add/update tests for new code
    /// - Create meaningful commit message
    Cleanup,
}

impl WorkflowStage {
    pub fn next(&self) -> Option<WorkflowStage> {
        match self {
            WorkflowStage::Research => Some(WorkflowStage::Planning),
            WorkflowStage::Planning => Some(WorkflowStage::Implementation),
            WorkflowStage::Implementation => Some(WorkflowStage::Validation),
            WorkflowStage::Validation => Some(WorkflowStage::Cleanup),
            WorkflowStage::Cleanup => None,
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            WorkflowStage::Research => "Gathering context and understanding requirements",
            WorkflowStage::Planning => "Breaking down task into actionable steps",
            WorkflowStage::Implementation => "Executing planned changes",
            WorkflowStage::Validation => "Verifying changes work correctly",
            WorkflowStage::Cleanup => "Polishing code and updating documentation",
        }
    }
}

/// Configuration for each workflow stage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageConfig {
    /// Skip this stage entirely
    pub skip: bool,
    /// Maximum time for this stage (seconds)
    pub timeout_secs: u64,
    /// Maximum LLM calls for this stage
    pub max_llm_calls: usize,
    /// Specific tools allowed in this stage
    pub allowed_tools: Option<Vec<String>>,
}

impl Default for StageConfig {
    fn default() -> Self {
        Self {
            skip: false,
            timeout_secs: 300,
            max_llm_calls: 50,
            allowed_tools: None,
        }
    }
}

/// Workflow configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowConfig {
    pub research: StageConfig,
    pub planning: StageConfig,
    pub implementation: StageConfig,
    pub validation: ValidationConfig,
    pub cleanup: CleanupConfig,
}

impl Default for WorkflowConfig {
    fn default() -> Self {
        Self {
            research: StageConfig::default(),
            planning: StageConfig::default(),
            implementation: StageConfig::default(),
            validation: ValidationConfig::default(),
            cleanup: CleanupConfig::default(),
        }
    }
}

/// Validation stage specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    pub base: StageConfig,
    /// Command to run tests
    pub test_cmd: Option<String>,
    /// Run tests automatically after implementation
    pub auto_test: bool,
    /// Command to run linter
    pub lint_cmd: Option<String>,
    /// Command to run formatter
    pub format_cmd: Option<String>,
    /// Re-read and verify changes (self-review)
    pub self_review: bool,
    /// Maximum test retry attempts
    pub max_test_retries: usize,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            base: StageConfig::default(),
            test_cmd: None,
            auto_test: true,
            lint_cmd: None,
            format_cmd: None,
            self_review: true,
            max_test_retries: 3,
        }
    }
}

/// Cleanup stage specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleanupConfig {
    pub base: StageConfig,
    /// Patterns to detect debug statements
    pub debug_patterns: Vec<String>,
    /// Update documentation files
    pub update_docs: bool,
    /// Add tests for new code
    pub add_tests: bool,
    /// Generate commit message
    pub generate_commit: bool,
}

impl Default for CleanupConfig {
    fn default() -> Self {
        Self {
            base: StageConfig::default(),
            debug_patterns: vec![
                "console.log".to_string(),
                "print(".to_string(),
                "dbg!".to_string(),
                "println!".to_string(),
                "// TODO".to_string(),
                "// FIXME".to_string(),
                "# DEBUG".to_string(),
            ],
            update_docs: true,
            add_tests: false, // Off by default - can be complex
            generate_commit: true,
        }
    }
}

/// Result of a workflow stage execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageResult {
    pub stage: WorkflowStage,
    pub success: bool,
    pub output: String,
    pub errors: Vec<String>,
    pub llm_calls: usize,
    pub duration_ms: u64,
    /// Should we proceed to the next stage?
    pub proceed: bool,
    /// Should we retry this stage?
    pub retry: bool,
}

impl StageResult {
    pub fn success(stage: WorkflowStage, output: impl Into<String>) -> Self {
        Self {
            stage,
            success: true,
            output: output.into(),
            errors: Vec::new(),
            llm_calls: 0,
            duration_ms: 0,
            proceed: true,
            retry: false,
        }
    }

    pub fn failure(stage: WorkflowStage, error: impl Into<String>) -> Self {
        Self {
            stage,
            success: false,
            output: String::new(),
            errors: vec![error.into()],
            llm_calls: 0,
            duration_ms: 0,
            proceed: false,
            retry: false,
        }
    }

    pub fn needs_retry(mut self) -> Self {
        self.retry = true;
        self.proceed = false;
        self
    }
}

/// Validation check results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResults {
    pub tests_passed: Option<bool>,
    pub test_output: Option<String>,
    pub lint_passed: Option<bool>,
    pub lint_issues: Vec<String>,
    pub format_passed: Option<bool>,
    pub self_review_passed: bool,
    pub self_review_notes: Vec<String>,
}

impl ValidationResults {
    pub fn all_passed(&self) -> bool {
        self.tests_passed.unwrap_or(true)
            && self.lint_passed.unwrap_or(true)
            && self.format_passed.unwrap_or(true)
            && self.self_review_passed
    }
}

/// Cleanup check results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleanupResults {
    pub debug_statements_removed: usize,
    pub docs_updated: bool,
    pub tests_added: bool,
    pub commit_message: Option<String>,
}

/// Prompts for each workflow stage
pub mod prompts {
    /// System prompt for the validation stage
    pub const VALIDATION_SYSTEM: &str = r#"
You are reviewing code changes for correctness and quality.

Your tasks:
1. Re-read all modified files to verify the changes are correct
2. Check for obvious bugs, edge cases, or missing error handling
3. Verify the changes match what was requested
4. Look for any unintended side effects

Be critical but constructive. If you find issues, explain what's wrong
and suggest fixes. Focus on functional correctness, not style.
"#;

    /// System prompt for the cleanup stage
    pub const CLEANUP_SYSTEM: &str = r#"
You are cleaning up code after implementation.

Your tasks:
1. Remove any debug statements (console.log, print, dbg!, etc.)
2. Remove any TODO/FIXME comments that were addressed
3. Ensure code follows project conventions
4. Update relevant documentation if needed
5. Write a clear, concise commit message summarizing the changes

Be thorough but don't over-engineer. Keep changes minimal and focused.
"#;

    /// Prompt template for self-review
    pub const SELF_REVIEW: &str = r#"
Review the following code changes:

{changes}

Questions to answer:
1. Do these changes correctly implement the requested feature/fix?
2. Are there any obvious bugs or edge cases not handled?
3. Is error handling appropriate?
4. Are there any security concerns?
5. Would you approve this code review?

Provide specific feedback if issues are found.
"#;
}
