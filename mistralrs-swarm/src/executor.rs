//! Simple, Elegant Workflow Executor
//!
//! One function. Five stages. Git isolation. Human gates.
//!
//! ```rust,ignore
//! let result = execute("Fix the login bug", &repo_path).await?;
//! ```

use crate::git_sandbox::{GitSandbox, SandboxConfig, StageReview};
use crate::learning::LearningSystem;
use crate::tools::ToolRegistry;
use crate::workflow::WorkflowStage;
use anyhow::Result;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;

/// The simplest possible API - one function to run a complete workflow
pub async fn execute(goal: &str, repo_path: &Path) -> Result<ExecutionResult> {
    Executor::new(repo_path).await?.run(goal).await
}

/// Execution result
#[derive(Debug)]
pub struct ExecutionResult {
    pub task_id: String,
    pub goal: String,
    pub stages_completed: Vec<WorkflowStage>,
    pub final_output: String,
    pub branch: String,
    pub merged: bool,
}

/// Callback for human review
pub type ReviewCallback = Box<dyn Fn(&StageReview) -> ReviewDecision + Send + Sync>;

/// Human's decision at a review gate
#[derive(Debug, Clone)]
pub enum ReviewDecision {
    Approve,
    Reject(String),
    Retry,
    Abort,
}

/// The executor - runs the full workflow
pub struct Executor {
    sandbox: GitSandbox,
    tools: ToolRegistry,
    learning: Option<Arc<Mutex<LearningSystem>>>,
    on_review: Option<ReviewCallback>,
    config: ExecutorConfig,
}

/// Simple configuration
#[derive(Debug, Clone)]
pub struct ExecutorConfig {
    /// Model for local agents
    pub model_id: String,
    /// Anthropic API key for orchestrator (None = local only)
    pub anthropic_key: Option<String>,
    /// Stages that require human approval
    pub human_gates: Vec<WorkflowStage>,
    /// Test command to run in validation
    pub test_cmd: Option<String>,
    /// Lint command
    pub lint_cmd: Option<String>,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            model_id: "Qwen/Qwen2.5-Coder-7B-Instruct".to_string(),
            anthropic_key: None,
            // By default, require human approval after implementation and before final merge
            human_gates: vec![
                WorkflowStage::Implementation,
                WorkflowStage::Cleanup,
            ],
            test_cmd: None,
            lint_cmd: None,
        }
    }
}

impl Executor {
    /// Create a new executor for a repository
    pub async fn new(repo_path: &Path) -> Result<Self> {
        let task_id = generate_task_id();
        let sandbox = GitSandbox::new(
            repo_path,
            &task_id,
            SandboxConfig::default(),
        ).await?;

        Ok(Self {
            sandbox,
            tools: ToolRegistry::with_builtins(),
            learning: None,
            on_review: None,
            config: ExecutorConfig::default(),
        })
    }

    /// Configure the executor
    pub fn with_config(mut self, config: ExecutorConfig) -> Self {
        self.config = config;
        self
    }

    /// Enable learning system
    pub fn with_learning(mut self, path: &Path) -> Result<Self> {
        let learning = LearningSystem::new(path)?;
        self.learning = Some(Arc::new(Mutex::new(learning)));
        Ok(self)
    }

    /// Set custom review callback
    pub fn on_review<F>(mut self, callback: F) -> Self
    where
        F: Fn(&StageReview) -> ReviewDecision + Send + Sync + 'static,
    {
        self.on_review = Some(Box::new(callback));
        self
    }

    /// Run the complete workflow
    pub async fn run(&mut self, goal: &str) -> Result<ExecutionResult> {
        let task_id = self.sandbox.task_id().to_string();

        // Initialize git sandbox
        self.sandbox.initialize().await?;

        let mut stages_completed = Vec::new();
        let mut final_output = String::new();

        // Run each stage
        for stage in all_stages() {
            // Start stage branch
            self.sandbox.start_stage(stage).await?;

            // Execute the stage
            let output = self.execute_stage(stage, goal).await?;

            // Commit the work
            self.sandbox.agent_commit("executor", &stage_summary(stage)).await?;

            // Complete and review
            let mut review = self.sandbox.complete_stage().await?;

            // Auto-approve by orchestrator (could add AI review here)
            self.sandbox.orchestrator_approve(&mut review, None).await?;

            // Human gate?
            if self.config.human_gates.contains(&stage) {
                let decision = self.get_human_review(&review).await;

                match decision {
                    ReviewDecision::Approve => {
                        self.sandbox.human_approve(&mut review, None).await?;
                    }
                    ReviewDecision::Retry => {
                        self.sandbox.human_reject(
                            &mut review,
                            "Retry requested".into(),
                            crate::git_sandbox::RejectAction::Retry,
                        ).await?;
                        continue; // Retry this stage
                    }
                    ReviewDecision::Reject(reason) => {
                        self.sandbox.abort_task().await?;
                        return Err(anyhow::anyhow!("Rejected: {}", reason));
                    }
                    ReviewDecision::Abort => {
                        self.sandbox.abort_task().await?;
                        return Err(anyhow::anyhow!("Aborted by user"));
                    }
                }
            } else {
                // Auto-approve non-gated stages
                self.sandbox.human_approve(&mut review, None).await?;
            }

            stages_completed.push(stage);
            final_output = output;
        }

        // Finalize
        let finalize = self.sandbox.finalize().await?;

        Ok(ExecutionResult {
            task_id,
            goal: goal.to_string(),
            stages_completed,
            final_output,
            branch: finalize.task_branch,
            merged: false, // Human decides when to merge to main
        })
    }

    /// Execute a single stage
    async fn execute_stage(&self, stage: WorkflowStage, goal: &str) -> Result<String> {
        match stage {
            WorkflowStage::Research => self.research(goal).await,
            WorkflowStage::Planning => self.plan(goal).await,
            WorkflowStage::Implementation => self.implement(goal).await,
            WorkflowStage::Validation => self.validate().await,
            WorkflowStage::Cleanup => self.cleanup().await,
        }
    }

    async fn research(&self, goal: &str) -> Result<String> {
        // Use tools to understand the codebase
        let files = self.tools.execute(&crate::tools::ToolCall::new(
            "search_files",
            serde_json::json!({"pattern": "**/*.rs", "directory": "."}),
        )).await?;

        Ok(format!("Researched codebase for: {}\n{}", goal, files.output))
    }

    async fn plan(&self, goal: &str) -> Result<String> {
        // Break down the goal into steps
        // In real implementation, this would use the LLM
        Ok(format!("Plan for: {}\n1. Identify affected files\n2. Make changes\n3. Test", goal))
    }

    async fn implement(&self, _goal: &str) -> Result<String> {
        // Execute the plan using tools
        // In real implementation, this would be the agent loop
        Ok("Implementation complete".to_string())
    }

    async fn validate(&self) -> Result<String> {
        let mut results = Vec::new();

        // Run tests if configured
        if let Some(ref cmd) = self.config.test_cmd {
            let test_result = self.tools.execute(&crate::tools::ToolCall::new(
                "bash",
                serde_json::json!({"command": cmd}),
            )).await?;
            results.push(format!("Tests: {}", if test_result.success { "PASS" } else { "FAIL" }));
        }

        // Run linter if configured
        if let Some(ref cmd) = self.config.lint_cmd {
            let lint_result = self.tools.execute(&crate::tools::ToolCall::new(
                "bash",
                serde_json::json!({"command": cmd}),
            )).await?;
            results.push(format!("Lint: {}", if lint_result.success { "PASS" } else { "FAIL" }));
        }

        Ok(results.join("\n"))
    }

    async fn cleanup(&self) -> Result<String> {
        // Remove debug statements, format code, etc.
        Ok("Cleanup complete".to_string())
    }

    async fn get_human_review(&self, review: &StageReview) -> ReviewDecision {
        if let Some(ref callback) = self.on_review {
            callback(review)
        } else {
            // Default: print and wait for input (in real impl)
            println!("\n═══════════════════════════════════════");
            println!("  REVIEW: {:?}", review.stage);
            println!("═══════════════════════════════════════");
            println!("Branch: {}", review.branch_name);
            println!("Commits: {}", review.commits.len());
            println!("Files changed: {}", review.files_changed.len());
            println!("{}", review.diff_summary);
            println!("═══════════════════════════════════════\n");

            // Auto-approve for now (in real impl, would wait for user input)
            ReviewDecision::Approve
        }
    }
}


fn generate_task_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    format!("{:x}", timestamp)
}

fn all_stages() -> Vec<WorkflowStage> {
    vec![
        WorkflowStage::Research,
        WorkflowStage::Planning,
        WorkflowStage::Implementation,
        WorkflowStage::Validation,
        WorkflowStage::Cleanup,
    ]
}

fn stage_summary(stage: WorkflowStage) -> String {
    match stage {
        WorkflowStage::Research => "Complete research phase",
        WorkflowStage::Planning => "Complete planning phase",
        WorkflowStage::Implementation => "Complete implementation",
        WorkflowStage::Validation => "Complete validation",
        WorkflowStage::Cleanup => "Complete cleanup",
    }.to_string()
}

// ============================================================================
// THE SIMPLE API
// ============================================================================

/// Builder for elegant configuration
pub struct Run<'a> {
    goal: &'a str,
    repo: &'a Path,
    config: ExecutorConfig,
}

impl<'a> Run<'a> {
    pub fn new(goal: &'a str, repo: &'a Path) -> Self {
        Self {
            goal,
            repo,
            config: ExecutorConfig::default(),
        }
    }

    pub fn model(mut self, model_id: &str) -> Self {
        self.config.model_id = model_id.to_string();
        self
    }

    pub fn with_claude(mut self, api_key: &str) -> Self {
        self.config.anthropic_key = Some(api_key.to_string());
        self
    }

    pub fn test_cmd(mut self, cmd: &str) -> Self {
        self.config.test_cmd = Some(cmd.to_string());
        self
    }

    pub fn lint_cmd(mut self, cmd: &str) -> Self {
        self.config.lint_cmd = Some(cmd.to_string());
        self
    }

    pub fn gates(mut self, stages: Vec<WorkflowStage>) -> Self {
        self.config.human_gates = stages;
        self
    }

    pub async fn execute(self) -> Result<ExecutionResult> {
        Executor::new(self.repo)
            .await?
            .with_config(self.config)
            .run(self.goal)
            .await
    }
}

// ============================================================================
// USAGE EXAMPLES
// ============================================================================

#[cfg(test)]
mod examples {
    use super::*;
    use std::path::PathBuf;

    // Simplest possible usage
    async fn example_simple() -> Result<()> {
        let result = execute("Fix the login bug", Path::new(".")).await?;
        println!("Done! Branch: {}", result.branch);
        Ok(())
    }

    // With configuration
    async fn example_configured() -> Result<()> {
        let result = Run::new("Add user authentication", Path::new("."))
            .model("Qwen/Qwen2.5-Coder-14B-Instruct")
            .with_claude("sk-ant-...")
            .test_cmd("cargo test")
            .lint_cmd("cargo clippy")
            .gates(vec![WorkflowStage::Implementation])
            .execute()
            .await?;

        println!("Task {} complete", result.task_id);
        Ok(())
    }

    // With custom review handling
    async fn example_custom_review() -> Result<()> {
        let result = Executor::new(Path::new("."))
            .await?
            .on_review(|review| {
                // Custom logic - maybe show a UI, send to Slack, etc.
                println!("Reviewing {:?}...", review.stage);
                ReviewDecision::Approve
            })
            .run("Refactor the payment module")
            .await?;

        Ok(())
    }
}
