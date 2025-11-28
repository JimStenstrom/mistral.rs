//! Git-based Sandbox for Agentic Workflows
//!
//! Uses git branches to isolate each workflow stage, providing:
//! - Natural rollback (delete branch and retry)
//! - Full audit trail of agent actions
//! - Review points between stages
//! - Safe parallel agent execution
//!
//! ## Branch Structure
//!
//! ```text
//! main
//!   └── task/{task-id}                    # Task branch
//!         ├── task/{task-id}/1-research   # Stage branches
//!         ├── task/{task-id}/2-planning
//!         ├── task/{task-id}/3-implementation
//!         ├── task/{task-id}/4-validation
//!         └── task/{task-id}/5-cleanup
//! ```

use crate::workflow::WorkflowStage;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use tokio::process::Command;

/// Git sandbox for isolating workflow stages
pub struct GitSandbox {
    /// Repository root path
    repo_path: PathBuf,
    /// Task identifier
    task_id: String,
    /// Base branch (usually main/master)
    base_branch: String,
    /// Current stage
    current_stage: Option<WorkflowStage>,
    /// Configuration
    config: SandboxConfig,
}

/// Configuration for the git sandbox
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxConfig {
    /// Prefix for task branches
    pub branch_prefix: String,
    /// Whether to auto-commit after each tool execution
    pub auto_commit: bool,
    /// Whether to require human approval between stages
    pub require_human_approval: bool,
    /// Whether to squash commits when merging stages
    pub squash_on_merge: bool,
    /// Commit message prefix for agent commits
    pub agent_commit_prefix: String,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            branch_prefix: "task".to_string(),
            auto_commit: true,
            require_human_approval: true,
            squash_on_merge: false,
            agent_commit_prefix: "[agent]".to_string(),
        }
    }
}

/// Result of a stage review
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageReview {
    pub stage: WorkflowStage,
    pub branch_name: String,
    pub commits: Vec<CommitInfo>,
    pub files_changed: Vec<FileChange>,
    pub diff_summary: String,
    pub orchestrator_approved: bool,
    pub orchestrator_notes: Option<String>,
    pub human_approved: Option<bool>,
    pub human_notes: Option<String>,
}

/// Information about a commit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitInfo {
    pub hash: String,
    pub short_hash: String,
    pub author: String,
    pub message: String,
    pub timestamp: String,
}

/// Information about a changed file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileChange {
    pub path: String,
    pub change_type: ChangeType,
    pub additions: usize,
    pub deletions: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeType {
    Added,
    Modified,
    Deleted,
    Renamed,
}

impl GitSandbox {
    /// Create a new git sandbox for a task
    pub async fn new(
        repo_path: impl Into<PathBuf>,
        task_id: impl Into<String>,
        config: SandboxConfig,
    ) -> Result<Self> {
        let repo_path = repo_path.into();
        let task_id = task_id.into();

        // Verify it's a git repo
        if !repo_path.join(".git").exists() {
            return Err(anyhow!("Not a git repository: {:?}", repo_path));
        }

        // Get the base branch
        let base_branch = Self::get_current_branch_static(&repo_path).await?;

        Ok(Self {
            repo_path,
            task_id,
            base_branch,
            current_stage: None,
            config,
        })
    }

    /// Get the task ID
    pub fn task_id(&self) -> &str {
        &self.task_id
    }

    /// Initialize the sandbox - create task branch from base
    pub async fn initialize(&mut self) -> Result<()> {
        let task_branch = self.task_branch_name();

        // Stash any uncommitted changes
        self.git(&["stash", "push", "-m", "Pre-task stash"]).await.ok();

        // Create and checkout the task branch
        self.git(&["checkout", "-b", &task_branch]).await?;

        Ok(())
    }

    /// Start a workflow stage - creates stage branch
    pub async fn start_stage(&mut self, stage: WorkflowStage) -> Result<String> {
        let stage_branch = self.stage_branch_name(stage);
        let task_branch = self.task_branch_name();

        // Ensure we're on the task branch
        self.git(&["checkout", &task_branch]).await?;

        // Create stage branch
        self.git(&["checkout", "-b", &stage_branch]).await?;

        self.current_stage = Some(stage);

        Ok(stage_branch)
    }

    /// Commit changes made by an agent
    pub async fn agent_commit(&self, agent_id: &str, message: &str) -> Result<String> {
        // Stage all changes
        self.git(&["add", "-A"]).await?;

        // Check if there are changes to commit
        let status = self.git(&["status", "--porcelain"]).await?;
        if status.trim().is_empty() {
            return Ok("No changes to commit".to_string());
        }

        // Create commit with agent prefix
        let full_message = format!(
            "{} [{}] {}",
            self.config.agent_commit_prefix, agent_id, message
        );

        self.git(&["commit", "-m", &full_message]).await?;

        // Get the commit hash
        let hash = self.git(&["rev-parse", "HEAD"]).await?;
        Ok(hash.trim().to_string())
    }

    /// Complete a stage and prepare for review
    pub async fn complete_stage(&self) -> Result<StageReview> {
        let stage = self.current_stage
            .ok_or_else(|| anyhow!("No stage in progress"))?;

        let stage_branch = self.stage_branch_name(stage);
        let task_branch = self.task_branch_name();

        // Get commits in this stage
        let commits = self.get_commits_since(&task_branch).await?;

        // Get file changes
        let files_changed = self.get_files_changed(&task_branch).await?;

        // Get diff summary
        let diff_summary = self.git(&[
            "diff", "--stat", &format!("{}..{}", task_branch, stage_branch)
        ]).await?;

        Ok(StageReview {
            stage,
            branch_name: stage_branch,
            commits,
            files_changed,
            diff_summary,
            orchestrator_approved: false,
            orchestrator_notes: None,
            human_approved: None,
            human_notes: None,
        })
    }

    /// Orchestrator approves the stage - prepares for human review
    pub async fn orchestrator_approve(&self, review: &mut StageReview, notes: Option<String>) -> Result<()> {
        review.orchestrator_approved = true;
        review.orchestrator_notes = notes;
        Ok(())
    }

    /// Human approves the stage - merges to task branch
    pub async fn human_approve(&mut self, review: &mut StageReview, notes: Option<String>) -> Result<()> {
        if !review.orchestrator_approved {
            return Err(anyhow!("Orchestrator must approve before human"));
        }

        review.human_approved = Some(true);
        review.human_notes = notes;

        // Merge stage branch to task branch
        self.merge_stage_to_task(review.stage).await?;

        self.current_stage = None;

        Ok(())
    }

    /// Human rejects the stage - options for retry or abort
    pub async fn human_reject(
        &mut self,
        review: &mut StageReview,
        notes: String,
        action: RejectAction,
    ) -> Result<()> {
        review.human_approved = Some(false);
        review.human_notes = Some(notes);

        match action {
            RejectAction::Retry => {
                // Delete the stage branch and start fresh
                let stage = review.stage;
                self.abort_stage().await?;
                self.start_stage(stage).await?;
            }
            RejectAction::Abort => {
                self.abort_stage().await?;
            }
            RejectAction::ModifyAndContinue => {
                // Stay on stage branch, human will make changes
            }
        }

        Ok(())
    }

    /// Abort current stage - delete stage branch
    pub async fn abort_stage(&mut self) -> Result<()> {
        if let Some(stage) = self.current_stage {
            let stage_branch = self.stage_branch_name(stage);
            let task_branch = self.task_branch_name();

            // Switch to task branch
            self.git(&["checkout", &task_branch]).await?;

            // Delete stage branch
            self.git(&["branch", "-D", &stage_branch]).await?;

            self.current_stage = None;
        }
        Ok(())
    }

    /// Merge stage branch to task branch
    async fn merge_stage_to_task(&self, stage: WorkflowStage) -> Result<()> {
        let stage_branch = self.stage_branch_name(stage);
        let task_branch = self.task_branch_name();

        // Checkout task branch
        self.git(&["checkout", &task_branch]).await?;

        // Merge stage branch
        if self.config.squash_on_merge {
            self.git(&["merge", "--squash", &stage_branch]).await?;
            let message = format!("Complete {} stage", stage_name(stage));
            self.git(&["commit", "-m", &message]).await?;
        } else {
            let message = format!("Merge {} stage", stage_name(stage));
            self.git(&["merge", "--no-ff", "-m", &message, &stage_branch]).await?;
        }

        // Delete stage branch (it's merged)
        self.git(&["branch", "-d", &stage_branch]).await?;

        Ok(())
    }

    /// Finalize task - merge to base branch
    pub async fn finalize(&self) -> Result<FinalizeResult> {
        let task_branch = self.task_branch_name();

        // Get summary of all changes
        let diff_summary = self.git(&[
            "diff", "--stat", &format!("{}..{}", self.base_branch, task_branch)
        ]).await?;

        let commits = self.get_all_task_commits().await?;

        Ok(FinalizeResult {
            task_branch: task_branch.clone(),
            target_branch: self.base_branch.clone(),
            commits,
            diff_summary,
            ready_to_merge: true,
        })
    }

    /// Merge task to base branch (after final human approval)
    pub async fn merge_to_base(&self, squash: bool) -> Result<()> {
        let task_branch = self.task_branch_name();

        // Checkout base branch
        self.git(&["checkout", &self.base_branch]).await?;

        if squash {
            self.git(&["merge", "--squash", &task_branch]).await?;
            let message = format!("Complete task: {}", self.task_id);
            self.git(&["commit", "-m", &message]).await?;
        } else {
            let message = format!("Merge task: {}", self.task_id);
            self.git(&["merge", "--no-ff", "-m", &message, &task_branch]).await?;
        }

        // Optionally delete task branch
        // self.git(&["branch", "-d", &task_branch]).await?;

        Ok(())
    }

    /// Abort entire task - delete task branch
    pub async fn abort_task(&mut self) -> Result<()> {
        let task_branch = self.task_branch_name();

        // Checkout base branch
        self.git(&["checkout", &self.base_branch]).await?;

        // Delete task branch (force, may have unmerged changes)
        self.git(&["branch", "-D", &task_branch]).await?;

        // Restore stashed changes if any
        self.git(&["stash", "pop"]).await.ok();

        self.current_stage = None;

        Ok(())
    }

    /// Get the current git diff (for agent context)
    pub async fn get_current_diff(&self) -> Result<String> {
        self.git(&["diff"]).await
    }

    /// Get staged changes diff
    pub async fn get_staged_diff(&self) -> Result<String> {
        self.git(&["diff", "--staged"]).await
    }

    /// Get list of modified files
    pub async fn get_modified_files(&self) -> Result<Vec<String>> {
        let output = self.git(&["status", "--porcelain"]).await?;
        Ok(output
            .lines()
            .filter_map(|line| {
                if line.len() > 3 {
                    Some(line[3..].to_string())
                } else {
                    None
                }
            })
            .collect())
    }

    // Helper methods

    fn task_branch_name(&self) -> String {
        format!("{}/{}", self.config.branch_prefix, self.task_id)
    }

    fn stage_branch_name(&self, stage: WorkflowStage) -> String {
        format!(
            "{}/{}/{}",
            self.config.branch_prefix,
            self.task_id,
            stage_branch_suffix(stage)
        )
    }

    async fn git(&self, args: &[&str]) -> Result<String> {
        let output = Command::new("git")
            .args(args)
            .current_dir(&self.repo_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await?;

        if output.status.success() {
            Ok(String::from_utf8_lossy(&output.stdout).to_string())
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(anyhow!("git {:?} failed: {}", args, stderr))
        }
    }

    async fn get_current_branch_static(repo_path: &Path) -> Result<String> {
        let output = Command::new("git")
            .args(["branch", "--show-current"])
            .current_dir(repo_path)
            .stdout(Stdio::piped())
            .output()
            .await?;

        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }

    async fn get_commits_since(&self, base: &str) -> Result<Vec<CommitInfo>> {
        let format = "%H|%h|%an|%s|%ci";
        let output = self.git(&[
            "log",
            &format!("{}..HEAD", base),
            &format!("--format={}", format),
        ]).await?;

        Ok(output
            .lines()
            .filter(|line| !line.is_empty())
            .filter_map(|line| {
                let parts: Vec<&str> = line.split('|').collect();
                if parts.len() >= 5 {
                    Some(CommitInfo {
                        hash: parts[0].to_string(),
                        short_hash: parts[1].to_string(),
                        author: parts[2].to_string(),
                        message: parts[3].to_string(),
                        timestamp: parts[4].to_string(),
                    })
                } else {
                    None
                }
            })
            .collect())
    }

    async fn get_files_changed(&self, base: &str) -> Result<Vec<FileChange>> {
        let output = self.git(&[
            "diff", "--numstat", &format!("{}..HEAD", base)
        ]).await?;

        Ok(output
            .lines()
            .filter(|line| !line.is_empty())
            .filter_map(|line| {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 3 {
                    let additions = parts[0].parse().unwrap_or(0);
                    let deletions = parts[1].parse().unwrap_or(0);
                    let path = parts[2].to_string();

                    let change_type = if additions > 0 && deletions == 0 {
                        ChangeType::Added
                    } else if additions == 0 && deletions > 0 {
                        ChangeType::Deleted
                    } else {
                        ChangeType::Modified
                    };

                    Some(FileChange {
                        path,
                        change_type,
                        additions,
                        deletions,
                    })
                } else {
                    None
                }
            })
            .collect())
    }

    async fn get_all_task_commits(&self) -> Result<Vec<CommitInfo>> {
        self.get_commits_since(&self.base_branch).await
    }
}

/// Action to take when human rejects a stage
#[derive(Debug, Clone, Copy)]
pub enum RejectAction {
    /// Delete stage branch and retry from scratch
    Retry,
    /// Abort the entire task
    Abort,
    /// Keep changes, let human modify before continuing
    ModifyAndContinue,
}

/// Result of finalizing a task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinalizeResult {
    pub task_branch: String,
    pub target_branch: String,
    pub commits: Vec<CommitInfo>,
    pub diff_summary: String,
    pub ready_to_merge: bool,
}

fn stage_branch_suffix(stage: WorkflowStage) -> &'static str {
    match stage {
        WorkflowStage::Research => "1-research",
        WorkflowStage::Planning => "2-planning",
        WorkflowStage::Implementation => "3-implementation",
        WorkflowStage::Validation => "4-validation",
        WorkflowStage::Cleanup => "5-cleanup",
    }
}

fn stage_name(stage: WorkflowStage) -> &'static str {
    match stage {
        WorkflowStage::Research => "research",
        WorkflowStage::Planning => "planning",
        WorkflowStage::Implementation => "implementation",
        WorkflowStage::Validation => "validation",
        WorkflowStage::Cleanup => "cleanup",
    }
}

/// Builder for creating a git-sandboxed workflow
pub struct SandboxedWorkflowBuilder {
    repo_path: PathBuf,
    task_id: String,
    config: SandboxConfig,
}

impl SandboxedWorkflowBuilder {
    pub fn new(repo_path: impl Into<PathBuf>, task_id: impl Into<String>) -> Self {
        Self {
            repo_path: repo_path.into(),
            task_id: task_id.into(),
            config: SandboxConfig::default(),
        }
    }

    pub fn with_config(mut self, config: SandboxConfig) -> Self {
        self.config = config;
        self
    }

    pub fn auto_commit(mut self, enabled: bool) -> Self {
        self.config.auto_commit = enabled;
        self
    }

    pub fn require_human_approval(mut self, required: bool) -> Self {
        self.config.require_human_approval = required;
        self
    }

    pub fn squash_on_merge(mut self, squash: bool) -> Self {
        self.config.squash_on_merge = squash;
        self
    }

    pub async fn build(self) -> Result<GitSandbox> {
        GitSandbox::new(self.repo_path, self.task_id, self.config).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_branch_names() {
        let config = SandboxConfig::default();
        assert_eq!(config.branch_prefix, "task");
    }

    #[test]
    fn test_stage_suffixes() {
        assert_eq!(stage_branch_suffix(WorkflowStage::Research), "1-research");
        assert_eq!(stage_branch_suffix(WorkflowStage::Cleanup), "5-cleanup");
    }
}
