//! Insights and reporting from learning data

use super::patterns::{GraduationCandidate, Pattern, PatternStore};
use super::recorder::LearningRecorder;
use super::templates::TemplateStore;
use super::ComplexityLevel;
use colored::Colorize;
use std::fmt;

/// Aggregated insights from the learning system
#[derive(Debug)]
pub struct Insights {
    /// Overall statistics
    pub overall: OverallStats,
    /// Per-task-type insights
    pub task_insights: Vec<TaskInsight>,
    /// Tasks ready for local-only execution
    pub graduation_candidates: Vec<GraduationCandidate>,
    /// Recommendations
    pub recommendations: Vec<Recommendation>,
}

impl Insights {
    /// Generate insights from the learning data
    pub fn generate(
        recorder: &LearningRecorder,
        patterns: &PatternStore,
        templates: &TemplateStore,
    ) -> Self {
        let overall = OverallStats::from_recorder(recorder);

        let task_insights: Vec<_> = patterns.all()
            .map(TaskInsight::from_pattern)
            .collect();

        let graduation_candidates = patterns.find_graduation_candidates();

        let recommendations = generate_recommendations(&overall, &task_insights, &graduation_candidates);

        Self {
            overall,
            task_insights,
            graduation_candidates,
            recommendations,
        }
    }
}

impl fmt::Display for Insights {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{}", "═══════════════════════════════════════════════════════════".bright_cyan())?;
        writeln!(f, "{}", "                    LEARNING INSIGHTS                       ".bright_cyan())?;
        writeln!(f, "{}", "═══════════════════════════════════════════════════════════".bright_cyan())?;

        // Overall stats
        writeln!(f, "\n{}", "📊 Overall Statistics".bright_white())?;
        writeln!(f, "   Total executions: {}", self.overall.total_executions)?;
        writeln!(f, "   Success rate: {:.1}%", self.overall.success_rate * 100.0)?;
        writeln!(f, "   Local execution rate: {:.1}%", self.overall.local_execution_rate * 100.0)?;
        writeln!(f, "   Local success rate: {:.1}%", self.overall.local_success_rate * 100.0)?;

        // Task insights
        if !self.task_insights.is_empty() {
            writeln!(f, "\n{}", "📋 Task Type Performance".bright_white())?;
            for insight in &self.task_insights {
                let status = if insight.local_success_rate > 0.8 {
                    "✓".green()
                } else if insight.local_success_rate > 0.5 {
                    "~".yellow()
                } else {
                    "✗".red()
                };

                writeln!(f, "   {} {} - {:.0}% local success ({} samples)",
                    status,
                    insight.task_type.bright_white(),
                    insight.local_success_rate * 100.0,
                    insight.total_attempts
                )?;
            }
        }

        // Graduation candidates
        if !self.graduation_candidates.is_empty() {
            writeln!(f, "\n{}", "🎓 Ready for Local-Only Execution".bright_green())?;
            for candidate in &self.graduation_candidates {
                writeln!(f, "   {} - {:.0}% success, {} samples, confidence: {:.2}",
                    candidate.task_type.bright_white(),
                    candidate.success_rate * 100.0,
                    candidate.sample_count,
                    candidate.confidence
                )?;
            }
        }

        // Recommendations
        if !self.recommendations.is_empty() {
            writeln!(f, "\n{}", "💡 Recommendations".bright_yellow())?;
            for rec in &self.recommendations {
                let icon = match rec.priority {
                    Priority::High => "❗",
                    Priority::Medium => "→",
                    Priority::Low => "·",
                };
                writeln!(f, "   {} {}", icon, rec.message)?;
            }
        }

        writeln!(f, "\n{}", "═══════════════════════════════════════════════════════════".bright_cyan())?;

        Ok(())
    }
}

/// Overall statistics
#[derive(Debug, Clone)]
pub struct OverallStats {
    pub total_executions: usize,
    pub successful_executions: usize,
    pub failed_executions: usize,
    pub success_rate: f32,
    pub local_execution_rate: f32,
    pub local_success_rate: f32,
    pub total_tasks: usize,
    pub local_tasks: usize,
    pub claude_tasks: usize,
}

impl OverallStats {
    pub fn from_recorder(recorder: &LearningRecorder) -> Self {
        let records = recorder.records();

        let total_executions = records.len();
        let successful_executions = recorder.successful_executions();
        let failed_executions = total_executions - successful_executions;

        let success_rate = if total_executions > 0 {
            successful_executions as f32 / total_executions as f32
        } else {
            0.0
        };

        let mut total_tasks = 0usize;
        let mut local_tasks = 0usize;
        let mut local_successes = 0usize;

        for record in records {
            for task in &record.task_executions {
                total_tasks += 1;
                if task.executed_locally {
                    local_tasks += 1;
                    if task.outcome.is_local_success() {
                        local_successes += 1;
                    }
                }
            }
        }

        let claude_tasks = total_tasks - local_tasks;

        let local_execution_rate = if total_tasks > 0 {
            local_tasks as f32 / total_tasks as f32
        } else {
            0.0
        };

        let local_success_rate = if local_tasks > 0 {
            local_successes as f32 / local_tasks as f32
        } else {
            0.0
        };

        Self {
            total_executions,
            successful_executions,
            failed_executions,
            success_rate,
            local_execution_rate,
            local_success_rate,
            total_tasks,
            local_tasks,
            claude_tasks,
        }
    }
}

/// Insight for a specific task type
#[derive(Debug, Clone)]
pub struct TaskInsight {
    pub task_type: String,
    pub total_attempts: usize,
    pub local_attempts: usize,
    pub local_successes: usize,
    pub local_failures: usize,
    pub local_success_rate: f32,
    pub common_tools: Vec<String>,
    pub avg_complexity: ComplexityLevel,
    pub recommendation: Option<String>,
}

impl TaskInsight {
    pub fn from_pattern(pattern: &Pattern) -> Self {
        let local_attempts = pattern.local_successes + pattern.local_failures;

        let recommendation = if pattern.is_graduation_ready() {
            Some("Ready for local-only execution".to_string())
        } else if pattern.local_success_rate() < 0.5 && local_attempts >= 3 {
            Some("Consider keeping with Claude".to_string())
        } else if local_attempts < 3 {
            Some("Need more samples".to_string())
        } else {
            None
        };

        let mut tools: Vec<_> = pattern.common_tools.iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        tools.sort_by(|a, b| b.1.cmp(&a.1));
        let common_tools = tools.into_iter().take(3).map(|(k, _)| k).collect();

        Self {
            task_type: pattern.task_type.clone(),
            total_attempts: pattern.total_count,
            local_attempts,
            local_successes: pattern.local_successes,
            local_failures: pattern.local_failures,
            local_success_rate: pattern.local_success_rate(),
            common_tools,
            avg_complexity: ComplexityLevel::from_steps(pattern.avg_local_complexity as usize),
            recommendation,
        }
    }
}

/// A recommendation from the system
#[derive(Debug, Clone)]
pub struct Recommendation {
    pub priority: Priority,
    pub message: String,
    pub action: Option<String>,
}

#[derive(Debug, Clone, Copy)]
pub enum Priority {
    High,
    Medium,
    Low,
}

/// Generate recommendations based on insights
fn generate_recommendations(
    overall: &OverallStats,
    task_insights: &[TaskInsight],
    graduation_candidates: &[GraduationCandidate],
) -> Vec<Recommendation> {
    let mut recommendations = Vec::new();

    // Check if local success rate is low
    if overall.local_success_rate < 0.6 && overall.local_tasks >= 5 {
        recommendations.push(Recommendation {
            priority: Priority::High,
            message: format!(
                "Local success rate is {:.0}%. Consider breaking tasks into smaller pieces.",
                overall.local_success_rate * 100.0
            ),
            action: Some("Use /plan to create more granular tasks".to_string()),
        });
    }

    // Highlight graduation candidates
    if !graduation_candidates.is_empty() {
        recommendations.push(Recommendation {
            priority: Priority::Medium,
            message: format!(
                "{} task type(s) ready for local-only execution",
                graduation_candidates.len()
            ),
            action: Some("Consider using templates for these tasks".to_string()),
        });
    }

    // Check for task types that always fail locally
    for insight in task_insights {
        if insight.local_failures >= 3 && insight.local_successes == 0 {
            recommendations.push(Recommendation {
                priority: Priority::Medium,
                message: format!(
                    "'{}' tasks always fail locally. Keep with Claude.",
                    insight.task_type
                ),
                action: None,
            });
        }
    }

    // Suggest more usage if not enough data
    if overall.total_executions < 10 {
        recommendations.push(Recommendation {
            priority: Priority::Low,
            message: "Not enough data for reliable insights yet. Keep using the system!".to_string(),
            action: None,
        });
    }

    // Celebrate high success rate
    if overall.local_success_rate > 0.9 && overall.local_tasks >= 10 {
        recommendations.push(Recommendation {
            priority: Priority::Low,
            message: format!(
                "Excellent! {:.0}% local success rate. Your local models are working well.",
                overall.local_success_rate * 100.0
            ),
            action: None,
        });
    }

    recommendations
}
