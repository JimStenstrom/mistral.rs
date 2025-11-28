//! Example: Code Analysis Swarm
//!
//! This example demonstrates using the swarm to analyze a codebase:
//! - Claude Opus 4.5 orchestrates the analysis
//! - Local workers (Qwen 8B) analyze individual files/components
//!
//! Run with:
//! ```bash
//! ANTHROPIC_API_KEY=your-key cargo run --example code_analysis --features cuda
//! ```

use anyhow::Result;
use mistralrs_swarm::{ClaudeConfig, Swarm, SwarmUpdate, ToolRegistry, WorkerConfig};
use tokio::sync::mpsc;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::from_default_env().add_directive("mistralrs_swarm=info".parse()?))
        .init();

    // Get API key from environment
    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .expect("ANTHROPIC_API_KEY environment variable must be set");

    println!("=== Code Analysis Swarm ===\n");
    println!("Building swarm with Claude orchestrator and local workers...\n");

    // Build the swarm
    let mut swarm = Swarm::builder()
        .with_claude_orchestrator(ClaudeConfig {
            api_key,
            model: "claude-opus-4-5-20250929".to_string(),
            max_tokens: 4096,
            ..Default::default()
        })
        .with_workers(WorkerConfig {
            model_id: "Qwen/Qwen3-8B".to_string(),
            num_workers: 8,
            quantization: Some("Q4K".to_string()),
            gpu_memory_mb: 16000,
            max_steps: 15,
            ..Default::default()
        })
        .with_tools(ToolRegistry::with_builtins())
        .build()
        .await?;

    // Create update channel for progress tracking
    let (update_tx, mut update_rx) = mpsc::channel(100);

    // Spawn task to print updates
    tokio::spawn(async move {
        while let Some(update) = update_rx.recv().await {
            match update {
                SwarmUpdate::Planning(goal) => {
                    println!("[ORCHESTRATOR] Planning: {}", goal);
                }
                SwarmUpdate::TasksCreated(n) => {
                    println!("[ORCHESTRATOR] Created {} tasks for workers", n);
                }
                SwarmUpdate::TaskDispatched(id) => {
                    println!("[DISPATCH] Task {} sent to worker pool", id);
                }
                SwarmUpdate::TaskInProgress { task_id, step } => {
                    println!("[WORKER] Task {} - step {}", task_id, step);
                }
                SwarmUpdate::TaskCompleted(id) => {
                    println!("[COMPLETE] Task {} finished", id);
                }
                SwarmUpdate::Synthesizing => {
                    println!("[ORCHESTRATOR] Synthesizing results...");
                }
                SwarmUpdate::Complete => {
                    println!("[DONE] Swarm execution complete");
                }
                SwarmUpdate::Error(e) => {
                    eprintln!("[ERROR] {}", e);
                }
            }
        }
    });

    // Define the goal
    let goal = r#"
    Analyze the mistralrs-swarm crate structure:
    1. List all source files and their purposes
    2. Identify the main components (orchestrator, worker, tools)
    3. Document the public API
    4. Find any potential issues or improvements
    "#;

    println!("Goal: {}\n", goal.trim());
    println!("Starting swarm execution...\n");

    // Execute with streaming updates
    let result = swarm.execute_streaming(goal, update_tx).await?;

    println!("\n=== Analysis Result ===\n");
    println!("{}", result);

    Ok(())
}
