//! Simple Swarm Example
//!
//! Demonstrates basic swarm usage with a simple task.
//!
//! Run with:
//! ```bash
//! ANTHROPIC_API_KEY=your-key cargo run --example simple_swarm --features cuda
//! ```

use anyhow::Result;
use mistralrs_swarm::{ClaudeConfig, Swarm, ToolRegistry, WorkerConfig};

#[tokio::main]
async fn main() -> Result<()> {
    // Get API key from environment
    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .expect("ANTHROPIC_API_KEY environment variable must be set");

    println!("Building swarm...");

    // Build with minimal configuration
    let mut swarm = Swarm::builder()
        .with_claude_orchestrator(ClaudeConfig {
            api_key,
            ..Default::default()
        })
        .with_workers(WorkerConfig {
            model_id: "Qwen/Qwen3-4B".to_string(), // Smaller model for quick testing
            num_workers: 4,
            quantization: Some("Q8_0".to_string()),
            ..Default::default()
        })
        .with_tools(ToolRegistry::with_builtins())
        .build()
        .await?;

    println!("Swarm ready!\n");

    // Simple task
    let goal = "List the files in the current directory and summarize what this project does based on the file structure.";

    println!("Goal: {}\n", goal);
    println!("Executing...\n");

    let result = swarm.execute(goal).await?;

    println!("=== Result ===\n");
    println!("{}", result);

    Ok(())
}
