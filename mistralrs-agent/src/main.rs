//! mistralrs-agent - Interactive agentic coding assistant
//!
//! A local-first CLI tool for agentic coding tasks using mistral.rs inference.
//! Features:
//! - Interactive REPL interface
//! - Todo list planning and tracking
//! - Worker factory for parallel task execution
//! - Tool calling for file operations, code execution, etc.

mod factory;
mod planner;
mod repl;
mod todo;

use anyhow::Result;
use clap::Parser;
use colored::Colorize;
use mistralrs::{IsqType, MemoryGpuConfig, PagedAttentionMetaBuilder, TextModelBuilder};
use std::sync::Arc;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

#[derive(Parser, Debug)]
#[command(name = "mistralrs-agent")]
#[command(about = "Interactive agentic coding assistant using local LLM inference")]
#[command(version)]
struct Args {
    /// Model ID to use (HuggingFace format)
    #[arg(short, long, default_value = "Qwen/Qwen3-8B")]
    model: String,

    /// Quantization level (Q2K, Q3K, Q4K, Q5K, Q6K, Q8_0)
    #[arg(short, long, default_value = "Q4K")]
    quantization: String,

    /// Number of worker agents for parallel tasks
    #[arg(short, long, default_value = "4")]
    workers: usize,

    /// GPU memory for KV cache in MB
    #[arg(long, default_value = "16000")]
    gpu_memory: usize,

    /// Working directory
    #[arg(short = 'd', long)]
    work_dir: Option<String>,

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Setup logging
    let filter = if args.verbose {
        "mistralrs_agent=debug,mistralrs=info"
    } else {
        "mistralrs_agent=info,mistralrs=warn"
    };

    tracing_subscriber::registry()
        .with(fmt::layer().without_time().with_target(false))
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| filter.parse().unwrap()))
        .init();

    // Print banner
    println!();
    println!("{}", "╔═══════════════════════════════════════════════════════╗".bright_cyan());
    println!("{}", "║           mistralrs-agent v0.6.0                      ║".bright_cyan());
    println!("{}", "║     Local Agentic Coding Assistant                    ║".bright_cyan());
    println!("{}", "╚═══════════════════════════════════════════════════════╝".bright_cyan());
    println!();

    // Set working directory
    if let Some(ref dir) = args.work_dir {
        std::env::set_current_dir(dir)?;
    }
    println!("{} {}", "Working directory:".dimmed(), std::env::current_dir()?.display());
    println!();

    // Build the model
    println!("{}", "Loading model...".yellow());
    println!("  Model: {}", args.model.bright_white());
    println!("  Quantization: {}", args.quantization.bright_white());
    println!("  Workers: {}", args.workers.to_string().bright_white());
    println!();

    let isq = match args.quantization.to_uppercase().as_str() {
        "Q2K" => IsqType::Q2K,
        "Q3K" => IsqType::Q3K,
        "Q4K" => IsqType::Q4K,
        "Q5K" => IsqType::Q5K,
        "Q6K" => IsqType::Q6K,
        "Q8_0" => IsqType::Q8_0,
        _ => {
            eprintln!("Unknown quantization '{}', using Q4K", args.quantization);
            IsqType::Q4K
        }
    };

    let model = TextModelBuilder::new(&args.model)
        .with_isq(isq)
        .with_logging()
        .with_paged_attn(|| {
            PagedAttentionMetaBuilder::default()
                .with_gpu_memory(MemoryGpuConfig::MbAmount(args.gpu_memory))
                .with_block_size(32)
                .build()
        })?
        .build()
        .await?;

    let model = Arc::new(model);

    println!("{}", "Model loaded successfully!".green());
    println!();

    // Create the agent runtime
    let runtime = repl::AgentRuntime::new(model, args.workers).await?;

    // Start the REPL
    println!("{}", "Type your request or use commands:".dimmed());
    println!("  {}  - Show this help", "/help".bright_cyan());
    println!("  {}  - Show current todo list", "/todo".bright_cyan());
    println!("  {} - Clear todo list", "/clear".bright_cyan());
    println!("  {}  - Plan tasks for a goal", "/plan".bright_cyan());
    println!("  {}   - Execute todo list with workers", "/run".bright_cyan());
    println!("  {}  - Exit", "/quit".bright_cyan());
    println!();

    runtime.run_repl().await?;

    Ok(())
}
