# mistralrs-swarm

A high-throughput agentic swarm system that uses **Claude Opus 4.5** as an orchestrator and **mistral.rs** for local LLM inference on worker agents.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CLAUDE ORCHESTRATOR                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │  Planner │→ │Decomposer│→ │Dispatcher│→ │Synthesize│        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
└─────────────────────────┬───────────────────────────────────────┘
                          │ Task Queue (async)
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │ Worker 1 │    │ Worker 2 │    │ Worker N │  (32+ workers)
    │  Agent   │    │  Agent   │    │  Agent   │
    └────┬─────┘    └────┬─────┘    └────┬─────┘
         │               │               │
         └───────────────┼───────────────┘
                         ▼
              ┌─────────────────────┐
              │   mistral.rs Engine │
              │  (Local Inference)  │
              │  Continuous Batch   │
              └─────────────────────┘
```

## Key Features

- **Claude Orchestrator**: Uses Claude Opus 4.5 for high-level planning, task decomposition, and result synthesis
- **Local Worker Agents**: Runs on mistral.rs with automatic batching for GPU saturation
- **Tool System**: Extensible tool registry for file operations, code execution, search
- **Async-first**: Built on Tokio for maximum concurrency
- **Continuous Batching**: Workers share a single model instance, requests are batched automatically

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
mistralrs-swarm = { path = "mistralrs-swarm" }

# Or with CUDA support
mistralrs-swarm = { path = "mistralrs-swarm", features = ["cuda"] }
```

## Quick Start

```rust
use mistralrs_swarm::{Swarm, ClaudeConfig, WorkerConfig, ToolRegistry};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let swarm = Swarm::builder()
        .with_claude_orchestrator(ClaudeConfig {
            api_key: std::env::var("ANTHROPIC_API_KEY")?,
            model: "claude-opus-4-5-20250929".to_string(),
            ..Default::default()
        })
        .with_workers(WorkerConfig {
            model_id: "Qwen/Qwen3-8B".to_string(),
            num_workers: 8,
            quantization: Some("Q4K".to_string()),
            ..Default::default()
        })
        .with_tools(ToolRegistry::with_builtins())
        .build()
        .await?;

    let result = swarm.execute("Analyze this codebase and find security issues").await?;
    println!("{}", result);
    Ok(())
}
```

## Configuration

### Claude Orchestrator

```rust
ClaudeConfig {
    api_key: String,           // Required: Anthropic API key
    model: String,             // Default: "claude-opus-4-5-20250929"
    max_tokens: usize,         // Default: 4096
    base_url: String,          // Default: "https://api.anthropic.com"
}
```

### Worker Agents

```rust
WorkerConfig {
    model_id: String,          // HuggingFace model ID (e.g., "Qwen/Qwen3-8B")
    num_workers: usize,        // Number of concurrent workers (default: 8)
    quantization: Option<String>, // Quantization level (e.g., "Q4K", "Q8_0")
    gpu_memory_mb: usize,      // GPU memory for KV cache (default: 16000)
    max_steps: usize,          // Max reasoning steps per task (default: 15)
    system_prompt: String,     // Custom system prompt for workers
}
```

## Built-in Tools

The `ToolRegistry::with_builtins()` includes:

| Tool | Description |
|------|-------------|
| `read_file` | Read file contents |
| `write_file` | Write content to a file |
| `list_directory` | List directory contents |
| `search_files` | Search for files using glob patterns |
| `execute_bash` | Execute safe bash commands |
| `grep` | Search file contents with regex |

## Custom Tools

```rust
use mistralrs_swarm::{Tool, ToolResult};
use async_trait::async_trait;

struct MyCustomTool;

#[async_trait]
impl Tool for MyCustomTool {
    fn name(&self) -> &str { "my_tool" }
    fn description(&self) -> &str { "Does something custom" }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "input": { "type": "string" }
            }
        })
    }

    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        // Your implementation
        Ok(ToolResult::success("Done!"))
    }
}

// Register
let mut registry = ToolRegistry::with_builtins();
registry.register(Arc::new(MyCustomTool));
```

## Examples

Run examples with:

```bash
# Simple swarm
ANTHROPIC_API_KEY=your-key cargo run --example simple_swarm --features cuda

# Code analysis
ANTHROPIC_API_KEY=your-key cargo run --example code_analysis --features cuda
```

## How It Works

1. **Goal Submission**: You submit a high-level goal to the swarm
2. **Planning**: Claude analyzes the goal and creates a plan
3. **Decomposition**: Claude breaks the plan into independent, parallelizable tasks
4. **Dispatch**: Tasks are dispatched to the worker pool
5. **Execution**: Workers run agent loops using local LLM for reasoning + tools for actions
6. **Batching**: mistral.rs automatically batches concurrent LLM requests for GPU efficiency
7. **Collection**: Results are collected as tasks complete
8. **Synthesis**: Claude synthesizes all results into a final coherent answer

## Performance Tips

- **Worker Count**: More workers = more concurrent tasks, but each needs KV cache memory
- **Model Size**: Smaller models (4B-8B) work well for workers; they process specific tasks
- **Quantization**: Q4K provides good balance of quality/memory; Q8_0 for better quality
- **GPU Memory**: Allocate 80-90% of VRAM to KV cache for maximum concurrency

## Syncing with Upstream mistral.rs

This is a fork of mistral.rs. To sync with upstream:

```bash
# Fetch latest
git fetch upstream master

# Merge or rebase
git merge upstream/master
# or
git rebase upstream/master
```

## License

MIT License - same as mistral.rs
