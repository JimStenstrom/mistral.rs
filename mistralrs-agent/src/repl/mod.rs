//! Interactive REPL interface

use crate::factory::WorkerFactory;
use crate::planner::Planner;
use crate::todo::TodoList;
use anyhow::Result;
use colored::Colorize;
use mistralrs_swarm::tools::ToolRegistry;
use rustyline::error::ReadlineError;
use rustyline::history::DefaultHistory;
use rustyline::Editor;
use std::sync::Arc;
use tokio::sync::Mutex;

/// The main agent runtime
pub struct AgentRuntime {
    model: Arc<mistralrs::Model>,
    planner: Planner,
    factory: WorkerFactory,
    todo_list: Arc<Mutex<TodoList>>,
}

impl AgentRuntime {
    pub async fn new(model: Arc<mistralrs::Model>, num_workers: usize) -> Result<Self> {
        let planner = Planner::new(model.clone());
        let tools = ToolRegistry::with_builtins();
        let factory = WorkerFactory::new(model.clone(), tools, num_workers);

        Ok(Self {
            model,
            planner,
            factory,
            todo_list: Arc::new(Mutex::new(TodoList::new())),
        })
    }

    /// Run the interactive REPL
    pub async fn run_repl(&self) -> Result<()> {
        let mut rl: Editor<(), DefaultHistory> = Editor::new()?;

        // Load history if available
        let history_path = dirs::data_dir()
            .map(|d| d.join("mistralrs-agent").join("history.txt"));

        if let Some(ref path) = history_path {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).ok();
            }
            rl.load_history(path).ok();
        }

        loop {
            let prompt = format!("{} ", "agent>".bright_cyan());
            let readline = rl.readline(&prompt);

            match readline {
                Ok(line) => {
                    let input = line.trim();
                    if input.is_empty() {
                        continue;
                    }

                    rl.add_history_entry(input).ok();

                    // Handle commands
                    if input.starts_with('/') {
                        match self.handle_command(input).await {
                            Ok(true) => continue,  // Command handled, continue
                            Ok(false) => break,    // Exit requested
                            Err(e) => {
                                println!("{} {}", "Error:".red(), e);
                                continue;
                            }
                        }
                    }

                    // Handle regular input
                    if let Err(e) = self.handle_input(input).await {
                        println!("{} {}", "Error:".red(), e);
                    }
                }
                Err(ReadlineError::Interrupted) => {
                    println!("{}", "Use /quit to exit".dimmed());
                    continue;
                }
                Err(ReadlineError::Eof) => {
                    break;
                }
                Err(e) => {
                    println!("{} {}", "Error:".red(), e);
                    break;
                }
            }
        }

        // Save history
        if let Some(ref path) = history_path {
            rl.save_history(path).ok();
        }

        println!("\n{}", "Goodbye!".bright_cyan());
        Ok(())
    }

    /// Handle a slash command
    async fn handle_command(&self, input: &str) -> Result<bool> {
        let parts: Vec<&str> = input.split_whitespace().collect();
        let cmd = parts[0];

        match cmd {
            "/help" | "/h" | "/?" => {
                self.print_help();
                Ok(true)
            }
            "/quit" | "/q" | "/exit" => {
                Ok(false)
            }
            "/todo" | "/t" => {
                let list = self.todo_list.lock().await;
                println!("\n{}", *list);
                Ok(true)
            }
            "/clear" | "/c" => {
                let mut list = self.todo_list.lock().await;
                list.clear();
                println!("{}", "Todo list cleared.".green());
                Ok(true)
            }
            "/plan" | "/p" => {
                if parts.len() < 2 {
                    println!("{}", "Usage: /plan <goal>".yellow());
                    return Ok(true);
                }
                let goal = parts[1..].join(" ");
                self.plan_goal(&goal).await?;
                Ok(true)
            }
            "/run" | "/r" => {
                self.run_workers().await?;
                Ok(true)
            }
            "/add" | "/a" => {
                if parts.len() < 2 {
                    println!("{}", "Usage: /add <task description>".yellow());
                    return Ok(true);
                }
                let desc = parts[1..].join(" ");
                let mut list = self.todo_list.lock().await;
                let id = list.add(&desc);
                println!("{} Added task [{}]: {}", "✓".green(), id, desc);
                Ok(true)
            }
            "/delegate" | "/d" => {
                if parts.len() < 2 {
                    println!("{}", "Usage: /delegate <task description>".yellow());
                    return Ok(true)
                }
                let desc = parts[1..].join(" ");
                let mut list = self.todo_list.lock().await;
                let id = list.add_delegated(&desc, "Execute this task autonomously");
                println!("{} Added delegated task [{}]: {}", "✓".green(), id, desc);
                Ok(true)
            }
            "/complete" => {
                if parts.len() < 2 {
                    println!("{}", "Usage: /complete <task_id>".yellow());
                    return Ok(true);
                }
                if let Ok(id) = parts[1].parse::<usize>() {
                    let mut list = self.todo_list.lock().await;
                    if list.complete(id, None) {
                        println!("{} Marked task [{}] as complete", "✓".green(), id);
                    } else {
                        println!("{} Task [{}] not found", "✗".red(), id);
                    }
                }
                Ok(true)
            }
            _ => {
                println!("{} Unknown command: {}", "?".yellow(), cmd);
                println!("Type {} for available commands", "/help".bright_cyan());
                Ok(true)
            }
        }
    }

    /// Handle regular (non-command) input
    async fn handle_input(&self, input: &str) -> Result<()> {
        // Check if we should plan or respond directly
        println!("{}", "Thinking...".dimmed());

        let should_plan = self.planner.should_plan(input).await?;

        if should_plan {
            println!("{}", "This looks like a complex task. Creating a plan...".yellow());
            self.plan_goal(input).await?;

            // Ask if user wants to execute
            println!("\n{}", "Would you like to execute this plan? Use /run to start workers.".dimmed());
        } else {
            // Direct response
            let response = self.planner.quick_response(input).await?;
            println!("\n{}", response);
        }

        Ok(())
    }

    /// Plan a goal and add to todo list
    async fn plan_goal(&self, goal: &str) -> Result<()> {
        println!("{} {}", "Planning:".yellow(), goal);

        let mut list = self.todo_list.lock().await;
        let count = self.planner.plan_to_todo(goal, &mut list).await?;

        println!("{} Created {} tasks\n", "✓".green(), count);
        println!("{}", *list);

        Ok(())
    }

    /// Run workers on delegated tasks
    async fn run_workers(&self) -> Result<()> {
        let results = self.factory.execute_todo_list(self.todo_list.clone()).await;

        if results.is_empty() {
            return Ok(());
        }

        // Print results summary
        println!("\n{}", "═══ Execution Results ═══".bright_cyan());

        for result in &results {
            let status = if result.success {
                "✓".green()
            } else {
                "✗".red()
            };

            println!("\n{} Task [{}] ({} ms)", status, result.task_id, result.duration_ms);

            if !result.tool_calls.is_empty() {
                println!("  Tools used: {}",
                    result.tool_calls.iter()
                        .map(|(name, ok)| {
                            if *ok {
                                name.green().to_string()
                            } else {
                                name.red().to_string()
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }

            // Print truncated output
            let output = if result.output.len() > 500 {
                format!("{}...", &result.output[..500])
            } else {
                result.output.clone()
            };
            println!("  Output: {}", output.dimmed());
        }

        // Print final todo list status
        println!();
        let list = self.todo_list.lock().await;
        println!("{}", *list);

        Ok(())
    }

    /// Print help
    fn print_help(&self) {
        println!();
        println!("{}", "═══ Commands ═══".bright_cyan());
        println!();
        println!("  {}     Show this help", "/help".bright_cyan());
        println!("  {}     Show current todo list", "/todo".bright_cyan());
        println!("  {}    Clear todo list", "/clear".bright_cyan());
        println!("  {} Plan tasks for a goal", "/plan <goal>".bright_cyan());
        println!("  {}      Execute delegated tasks with workers", "/run".bright_cyan());
        println!("  {} Add a task to the list", "/add <task>".bright_cyan());
        println!("  {} Add a delegated task (for workers)", "/delegate <task>".bright_cyan());
        println!("  {} Mark a task complete", "/complete <id>".bright_cyan());
        println!("  {}     Exit", "/quit".bright_cyan());
        println!();
        println!("{}", "═══ Usage ═══".bright_cyan());
        println!();
        println!("  Just type your request and the agent will either:");
        println!("  - Answer directly for simple questions");
        println!("  - Create a plan for complex tasks");
        println!();
        println!("  Use {} to execute delegated tasks in parallel.", "/run".bright_cyan());
        println!();
    }
}
