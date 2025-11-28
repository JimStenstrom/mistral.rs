# mistralrs-agent Roadmap

## The Gap: What Claude Can Do That Local Models Cannot (Yet)

This document outlines what needs to be built to make local agentic systems approach the capabilities of frontier models like Claude.

---

## 1. Memory & RAG System

**Problem**: Every request starts from scratch. No memory of previous conversations, no understanding of the codebase structure.

**Solution**: Implement a retrieval-augmented generation system.

```rust
pub struct Memory {
    // Long-term storage
    conversation_history: Vec<ConversationTurn>,
    codebase_index: CodebaseIndex,
    learned_preferences: UserPreferences,

    // Short-term working memory
    current_context: ContextWindow,
    relevant_files: Vec<FileContext>,
}

pub struct CodebaseIndex {
    // Semantic embeddings of code chunks
    embeddings: VectorStore,
    // File dependency graph
    dependency_graph: DependencyGraph,
    // Symbol table (functions, classes, etc.)
    symbols: SymbolTable,
}

impl Memory {
    /// Retrieve relevant context for a query
    pub async fn retrieve(&self, query: &str, max_tokens: usize) -> Context {
        // 1. Semantic search over codebase
        let relevant_chunks = self.embeddings.search(query, top_k=20);

        // 2. Expand with dependencies
        let expanded = self.dependency_graph.expand(relevant_chunks);

        // 3. Fit within token budget
        self.fit_to_budget(expanded, max_tokens)
    }

    /// Learn from this interaction
    pub fn learn(&mut self, interaction: &Interaction) {
        // Store successful patterns
        // Update preferences
        // Index new/changed files
    }
}
```

**Why it matters**: With 8K context, you must be surgical about what to include. RAG lets you find the right context automatically.

---

## 2. Self-Reflection & Verification Loop

**Problem**: Local models are confidently wrong. They don't check their work.

**Solution**: Implement a reflection loop after each action.

```rust
pub struct ReflectionLoop {
    verifiers: Vec<Box<dyn Verifier>>,
}

#[async_trait]
pub trait Verifier: Send + Sync {
    async fn verify(&self, action: &Action, result: &ActionResult) -> VerificationResult;
}

// Built-in verifiers
pub struct SyntaxVerifier;      // Does the code parse?
pub struct TestVerifier;        // Do tests pass?
pub struct TypeCheckVerifier;   // Does it type-check?
pub struct DiffVerifier;        // Is the diff reasonable?
pub struct SecurityVerifier;    // Any obvious vulnerabilities?

impl ReflectionLoop {
    pub async fn verify_and_correct(
        &self,
        action: &Action,
        result: &ActionResult,
        model: &Model,
        max_retries: usize,
    ) -> Result<ActionResult> {
        for attempt in 0..max_retries {
            // Run all verifiers
            let mut issues = Vec::new();
            for verifier in &self.verifiers {
                if let Err(issue) = verifier.verify(action, result).await {
                    issues.push(issue);
                }
            }

            if issues.is_empty() {
                return Ok(result);
            }

            // Ask model to fix issues
            let fix_prompt = format!(
                "Your previous action had these issues:\n{}\n\nPlease fix them.",
                issues.join("\n")
            );

            result = model.fix(action, &fix_prompt).await?;
        }

        Err(anyhow!("Could not fix issues after {} attempts", max_retries))
    }
}
```

**Why it matters**: Claude checks its work mentally. Local models need explicit verification.

---

## 3. Confidence Scoring & Escalation

**Problem**: Local models don't know what they don't know.

**Solution**: Implement uncertainty quantification and human escalation.

```rust
pub struct ConfidenceScorer {
    // Use logprobs, consistency checks, etc.
}

impl ConfidenceScorer {
    pub async fn score(&self, model: &Model, prompt: &str) -> f32 {
        // Method 1: Sample multiple times, check consistency
        let samples = model.sample_n(prompt, n=5).await;
        let consistency = self.measure_consistency(&samples);

        // Method 2: Check logprobs if available
        let logprob_confidence = model.get_logprob_confidence(prompt).await;

        // Method 3: Ask the model directly
        let self_assessed = model.ask("How confident are you? 0-100").await;

        // Combine signals
        (consistency + logprob_confidence + self_assessed) / 3.0
    }
}

pub enum Decision {
    Proceed(Action),
    AskHuman(Question),
    Abort(Reason),
}

pub async fn decide_with_confidence(
    action: Action,
    confidence: f32,
    risk_level: RiskLevel,
) -> Decision {
    match (confidence, risk_level) {
        (c, _) if c > 0.9 => Decision::Proceed(action),
        (c, RiskLevel::Low) if c > 0.7 => Decision::Proceed(action),
        (c, RiskLevel::High) => Decision::AskHuman(
            format!("I'm {}% confident about: {}. Proceed?", c*100, action)
        ),
        _ => Decision::AskHuman(...)
    }
}
```

**Why it matters**: I naturally express uncertainty. Local models need explicit mechanisms.

---

## 4. Smart Context Management

**Problem**: 8K tokens isn't enough. Must be surgical about what to include.

**Solution**: Dynamic context assembly based on task.

```rust
pub struct ContextManager {
    budget: usize,  // Max tokens
    strategy: ContextStrategy,
}

pub enum ContextStrategy {
    // For code changes: include file + imports + tests
    CodeChange {
        include_imports: bool,
        include_tests: bool,
        include_callers: bool,
    },
    // For debugging: include stack trace context
    Debugging {
        stack_depth: usize,
        include_logs: bool,
    },
    // For understanding: broader but shallower
    Understanding {
        breadth: usize,
        depth: usize,
    },
}

impl ContextManager {
    pub async fn assemble(&self, task: &Task, codebase: &CodebaseIndex) -> Context {
        let mut context = Context::new(self.budget);

        // Always include: task description, system prompt
        context.add_required(&task.description);
        context.add_required(&self.system_prompt);

        // Strategy-specific additions
        match &self.strategy {
            ContextStrategy::CodeChange { include_imports, .. } => {
                // Add target file
                context.add_priority(&task.target_file, Priority::High);

                // Add imports if they fit
                if *include_imports {
                    for import in codebase.get_imports(&task.target_file) {
                        context.add_if_fits(&import, Priority::Medium);
                    }
                }

                // Add tests if they fit
                // ... etc
            }
            // ... other strategies
        }

        context
    }
}
```

**Why it matters**: I have 200K tokens. Local models must be smart about their 8K.

---

## 5. Multi-Model Routing

**Problem**: One model size doesn't fit all tasks.

**Solution**: Route tasks to appropriate models.

```rust
pub struct ModelRouter {
    small: Model,   // Qwen 1.5B - fast, simple tasks
    medium: Model,  // Qwen 8B - most tasks
    large: Model,   // Qwen 32B or external API - complex reasoning
}

impl ModelRouter {
    pub async fn route(&self, task: &Task) -> &Model {
        // Classify task complexity
        let complexity = self.classify_complexity(task).await;

        match complexity {
            Complexity::Simple => &self.small,   // "What's in this file?"
            Complexity::Medium => &self.medium,  // "Refactor this function"
            Complexity::Complex => &self.large,  // "Design a new architecture"
        }
    }

    async fn classify_complexity(&self, task: &Task) -> Complexity {
        // Use small model to classify (fast)
        let prompt = format!(
            "Rate complexity 1-3:\n1=simple question\n2=code change\n3=complex reasoning\n\nTask: {}",
            task.description
        );

        let response = self.small.quick_response(&prompt).await;
        // Parse response...
    }
}
```

**Why it matters**: Don't use 8B params to answer "what time is it?"

---

## 6. Structured Output Enforcement

**Problem**: Local models often produce malformed JSON, incomplete tool calls.

**Solution**: Grammar-constrained generation + retry logic.

```rust
pub struct StructuredOutput {
    schema: JsonSchema,
    retries: usize,
}

impl StructuredOutput {
    pub async fn generate<T: DeserializeOwned>(
        &self,
        model: &Model,
        prompt: &str,
    ) -> Result<T> {
        for attempt in 0..self.retries {
            // Use grammar-constrained generation if available
            let response = model.generate_with_grammar(
                prompt,
                &self.schema.to_grammar()
            ).await?;

            // Try to parse
            match serde_json::from_str::<T>(&response) {
                Ok(parsed) => return Ok(parsed),
                Err(e) => {
                    // Ask model to fix
                    let fix_prompt = format!(
                        "Your JSON was invalid: {}\n\nOriginal: {}\n\nPlease fix:",
                        e, response
                    );
                    // ... retry
                }
            }
        }

        Err(anyhow!("Could not generate valid output"))
    }
}
```

**Why it matters**: Tool calling fails silently with bad JSON. Must be robust.

---

## 7. Human-in-the-Loop Checkpoints

**Problem**: Agents can go off the rails. Need human oversight.

**Solution**: Approval gates for risky operations.

```rust
pub enum ApprovalRequired {
    Never,
    Always,
    WhenRisky,
    WhenUncertain(f32),  // Below this confidence
}

pub struct ApprovalGate {
    policy: ApprovalRequired,
    risky_patterns: Vec<Regex>,  // rm -rf, DROP TABLE, etc.
}

impl ApprovalGate {
    pub async fn check(&self, action: &Action, confidence: f32) -> GateResult {
        let is_risky = self.risky_patterns.iter().any(|p| p.is_match(&action.command));

        let needs_approval = match &self.policy {
            ApprovalRequired::Never => false,
            ApprovalRequired::Always => true,
            ApprovalRequired::WhenRisky => is_risky,
            ApprovalRequired::WhenUncertain(threshold) => confidence < *threshold,
        };

        if needs_approval {
            // Show diff, explain action, ask for approval
            GateResult::NeedsApproval(ApprovalRequest {
                action: action.clone(),
                reason: if is_risky { "Risky operation" } else { "Low confidence" },
                diff: action.preview_diff(),
            })
        } else {
            GateResult::Approved
        }
    }
}
```

**Why it matters**: I ask before `rm -rf`. Local agents should too.

---

## 8. Learning & Adaptation

**Problem**: Every session starts fresh. No learning from mistakes.

**Solution**: Store successful patterns, avoid past failures.

```rust
pub struct LearningSystem {
    successful_patterns: PatternStore,
    failed_patterns: PatternStore,
    user_preferences: PreferenceStore,
}

impl LearningSystem {
    pub fn record_outcome(&mut self, task: &Task, plan: &Plan, outcome: Outcome) {
        match outcome {
            Outcome::Success => {
                // Store this plan pattern for similar future tasks
                self.successful_patterns.add(task.signature(), plan.clone());
            }
            Outcome::Failure(reason) => {
                // Remember to avoid this approach
                self.failed_patterns.add(task.signature(), FailedAttempt {
                    plan: plan.clone(),
                    reason,
                });
            }
        }
    }

    pub fn suggest_approach(&self, task: &Task) -> Option<Plan> {
        // Check if we've solved similar tasks before
        if let Some(pattern) = self.successful_patterns.find_similar(task) {
            return Some(pattern.adapt_to(task));
        }
        None
    }

    pub fn approaches_to_avoid(&self, task: &Task) -> Vec<FailedAttempt> {
        self.failed_patterns.find_similar(task)
    }
}
```

**Why it matters**: Humans learn from mistakes. Agents should too.

---

## Priority Implementation Order

1. **Self-Reflection Loop** - Catch mistakes immediately (highest impact)
2. **Smart Context Management** - Make 8K tokens work (necessary for quality)
3. **Confidence Scoring** - Know when to ask for help (safety)
4. **Memory/RAG** - Remember context across turns (usability)
5. **Human Approval Gates** - Prevent disasters (safety)
6. **Structured Output** - Reliable tool calling (reliability)
7. **Multi-Model Routing** - Efficiency (performance)
8. **Learning System** - Long-term improvement (future)

---

## The Honest Truth

Even with all these systems, a local 8B model won't match Claude's reasoning on complex tasks. The goal isn't to replicate Claude—it's to:

1. **Handle 80% of tasks locally** (simple edits, searches, tests)
2. **Know when to escalate** the hard 20% to a human or larger model
3. **Be reliable** even if not brilliant
4. **Be fast** for interactive use
5. **Work offline** without API costs

The architecture should be: **Local-first, with intelligent escalation.**
