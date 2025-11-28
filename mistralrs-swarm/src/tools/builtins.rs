//! Built-in tools for common operations

use super::{Tool, ToolResult};
use anyhow::Result;
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::Path;
use std::process::Stdio;
use tokio::fs;
use tokio::process::Command;

/// Read file contents
pub struct ReadFileTool;

#[derive(Deserialize)]
struct ReadFileArgs {
    path: String,
    #[serde(default)]
    max_lines: Option<usize>,
    #[serde(default)]
    offset: Option<usize>,
}

#[async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Read the contents of a file. Returns the file content as text."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read"
                },
                "max_lines": {
                    "type": "integer",
                    "description": "Maximum number of lines to read (optional)"
                },
                "offset": {
                    "type": "integer",
                    "description": "Line number to start reading from (0-indexed, optional)"
                }
            },
            "required": ["path"]
        })
    }

    async fn execute(&self, arguments: Value) -> Result<ToolResult> {
        let args: ReadFileArgs = serde_json::from_value(arguments)?;

        let content = match fs::read_to_string(&args.path).await {
            Ok(c) => c,
            Err(e) => return Ok(ToolResult::error(format!("Failed to read file: {}", e))),
        };

        let lines: Vec<&str> = content.lines().collect();
        let offset = args.offset.unwrap_or(0);
        let max_lines = args.max_lines.unwrap_or(lines.len());

        let selected: Vec<&str> = lines.into_iter().skip(offset).take(max_lines).collect();

        Ok(ToolResult::success(selected.join("\n")))
    }

    fn timeout_secs(&self) -> u64 {
        10
    }
}

/// Write content to a file
pub struct WriteFileTool;

#[derive(Deserialize)]
struct WriteFileArgs {
    path: String,
    content: String,
    #[serde(default)]
    append: bool,
}

#[async_trait]
impl Tool for WriteFileTool {
    fn name(&self) -> &str {
        "write_file"
    }

    fn description(&self) -> &str {
        "Write content to a file. Creates the file if it doesn't exist."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                },
                "append": {
                    "type": "boolean",
                    "description": "If true, append to the file instead of overwriting"
                }
            },
            "required": ["path", "content"]
        })
    }

    async fn execute(&self, arguments: Value) -> Result<ToolResult> {
        let args: WriteFileArgs = serde_json::from_value(arguments)?;

        // Create parent directories if needed
        if let Some(parent) = Path::new(&args.path).parent() {
            if !parent.exists() {
                if let Err(e) = fs::create_dir_all(parent).await {
                    return Ok(ToolResult::error(format!(
                        "Failed to create directories: {}",
                        e
                    )));
                }
            }
        }

        let result = if args.append {
            use tokio::io::AsyncWriteExt;
            let mut file = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&args.path)
                .await?;
            file.write_all(args.content.as_bytes()).await
        } else {
            fs::write(&args.path, &args.content).await
        };

        match result {
            Ok(_) => Ok(ToolResult::success(format!(
                "Successfully wrote {} bytes to {}",
                args.content.len(),
                args.path
            ))),
            Err(e) => Ok(ToolResult::error(format!("Failed to write file: {}", e))),
        }
    }

    fn timeout_secs(&self) -> u64 {
        10
    }
}

/// List directory contents
pub struct ListDirectoryTool;

#[derive(Deserialize)]
struct ListDirArgs {
    path: String,
    #[serde(default)]
    recursive: bool,
    #[serde(default)]
    max_depth: Option<usize>,
}

#[async_trait]
impl Tool for ListDirectoryTool {
    fn name(&self) -> &str {
        "list_directory"
    }

    fn description(&self) -> &str {
        "List files and directories in a given path."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to list"
                },
                "recursive": {
                    "type": "boolean",
                    "description": "If true, list recursively"
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum recursion depth (only used if recursive is true)"
                }
            },
            "required": ["path"]
        })
    }

    async fn execute(&self, arguments: Value) -> Result<ToolResult> {
        let args: ListDirArgs = serde_json::from_value(arguments)?;

        let mut entries = Vec::new();

        if args.recursive {
            list_recursive(&args.path, 0, args.max_depth.unwrap_or(3), &mut entries).await?;
        } else {
            let mut dir = match fs::read_dir(&args.path).await {
                Ok(d) => d,
                Err(e) => {
                    return Ok(ToolResult::error(format!(
                        "Failed to read directory: {}",
                        e
                    )))
                }
            };

            while let Some(entry) = dir.next_entry().await? {
                let file_type = entry.file_type().await?;
                let prefix = if file_type.is_dir() { "d " } else { "f " };
                entries.push(format!("{}{}", prefix, entry.path().display()));
            }
        }

        entries.sort();
        Ok(ToolResult::success(entries.join("\n")))
    }

    fn timeout_secs(&self) -> u64 {
        30
    }
}

#[async_recursion::async_recursion]
async fn list_recursive(
    path: &str,
    depth: usize,
    max_depth: usize,
    entries: &mut Vec<String>,
) -> Result<()> {
    if depth > max_depth {
        return Ok(());
    }

    let mut dir = fs::read_dir(path).await?;
    while let Some(entry) = dir.next_entry().await? {
        let file_type = entry.file_type().await?;
        let indent = "  ".repeat(depth);
        let prefix = if file_type.is_dir() { "d " } else { "f " };
        entries.push(format!("{}{}{}", indent, prefix, entry.path().display()));

        if file_type.is_dir() {
            if let Err(_) = list_recursive(
                &entry.path().to_string_lossy(),
                depth + 1,
                max_depth,
                entries,
            )
            .await
            {
                // Skip directories we can't read
            }
        }
    }
    Ok(())
}

/// Search for files matching a pattern
pub struct SearchFilesTool;

#[derive(Deserialize)]
struct SearchFilesArgs {
    pattern: String,
    #[serde(default = "default_search_path")]
    path: String,
}

fn default_search_path() -> String {
    ".".to_string()
}

#[async_trait]
impl Tool for SearchFilesTool {
    fn name(&self) -> &str {
        "search_files"
    }

    fn description(&self) -> &str {
        "Search for files matching a glob pattern (e.g., '**/*.rs')."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match files (e.g., '**/*.rs', 'src/**/*.py')"
                },
                "path": {
                    "type": "string",
                    "description": "Base directory to search in (default: current directory)"
                }
            },
            "required": ["pattern"]
        })
    }

    async fn execute(&self, arguments: Value) -> Result<ToolResult> {
        let args: SearchFilesArgs = serde_json::from_value(arguments)?;

        // Use glob crate for pattern matching
        let full_pattern = format!("{}/{}", args.path, args.pattern);
        let paths: Vec<String> = glob::glob(&full_pattern)
            .map_err(|e| anyhow::anyhow!("Invalid glob pattern: {}", e))?
            .filter_map(|entry| entry.ok())
            .map(|path| path.display().to_string())
            .take(100) // Limit results
            .collect();

        if paths.is_empty() {
            Ok(ToolResult::success("No files found matching pattern"))
        } else {
            Ok(ToolResult::success(paths.join("\n")))
        }
    }

    fn timeout_secs(&self) -> u64 {
        30
    }
}

/// Execute a bash command
pub struct ExecuteBashTool {
    /// Allowed command prefixes (for security)
    allowed_prefixes: Vec<String>,
    /// Working directory
    work_dir: Option<String>,
}

impl ExecuteBashTool {
    pub fn new() -> Self {
        Self {
            allowed_prefixes: vec![
                "ls".to_string(),
                "cat".to_string(),
                "head".to_string(),
                "tail".to_string(),
                "grep".to_string(),
                "find".to_string(),
                "wc".to_string(),
                "echo".to_string(),
                "pwd".to_string(),
                "which".to_string(),
                "cargo".to_string(),
                "rustc".to_string(),
                "python".to_string(),
                "pip".to_string(),
                "node".to_string(),
                "npm".to_string(),
                "git".to_string(),
            ],
            work_dir: None,
        }
    }

    pub fn with_allowed_commands(mut self, commands: Vec<String>) -> Self {
        self.allowed_prefixes = commands;
        self
    }

    pub fn with_work_dir(mut self, dir: String) -> Self {
        self.work_dir = Some(dir);
        self
    }

    fn is_command_allowed(&self, command: &str) -> bool {
        let cmd = command.trim().split_whitespace().next().unwrap_or("");
        self.allowed_prefixes.iter().any(|p| cmd == p || cmd.ends_with(&format!("/{}", p)))
    }
}

impl Default for ExecuteBashTool {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Deserialize)]
struct BashArgs {
    command: String,
    #[serde(default)]
    timeout_secs: Option<u64>,
}

#[async_trait]
impl Tool for ExecuteBashTool {
    fn name(&self) -> &str {
        "execute_bash"
    }

    fn description(&self) -> &str {
        "Execute a bash command and return the output. Only safe commands are allowed."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute"
                },
                "timeout_secs": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: 30)"
                }
            },
            "required": ["command"]
        })
    }

    async fn execute(&self, arguments: Value) -> Result<ToolResult> {
        let args: BashArgs = serde_json::from_value(arguments)?;

        // Security check
        if !self.is_command_allowed(&args.command) {
            return Ok(ToolResult::error(format!(
                "Command not allowed. Allowed commands: {}",
                self.allowed_prefixes.join(", ")
            )));
        }

        let mut cmd = Command::new("bash");
        cmd.arg("-c").arg(&args.command);
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        if let Some(ref dir) = self.work_dir {
            cmd.current_dir(dir);
        }

        let timeout = std::time::Duration::from_secs(args.timeout_secs.unwrap_or(30));

        let output = match tokio::time::timeout(timeout, cmd.output()).await {
            Ok(Ok(output)) => output,
            Ok(Err(e)) => return Ok(ToolResult::error(format!("Failed to execute: {}", e))),
            Err(_) => return Ok(ToolResult::error("Command timed out")),
        };

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        if output.status.success() {
            let mut result = stdout.to_string();
            if !stderr.is_empty() {
                result.push_str("\n[stderr]\n");
                result.push_str(&stderr);
            }
            Ok(ToolResult::success(result))
        } else {
            Ok(ToolResult::error(format!(
                "Command failed with exit code {:?}\nstdout: {}\nstderr: {}",
                output.status.code(),
                stdout,
                stderr
            )))
        }
    }

    fn timeout_secs(&self) -> u64 {
        60
    }
}

/// Search file contents with grep/ripgrep
pub struct GrepTool;

#[derive(Deserialize)]
struct GrepArgs {
    pattern: String,
    path: String,
    #[serde(default)]
    case_insensitive: bool,
    #[serde(default)]
    context_lines: Option<usize>,
}

#[async_trait]
impl Tool for GrepTool {
    fn name(&self) -> &str {
        "grep"
    }

    fn description(&self) -> &str {
        "Search for a pattern in files using ripgrep (if available) or grep."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for"
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search in"
                },
                "case_insensitive": {
                    "type": "boolean",
                    "description": "Perform case-insensitive search"
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Number of context lines to show before and after matches"
                }
            },
            "required": ["pattern", "path"]
        })
    }

    async fn execute(&self, arguments: Value) -> Result<ToolResult> {
        let args: GrepArgs = serde_json::from_value(arguments)?;

        // Try ripgrep first, fall back to grep
        let (cmd, cmd_args) = if which::which("rg").is_ok() {
            let mut rg_args = vec!["-n".to_string()];
            if args.case_insensitive {
                rg_args.push("-i".to_string());
            }
            if let Some(ctx) = args.context_lines {
                rg_args.push(format!("-C{}", ctx));
            }
            rg_args.push(args.pattern.clone());
            rg_args.push(args.path.clone());
            ("rg", rg_args)
        } else {
            let mut grep_args = vec!["-rn".to_string()];
            if args.case_insensitive {
                grep_args.push("-i".to_string());
            }
            if let Some(ctx) = args.context_lines {
                grep_args.push(format!("-C{}", ctx));
            }
            grep_args.push(args.pattern.clone());
            grep_args.push(args.path.clone());
            ("grep", grep_args)
        };

        let output = Command::new(cmd)
            .args(&cmd_args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        // grep returns exit code 1 when no matches found
        if output.status.success() || output.status.code() == Some(1) {
            if stdout.is_empty() {
                Ok(ToolResult::success("No matches found"))
            } else {
                // Truncate if too long
                let result = if stdout.len() > 10000 {
                    format!("{}...\n[truncated, {} total bytes]", &stdout[..10000], stdout.len())
                } else {
                    stdout.to_string()
                };
                Ok(ToolResult::success(result))
            }
        } else {
            Ok(ToolResult::error(format!("Search failed: {}", stderr)))
        }
    }

    fn timeout_secs(&self) -> u64 {
        60
    }
}

/// Parse email file and extract contact information
pub struct ParseEmailTool;

#[derive(Deserialize)]
struct ParseEmailArgs {
    path: String,
}

#[derive(serde::Serialize)]
struct EmailContact {
    name: Option<String>,
    email: String,
    role: String, // "from", "to", "cc"
}

#[derive(serde::Serialize)]
struct ParsedEmail {
    from: Vec<EmailContact>,
    to: Vec<EmailContact>,
    cc: Vec<EmailContact>,
    subject: Option<String>,
    date: Option<String>,
}

#[async_trait]
impl Tool for ParseEmailTool {
    fn name(&self) -> &str {
        "parse_email"
    }

    fn description(&self) -> &str {
        "Parse an email file (.eml) and extract sender, recipients, subject, and date. Returns structured contact information."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the .eml file to parse"
                }
            },
            "required": ["path"]
        })
    }

    async fn execute(&self, arguments: Value) -> Result<ToolResult> {
        let args: ParseEmailArgs = serde_json::from_value(arguments)?;

        let content = match fs::read_to_string(&args.path).await {
            Ok(c) => c,
            Err(e) => return Ok(ToolResult::error(format!("Failed to read file: {}", e))),
        };

        let parsed = parse_email_headers(&content);
        Ok(ToolResult::success(serde_json::to_string_pretty(&parsed)?))
    }

    fn timeout_secs(&self) -> u64 {
        10
    }
}

fn parse_email_headers(content: &str) -> ParsedEmail {
    let mut from = Vec::new();
    let mut to = Vec::new();
    let mut cc = Vec::new();
    let mut subject = None;
    let mut date = None;

    // Split headers from body (empty line separates them)
    let header_section = content.split("\r\n\r\n")
        .next()
        .or_else(|| content.split("\n\n").next())
        .unwrap_or(content);

    // Handle folded headers (continuation lines start with whitespace)
    let mut current_header = String::new();
    let mut headers = Vec::new();

    for line in header_section.lines() {
        if line.starts_with(' ') || line.starts_with('\t') {
            // Continuation of previous header
            current_header.push(' ');
            current_header.push_str(line.trim());
        } else {
            if !current_header.is_empty() {
                headers.push(current_header.clone());
            }
            current_header = line.to_string();
        }
    }
    if !current_header.is_empty() {
        headers.push(current_header);
    }

    for header in headers {
        let lower = header.to_lowercase();
        if lower.starts_with("from:") {
            from.extend(parse_address_list(&header[5..], "from"));
        } else if lower.starts_with("to:") {
            to.extend(parse_address_list(&header[3..], "to"));
        } else if lower.starts_with("cc:") {
            cc.extend(parse_address_list(&header[3..], "cc"));
        } else if lower.starts_with("subject:") {
            subject = Some(header[8..].trim().to_string());
        } else if lower.starts_with("date:") {
            date = Some(header[5..].trim().to_string());
        }
    }

    ParsedEmail { from, to, cc, subject, date }
}

fn parse_address_list(addresses: &str, role: &str) -> Vec<EmailContact> {
    let mut contacts = Vec::new();

    for addr in addresses.split(',') {
        let addr = addr.trim();
        if addr.is_empty() {
            continue;
        }

        // Parse formats like: "Name <email@example.com>" or just "email@example.com"
        if let Some(email_start) = addr.find('<') {
            if let Some(email_end) = addr.find('>') {
                let name = addr[..email_start].trim().trim_matches('"').to_string();
                let email = addr[email_start + 1..email_end].trim().to_string();
                contacts.push(EmailContact {
                    name: if name.is_empty() { None } else { Some(name) },
                    email,
                    role: role.to_string(),
                });
            }
        } else if addr.contains('@') {
            contacts.push(EmailContact {
                name: None,
                email: addr.to_string(),
                role: role.to_string(),
            });
        }
    }

    contacts
}

/// Create a vCard contact file
pub struct CreateVCardTool;

#[derive(Deserialize)]
struct CreateVCardArgs {
    output_path: String,
    name: String,
    email: String,
    #[serde(default)]
    phone: Option<String>,
    #[serde(default)]
    organization: Option<String>,
    #[serde(default)]
    append: bool,
}

#[async_trait]
impl Tool for CreateVCardTool {
    fn name(&self) -> &str {
        "create_vcard"
    }

    fn description(&self) -> &str {
        "Create a vCard (.vcf) contact file. Can append to existing file for multiple contacts."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "output_path": {
                    "type": "string",
                    "description": "Path for the output .vcf file"
                },
                "name": {
                    "type": "string",
                    "description": "Full name of the contact"
                },
                "email": {
                    "type": "string",
                    "description": "Email address"
                },
                "phone": {
                    "type": "string",
                    "description": "Phone number (optional)"
                },
                "organization": {
                    "type": "string",
                    "description": "Organization/company (optional)"
                },
                "append": {
                    "type": "boolean",
                    "description": "If true, append to existing file instead of overwriting"
                }
            },
            "required": ["output_path", "name", "email"]
        })
    }

    async fn execute(&self, arguments: Value) -> Result<ToolResult> {
        let args: CreateVCardArgs = serde_json::from_value(arguments)?;

        // Generate vCard 3.0 format
        let mut vcard = String::new();
        vcard.push_str("BEGIN:VCARD\n");
        vcard.push_str("VERSION:3.0\n");
        vcard.push_str(&format!("FN:{}\n", args.name));

        // Try to split name into first/last
        let name_parts: Vec<&str> = args.name.split_whitespace().collect();
        if name_parts.len() >= 2 {
            let last = name_parts.last().unwrap();
            let first = name_parts[..name_parts.len() - 1].join(" ");
            vcard.push_str(&format!("N:{};{};;;\n", last, first));
        } else {
            vcard.push_str(&format!("N:{}\n", args.name));
        }

        vcard.push_str(&format!("EMAIL:{}\n", args.email));

        if let Some(ref phone) = args.phone {
            vcard.push_str(&format!("TEL:{}\n", phone));
        }

        if let Some(ref org) = args.organization {
            vcard.push_str(&format!("ORG:{}\n", org));
        }

        vcard.push_str("END:VCARD\n");

        // Write or append
        let result = if args.append {
            let mut existing = fs::read_to_string(&args.output_path).await.unwrap_or_default();
            existing.push_str(&vcard);
            fs::write(&args.output_path, existing).await
        } else {
            fs::write(&args.output_path, &vcard).await
        };

        match result {
            Ok(_) => Ok(ToolResult::success(format!(
                "Created vCard for {} at {}",
                args.name, args.output_path
            ))),
            Err(e) => Ok(ToolResult::error(format!("Failed to write vCard: {}", e))),
        }
    }

    fn timeout_secs(&self) -> u64 {
        10
    }
}

/// Batch process multiple email files to extract all contacts
pub struct ExtractContactsTool;

#[derive(Deserialize)]
struct ExtractContactsArgs {
    directory: String,
    #[serde(default = "default_pattern")]
    pattern: String,
    #[serde(default)]
    output_format: Option<String>, // "json" or "csv"
}

fn default_pattern() -> String {
    "*.eml".to_string()
}

#[async_trait]
impl Tool for ExtractContactsTool {
    fn name(&self) -> &str {
        "extract_contacts"
    }

    fn description(&self) -> &str {
        "Scan a directory for email files and extract all unique contacts. Returns deduplicated list of contacts found."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Directory to scan for email files"
                },
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern for email files (default: *.eml)"
                },
                "output_format": {
                    "type": "string",
                    "enum": ["json", "csv"],
                    "description": "Output format (default: json)"
                }
            },
            "required": ["directory"]
        })
    }

    async fn execute(&self, arguments: Value) -> Result<ToolResult> {
        let args: ExtractContactsArgs = serde_json::from_value(arguments)?;

        let pattern = format!("{}/{}", args.directory, args.pattern);
        let mut all_contacts: std::collections::HashMap<String, (Option<String>, Vec<String>)> =
            std::collections::HashMap::new();

        // Find matching files
        for entry in glob::glob(&pattern).map_err(|e| anyhow::anyhow!("Invalid pattern: {}", e))? {
            if let Ok(path) = entry {
                if let Ok(content) = fs::read_to_string(&path).await {
                    let parsed = parse_email_headers(&content);

                    for contact in parsed.from.into_iter()
                        .chain(parsed.to)
                        .chain(parsed.cc)
                    {
                        let email_lower = contact.email.to_lowercase();
                        let entry = all_contacts.entry(email_lower).or_insert((None, Vec::new()));

                        // Keep the best name we find
                        if entry.0.is_none() && contact.name.is_some() {
                            entry.0 = contact.name;
                        }

                        // Track roles
                        if !entry.1.contains(&contact.role) {
                            entry.1.push(contact.role);
                        }
                    }
                }
            }
        }

        // Format output
        let output_format = args.output_format.as_deref().unwrap_or("json");

        if output_format == "csv" {
            let mut csv = String::from("email,name,roles\n");
            for (email, (name, roles)) in &all_contacts {
                csv.push_str(&format!(
                    "{},{},{}\n",
                    email,
                    name.as_deref().unwrap_or(""),
                    roles.join(";")
                ));
            }
            Ok(ToolResult::success(format!(
                "Found {} unique contacts:\n{}",
                all_contacts.len(),
                csv
            )))
        } else {
            let contacts: Vec<_> = all_contacts
                .into_iter()
                .map(|(email, (name, roles))| {
                    json!({
                        "email": email,
                        "name": name,
                        "roles": roles
                    })
                })
                .collect();

            Ok(ToolResult::success(format!(
                "Found {} unique contacts:\n{}",
                contacts.len(),
                serde_json::to_string_pretty(&contacts)?
            )))
        }
    }

    fn timeout_secs(&self) -> u64 {
        120
    }
}
