use std::{
    collections::BTreeMap,
    env, fs,
    io::{self, Write},
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::{LazyLock, Mutex},
    time::Duration,
};

use clap::{ArgAction, Args, Parser, Subcommand, ValueEnum};
use reqwest::multipart;
use serde_json::{Map, Value, json};
use tokio::net::TcpListener;

type CliResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

#[derive(Debug, Parser)]
#[command(
    name = "socartes",
    about = "Socartes CLI - agent-first interface for capabilities, tools, and knowledge.",
    arg_required_else_help = true
)]
struct Cli {
    #[arg(
        long,
        global = true,
        env = "SOCARTES_API_URL",
        default_value = "http://127.0.0.1:8000"
    )]
    api_url: String,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    #[command(about = "Run any capability in a single turn.")]
    Run(RunArgs),
    #[command(about = "Launch the Socartes Rust backend.")]
    Start(StartArgs),
    #[command(about = "Start the Socartes API server.")]
    Serve(ServeArgs),
    #[command(about = "Interactive chat REPL.")]
    Chat(ChatArgs),
    #[command(subcommand, about = "Manage interactive Books.")]
    Book(BookCommand),
    #[command(subcommand, about = "Manage TutorBot instances.")]
    Bot(BotCommand),
    #[command(subcommand, about = "Manage knowledge bases.")]
    Kb(KbCommand),
    #[command(subcommand, about = "Manage notebooks and markdown records.")]
    Notebook(NotebookCommand),
    #[command(subcommand, about = "View and manage lightweight memory.")]
    Memory(MemoryCommand),
    #[command(subcommand, about = "List plugins and capabilities.")]
    Plugin(PluginCommand),
    #[command(subcommand, about = "Inspect configuration.")]
    Config(ConfigCommand),
    #[command(subcommand, about = "Manage shared sessions.")]
    Session(SessionCommand),
    #[command(subcommand, about = "Manage provider authentication.")]
    Provider(ProviderCommand),
    #[command(about = "Initialize local Socartes runtime files.")]
    Init(InitArgs),
}

static TOOL_RESULTS: LazyLock<Mutex<ToolResultBuffer>> =
    LazyLock::new(|| Mutex::new(ToolResultBuffer::default()));

#[derive(Debug, Clone, Copy, ValueEnum)]
enum OutputFormat {
    Rich,
    Json,
}

#[derive(Debug, Args)]
struct RunArgs {
    capability: String,
    message: String,
    #[arg(long)]
    session: Option<String>,
    #[arg(long = "tool", short = 't', action = ArgAction::Append)]
    tool: Vec<String>,
    #[arg(long = "kb", action = ArgAction::Append)]
    kb: Vec<String>,
    #[arg(long = "notebook-ref", action = ArgAction::Append)]
    notebook_ref: Vec<String>,
    #[arg(long = "history-ref", action = ArgAction::Append)]
    history_ref: Vec<String>,
    #[arg(long, short = 'l', default_value = "en")]
    language: String,
    #[arg(long = "config", action = ArgAction::Append)]
    config: Vec<String>,
    #[arg(long = "config-json")]
    config_json: Option<String>,
    #[arg(long = "format", short = 'f', value_enum, default_value_t = OutputFormat::Rich)]
    format: OutputFormat,
}

#[derive(Debug, Args)]
struct StartArgs {
    #[arg(long, default_value = "0.0.0.0")]
    host: String,
    #[arg(long, default_value_t = 8000)]
    port: u16,
    #[arg(long)]
    home: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct ServeArgs {
    #[arg(long, default_value = "0.0.0.0")]
    host: String,
    #[arg(long, default_value_t = 8000)]
    port: u16,
    #[arg(long, action = ArgAction::SetTrue)]
    reload: bool,
    #[arg(long)]
    home: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct ChatArgs {
    #[arg(long)]
    session: Option<String>,
    #[arg(long = "tool", short = 't', action = ArgAction::Append)]
    tool: Vec<String>,
    #[arg(long, short = 'c', default_value = "chat")]
    capability: String,
    #[arg(long = "kb", action = ArgAction::Append)]
    kb: Vec<String>,
    #[arg(long = "notebook-ref", action = ArgAction::Append)]
    notebook_ref: Vec<String>,
    #[arg(long = "history-ref", action = ArgAction::Append)]
    history_ref: Vec<String>,
    #[arg(long, short = 'l', default_value = "en")]
    language: String,
}

#[derive(Debug, Subcommand)]
enum BookCommand {
    List(JsonFormatArgs),
    Health { book_id: String },
    RefreshFingerprints { book_id: String },
}

#[derive(Debug, Subcommand)]
enum BotCommand {
    List(JsonFormatArgs),
    Start {
        name: String,
    },
    Stop {
        name: String,
    },
    Create {
        name: String,
        #[arg(long = "name", short = 'n', default_value = "")]
        display_name: String,
        #[arg(long, short = 'p', default_value = "")]
        persona: String,
        #[arg(long, short = 'm', default_value = "")]
        model: String,
    },
}

#[derive(Debug, Subcommand)]
enum KbCommand {
    List(JsonFormatArgs),
    Info {
        name: String,
    },
    SetDefault {
        name: String,
    },
    Create(KbCreateArgs),
    Add(KbAddArgs),
    Delete {
        name: String,
        #[arg(long, short = 'f', action = ArgAction::SetTrue)]
        force: bool,
    },
    Search {
        name: String,
        query: String,
        #[arg(long, default_value = "hybrid")]
        mode: String,
        #[arg(long = "format", short = 'f', value_enum, default_value_t = OutputFormat::Rich)]
        format: OutputFormat,
    },
}

#[derive(Debug, Args)]
struct KbCreateArgs {
    name: String,
    #[arg(long = "doc", short = 'd', action = ArgAction::Append)]
    docs: Vec<PathBuf>,
    #[arg(long = "docs-dir")]
    docs_dir: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct KbAddArgs {
    name: String,
    #[arg(long = "doc", short = 'd', action = ArgAction::Append)]
    docs: Vec<PathBuf>,
    #[arg(long = "docs-dir")]
    docs_dir: Option<PathBuf>,
}

#[derive(Debug, Subcommand)]
enum NotebookCommand {
    List(JsonFormatArgs),
    Create {
        name: String,
        #[arg(long, default_value = "")]
        description: String,
    },
    Show {
        notebook_id: String,
        #[arg(long = "format", value_enum, default_value_t = OutputFormat::Rich)]
        format: OutputFormat,
    },
    RemoveRecord {
        notebook_id: String,
        record_id: String,
    },
    AddMd {
        notebook_id: String,
        file_path: PathBuf,
        #[arg(long, default_value = "")]
        title: String,
        #[arg(long = "type", default_value = "chat")]
        record_type: String,
    },
    ReplaceMd {
        notebook_id: String,
        record_id: String,
        file_path: PathBuf,
    },
}

#[derive(Debug, Subcommand)]
enum MemoryCommand {
    Show {
        #[arg(default_value = "all")]
        file: String,
        #[arg(long = "format", short = 'f', value_enum, default_value_t = OutputFormat::Rich)]
        format: OutputFormat,
    },
    Clear {
        #[arg(default_value = "all")]
        file: String,
        #[arg(long, short = 'f', action = ArgAction::SetTrue)]
        force: bool,
    },
}

#[derive(Debug, Subcommand)]
enum PluginCommand {
    List(JsonFormatArgs),
    Info { name: String },
}

#[derive(Debug, Subcommand)]
enum ConfigCommand {
    Show(ConfigShowArgs),
}

#[derive(Debug, Args)]
struct ConfigShowArgs {
    #[arg(long)]
    home: Option<PathBuf>,
    #[arg(long, action = ArgAction::SetTrue)]
    api: bool,
}

#[derive(Debug, Subcommand)]
enum SessionCommand {
    List {
        #[arg(long, default_value_t = 20)]
        limit: usize,
        #[arg(long = "format", short = 'f', value_enum, default_value_t = OutputFormat::Rich)]
        format: OutputFormat,
    },
    Show {
        session_id: String,
        #[arg(long = "format", value_enum, default_value_t = OutputFormat::Rich)]
        format: OutputFormat,
    },
    Open {
        session_id: String,
    },
    Delete {
        session_id: String,
    },
    Rename {
        session_id: String,
        #[arg(long)]
        title: String,
    },
}

#[derive(Debug, Subcommand)]
enum ProviderCommand {
    Login { provider: String },
}

#[derive(Debug, Subcommand)]
enum InitCommand {
    #[command(about = "Run the interactive-compatible setup wizard.")]
    Wizard(InitWizardArgs),
}

#[derive(Debug, Args)]
struct InitArgs {
    #[arg(long, action = ArgAction::SetTrue)]
    yes: bool,
    #[arg(long, action = ArgAction::SetTrue)]
    cli: bool,
    #[arg(long)]
    home: Option<PathBuf>,
    #[command(subcommand)]
    command: Option<InitCommand>,
}

#[derive(Debug, Args)]
struct InitWizardArgs {
    #[arg(long, action = ArgAction::SetTrue)]
    yes: bool,
    #[arg(long, action = ArgAction::SetTrue)]
    cli: bool,
    #[arg(long)]
    home: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct JsonFormatArgs {
    #[arg(long = "format", short = 'f', value_enum, default_value_t = OutputFormat::Rich)]
    format: OutputFormat,
}

struct ApiClient {
    base_url: String,
    client: reqwest::Client,
    token: Option<String>,
}

impl ApiClient {
    fn new(base_url: &str) -> CliResult<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()?;
        let token = env::var("SOCARTES_API_TOKEN")
            .or_else(|_| env::var("SOCARTES_TOKEN"))
            .ok()
            .filter(|value| !value.trim().is_empty());
        Ok(Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            client,
            token,
        })
    }

    fn url(&self, path: &str) -> String {
        format!("{}/{}", self.base_url, path.trim_start_matches('/'))
    }

    fn with_auth(&self, request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.token {
            Some(token) => request.bearer_auth(token),
            None => request,
        }
    }

    async fn get_json(&self, path: &str) -> CliResult<Value> {
        let response = self
            .with_auth(self.client.get(self.url(path)))
            .send()
            .await?;
        self.response_json(response).await
    }

    async fn post_json(&self, path: &str, body: &Value) -> CliResult<Value> {
        let response = self
            .with_auth(self.client.post(self.url(path)).json(body))
            .send()
            .await?;
        self.response_json(response).await
    }

    async fn put_json(&self, path: &str, body: &Value) -> CliResult<Value> {
        let response = self
            .with_auth(self.client.put(self.url(path)).json(body))
            .send()
            .await?;
        self.response_json(response).await
    }

    async fn patch_json(&self, path: &str, body: &Value) -> CliResult<Value> {
        let response = self
            .with_auth(self.client.patch(self.url(path)).json(body))
            .send()
            .await?;
        self.response_json(response).await
    }

    async fn delete_json(&self, path: &str) -> CliResult<Value> {
        let response = self
            .with_auth(self.client.delete(self.url(path)))
            .send()
            .await?;
        self.response_json(response).await
    }

    async fn post_multipart(&self, path: &str, form: multipart::Form) -> CliResult<Value> {
        let response = self
            .with_auth(self.client.post(self.url(path)).multipart(form))
            .send()
            .await?;
        self.response_json(response).await
    }

    async fn post_sse_text(&self, path: &str, body: &Value) -> CliResult<String> {
        let response = self
            .with_auth(self.client.post(self.url(path)).json(body))
            .send()
            .await?;
        let status = response.status();
        let text = response.text().await?;
        if !status.is_success() {
            return Err(format!("HTTP {}: {}", status.as_u16(), text.trim()).into());
        }
        Ok(text)
    }

    async fn response_json(&self, response: reqwest::Response) -> CliResult<Value> {
        let status = response.status();
        let text = response.text().await?;
        if !status.is_success() {
            return Err(format!("HTTP {}: {}", status.as_u16(), text.trim()).into());
        }
        if text.trim().is_empty() {
            return Ok(json!({}));
        }
        serde_json::from_str(&text)
            .map_err(|error| format!("Invalid JSON response: {error}: {text}").into())
    }
}

#[tokio::main]
async fn main() -> CliResult {
    let cli = Cli::parse();
    match cli.command {
        Command::Serve(args) => serve(args.host, args.port, args.reload, args.home).await,
        Command::Start(args) => serve(args.host, args.port, false, args.home).await,
        command => {
            let api = ApiClient::new(&cli.api_url)?;
            dispatch_api_command(api, command).await
        }
    }
}

async fn dispatch_api_command(api: ApiClient, command: Command) -> CliResult {
    match command {
        Command::Run(args) => run_capability(&api, &args).await,
        Command::Chat(args) => chat_repl(&api, args).await,
        Command::Book(command) => book_command(&api, command).await,
        Command::Bot(command) => bot_command(&api, command).await,
        Command::Kb(command) => kb_command(&api, command).await,
        Command::Notebook(command) => notebook_command(&api, command).await,
        Command::Memory(command) => memory_command(&api, command).await,
        Command::Plugin(command) => plugin_command(&api, command).await,
        Command::Config(command) => config_command(&api, command).await,
        Command::Session(command) => session_command(&api, command).await,
        Command::Provider(command) => provider_command(&api, command).await,
        Command::Init(args) => init_command(args).await,
        Command::Serve(_) | Command::Start(_) => {
            unreachable!("serve/start handled before API dispatch")
        }
    }
}

async fn serve(host: String, port: u16, reload: bool, home: Option<PathBuf>) -> CliResult {
    if reload {
        eprintln!(
            "--reload is accepted for Python CLI parity; the Rust binary runs without hot reload."
        );
    }
    let address: SocketAddr = format!("{host}:{port}").parse()?;
    let listener = TcpListener::bind(address).await?;
    eprintln!("Socartes Rust API listening on http://{address}");
    let app = match home {
        Some(home) => {
            socartes_backend::app_with_knowledge_root(home.join("data").join("knowledge"))
        }
        None => socartes_backend::app(),
    };
    axum::serve(listener, app).await?;
    Ok(())
}

async fn run_capability(api: &ApiClient, args: &RunArgs) -> CliResult {
    let config = merged_config(&args.config_json, &args.config)?;
    let notebook_references = parse_notebook_refs(&args.notebook_ref)?;
    execute_capability(
        api,
        CapabilityTurn {
            capability: &args.capability,
            content: &args.message,
            session_id: args.session.as_deref(),
            tools: &args.tool,
            knowledge_bases: &args.kb,
            notebook_references,
            history_references: args.history_ref.clone(),
            language: &args.language,
            config,
            format: args.format,
        },
    )
    .await?;
    Ok(())
}

struct CapabilityTurn<'a> {
    capability: &'a str,
    content: &'a str,
    session_id: Option<&'a str>,
    tools: &'a [String],
    knowledge_bases: &'a [String],
    notebook_references: Vec<Value>,
    history_references: Vec<String>,
    language: &'a str,
    config: Value,
    format: OutputFormat,
}

async fn execute_capability(
    api: &ApiClient,
    turn: CapabilityTurn<'_>,
) -> CliResult<Option<StreamIdentity>> {
    let mut payload = json!({
        "content": turn.content,
        "tools": turn.tools,
        "knowledge_bases": turn.knowledge_bases,
        "language": turn.language,
        "config": turn.config,
        "notebook_references": turn.notebook_references,
        "history_references": turn.history_references,
        "attachments": []
    });
    if let Some(session_id) = turn.session_id {
        payload["session_id"] = json!(session_id);
    }
    let text = api
        .post_sse_text(
            &format!(
                "/api/v1/plugins/capabilities/{}/execute-stream",
                turn.capability
            ),
            &payload,
        )
        .await?;
    let identity = stream_identity_from_sse(&text);
    render_sse(&text, turn.format)?;
    Ok(identity)
}

fn render_sse(text: &str, format: OutputFormat) -> CliResult {
    let events = parse_sse_events(text);
    if events.is_empty() {
        if let Ok(value) = serde_json::from_str::<Value>(text) {
            print_value(&value, format)?;
        } else {
            println!("{text}");
        }
        return Ok(());
    }

    for (event, payload) in events {
        match format {
            OutputFormat::Json => println!("{}", serde_json::to_string(&payload)?),
            OutputFormat::Rich => render_stream_payload(event.as_deref(), &payload),
        }
    }
    Ok(())
}

fn parse_sse_events(text: &str) -> Vec<(Option<String>, Value)> {
    text.split("\n\n")
        .filter_map(|block| {
            let mut event = None::<String>;
            let mut data_lines = Vec::new();
            for line in block.lines() {
                if let Some(value) = line.strip_prefix("event:") {
                    event = Some(value.trim().to_string());
                } else if let Some(value) = line.strip_prefix("data:") {
                    data_lines.push(value.trim_start());
                }
            }
            if data_lines.is_empty() {
                return None;
            }
            let data = data_lines.join("\n");
            serde_json::from_str::<Value>(&data)
                .ok()
                .map(|payload| (event, payload))
        })
        .collect()
}

#[derive(Debug, Clone)]
struct ToolResultEntry {
    index: usize,
    label: String,
    body: String,
}

#[derive(Debug)]
struct ToolResultBuffer {
    entries: Vec<ToolResultEntry>,
    next_index: usize,
    capacity: usize,
}

impl Default for ToolResultBuffer {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            next_index: 1,
            capacity: 32,
        }
    }
}

impl ToolResultBuffer {
    fn remember(&mut self, label: &str, body: &str) -> ToolResultEntry {
        let entry = ToolResultEntry {
            index: self.next_index,
            label: if label.trim().is_empty() {
                "tool".to_string()
            } else {
                label.to_string()
            },
            body: body.to_string(),
        };
        self.next_index += 1;
        self.entries.push(entry.clone());
        if self.entries.len() > self.capacity {
            let overflow = self.entries.len() - self.capacity;
            self.entries.drain(0..overflow);
        }
        entry
    }

    fn get(&self, selector: &str) -> Option<ToolResultEntry> {
        let selector = selector.trim();
        if selector.is_empty() || selector == "last" {
            return self.entries.last().cloned();
        }
        if let Ok(index) = selector.parse::<usize>() {
            return self
                .entries
                .iter()
                .find(|entry| entry.index == index)
                .cloned();
        }
        self.entries
            .iter()
            .rev()
            .find(|entry| entry.label == selector)
            .cloned()
    }

    fn indexes(&self) -> Vec<usize> {
        self.entries.iter().map(|entry| entry.index).collect()
    }
}

fn truncate_tool_result(body: &str, head_lines: usize, line_hard_cap: usize) -> (String, usize) {
    if head_lines == 0 {
        return (String::new(), body.lines().count());
    }
    let lines = body.split('\n').collect::<Vec<_>>();
    let hidden = lines.len().saturating_sub(head_lines);
    let visible = lines
        .iter()
        .take(head_lines)
        .map(|line| clip_line(line, line_hard_cap))
        .collect::<Vec<_>>()
        .join("\n");
    (visible, hidden)
}

fn clip_line(line: &str, line_hard_cap: usize) -> String {
    if line_hard_cap == 0 || line.chars().count() <= line_hard_cap {
        return line.to_string();
    }
    let mut clipped = line
        .chars()
        .take(line_hard_cap.saturating_sub(1))
        .collect::<String>();
    clipped.push_str("...");
    clipped
}

fn render_stream_payload(event: Option<&str>, payload: &Value) {
    let payload_type = payload["type"].as_str().or(event).unwrap_or("event");
    match payload_type {
        "process_log" => {
            if let Some(message) = payload["message"].as_str() {
                eprintln!("{message}");
            }
        }
        "stage_start" => println!("\n> {}", payload["stage"].as_str().unwrap_or("working")),
        "thinking" | "progress" => {
            if let Some(content) = payload["content"].as_str() {
                eprintln!("  {content}");
            }
        }
        "content" => {
            if let Some(content) = payload["content"].as_str() {
                print!("{content}");
                let _ = io::stdout().flush();
            }
        }
        "tool_result" => {
            render_tool_result_preview(payload);
        }
        "tool_call" | "sources" => {
            println!(
                "{}",
                serde_json::to_string_pretty(payload).unwrap_or_default()
            );
        }
        "error" => {
            eprintln!(
                "Error: {}",
                payload["detail"]
                    .as_str()
                    .or_else(|| payload["content"].as_str())
                    .unwrap_or("unknown error")
            );
        }
        "result" => {
            if let Some(result) = payload.pointer("/data/result")
                && let Some(response) = result["response"]
                    .as_str()
                    .or_else(|| result["content"].as_str())
                && !response.trim().is_empty()
            {
                println!("\n{response}");
            }
        }
        _ => {}
    }
}

fn render_tool_result_preview(payload: &Value) {
    let body = payload["content"].as_str().unwrap_or("");
    let label = payload["metadata"]["tool"].as_str().unwrap_or("tool");
    let entry = match TOOL_RESULTS.lock() {
        Ok(mut buffer) => buffer.remember(label, body),
        Err(_) => ToolResultEntry {
            index: 0,
            label: label.to_string(),
            body: body.to_string(),
        },
    };
    let (head, hidden) = truncate_tool_result(body, 10, 240);
    if !head.trim().is_empty() {
        for line in head.lines() {
            println!("  | {line}");
        }
    }
    if hidden > 0 {
        println!(
            "  #{} {} - +{} more line{}; run /show {} or /show last to expand",
            entry.index,
            entry.label,
            hidden,
            if hidden == 1 { "" } else { "s" },
            entry.index
        );
    } else if head.trim().is_empty() {
        println!("  #{} {} -> (empty result)", entry.index, entry.label);
    } else {
        println!("  #{} {}", entry.index, entry.label);
    }
}

fn render_tool_result_entry(selector: &str) {
    let (entry, available) = match TOOL_RESULTS.lock() {
        Ok(buffer) => (buffer.get(selector), buffer.indexes()),
        Err(_) => (None, Vec::new()),
    };
    if let Some(entry) = entry {
        println!("#{} {}", entry.index, entry.label);
        if entry.body.is_empty() {
            println!("(empty result)");
        } else {
            println!("{}", entry.body);
        }
    } else if selector.trim().is_empty() || selector.trim() == "last" {
        println!("No tool result captured yet in this session.");
    } else {
        println!(
            "No tool result matches {}. Available: {}.",
            selector,
            if available.is_empty() {
                "none".to_string()
            } else {
                available
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            }
        );
    }
}

async fn chat_repl(api: &ApiClient, args: ChatArgs) -> CliResult {
    let mut state = ChatState {
        session_id: args.session,
        capability: args.capability,
        tools: args.tool,
        knowledge_bases: args.kb,
        notebook_references: parse_notebook_refs(&args.notebook_ref)?,
        history_references: args.history_ref,
        language: args.language,
        config: json!({}),
    };

    load_existing_chat_session(api, &mut state).await?;

    println!("Socartes chat. Type /quit to exit, /session for state.");
    loop {
        print!("socartes> ");
        io::stdout().flush()?;
        let mut line = String::new();
        if io::stdin().read_line(&mut line)? == 0 {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('/') {
            let command = line.split_whitespace().next().unwrap_or_default();
            if matches!(command, "/regenerate" | "/retry") {
                regenerate_chat_turn(api, &mut state).await?;
                continue;
            }
            if apply_chat_command(line, &mut state)? {
                continue;
            }
            break;
        }
        let identity = execute_capability(
            api,
            CapabilityTurn {
                capability: &state.capability,
                content: line,
                session_id: state.session_id.as_deref(),
                tools: &state.tools,
                knowledge_bases: &state.knowledge_bases,
                notebook_references: state.notebook_references.clone(),
                history_references: state.history_references.clone(),
                language: &state.language,
                config: state.config.clone(),
                format: OutputFormat::Rich,
            },
        )
        .await?;
        if let Some(identity) = identity
            && let Some(session_id) = identity.session_id
        {
            state.session_id = Some(session_id);
        }
        println!();
    }
    Ok(())
}

async fn regenerate_chat_turn(api: &ApiClient, state: &mut ChatState) -> CliResult {
    let Some(session_id) = state
        .session_id
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
    else {
        println!("No active session yet - send a message first.");
        return Ok(());
    };
    let payload = json!({
        "overrides": {
            "capability": state.capability,
            "tools": state.tools,
            "knowledge_bases": state.knowledge_bases,
            "language": state.language,
            "notebook_references": state.notebook_references,
            "history_references": state.history_references,
            "config": state.config
        }
    });
    let text = api
        .post_sse_text(
            &format!("/api/v1/sessions/{session_id}/regenerate-stream"),
            &payload,
        )
        .await?;
    render_sse(&text, OutputFormat::Rich)?;
    if let Some(identity) = stream_identity_from_sse(&text) {
        if let Some(session_id) = identity.session_id {
            state.session_id = Some(session_id);
        }
        if let Some(turn_id) = identity.turn_id {
            let session_label = state.session_id.as_deref().unwrap_or(&session_id);
            println!(
                "\nsession={session_label} turn={turn_id} capability={} (regenerated)",
                state.capability
            );
        }
    }
    Ok(())
}

#[derive(Debug, Default)]
struct StreamIdentity {
    session_id: Option<String>,
    turn_id: Option<String>,
}

fn stream_identity_from_sse(text: &str) -> Option<StreamIdentity> {
    let mut identity = StreamIdentity::default();
    for (_, payload) in parse_sse_events(text) {
        if identity.session_id.is_none() {
            identity.session_id = payload["session_id"]
                .as_str()
                .or_else(|| payload["metadata"]["session_id"].as_str())
                .or_else(|| payload["data"]["session_id"].as_str())
                .map(ToString::to_string);
        }
        if identity.turn_id.is_none() {
            identity.turn_id = payload["turn_id"]
                .as_str()
                .or_else(|| payload["metadata"]["turn_id"].as_str())
                .or_else(|| payload["data"]["turn_id"].as_str())
                .map(ToString::to_string);
        }
    }
    (identity.session_id.is_some() || identity.turn_id.is_some()).then_some(identity)
}

struct ChatState {
    session_id: Option<String>,
    capability: String,
    tools: Vec<String>,
    knowledge_bases: Vec<String>,
    notebook_references: Vec<Value>,
    history_references: Vec<String>,
    language: String,
    config: Value,
}

fn apply_chat_command(raw: &str, state: &mut ChatState) -> CliResult<bool> {
    let parts = raw.split_whitespace().collect::<Vec<_>>();
    match parts.as_slice() {
        ["/quit"] | ["/exit"] => Ok(false),
        ["/session"] | ["/refs"] => {
            print_chat_state(state)?;
            Ok(true)
        }
        ["/new"] => {
            state.session_id = None;
            println!("Started a new session context.");
            Ok(true)
        }
        ["/tool", "on", name] => {
            if !state.tools.iter().any(|tool| tool == name) {
                state.tools.push((*name).to_string());
            }
            Ok(true)
        }
        ["/tool", "off", name] => {
            state.tools.retain(|tool| tool != name);
            Ok(true)
        }
        ["/cap", name] => {
            state.capability = (*name).to_string();
            Ok(true)
        }
        ["/kb", "none"] => {
            state.knowledge_bases.clear();
            Ok(true)
        }
        ["/kb", name] => {
            state.knowledge_bases = vec![(*name).to_string()];
            Ok(true)
        }
        ["/history", "add", id] => {
            state.history_references.push((*id).to_string());
            Ok(true)
        }
        ["/history", "clear"] => {
            state.history_references.clear();
            Ok(true)
        }
        ["/notebook", "add", reference] => {
            state
                .notebook_references
                .extend(parse_notebook_refs(&[(*reference).to_string()])?);
            Ok(true)
        }
        ["/notebook", "clear"] => {
            state.notebook_references.clear();
            Ok(true)
        }
        ["/show"] => {
            render_tool_result_entry("last");
            Ok(true)
        }
        ["/show", selector] => {
            render_tool_result_entry(selector);
            Ok(true)
        }
        ["/config", "show"] => {
            print_json(&state.config)?;
            Ok(true)
        }
        ["/config", "clear"] => {
            state.config = json!({});
            Ok(true)
        }
        ["/config", "set", item] => {
            let mut object = state.config.as_object().cloned().unwrap_or_default();
            let (key, value) = parse_config_item(item)?;
            object.insert(key, value);
            state.config = Value::Object(object);
            Ok(true)
        }
        _ => {
            eprintln!("Unknown chat command: {raw}");
            Ok(true)
        }
    }
}

async fn load_existing_chat_session(api: &ApiClient, state: &mut ChatState) -> CliResult {
    let Some(session_id) = state
        .session_id
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return Ok(());
    };
    let session = api
        .get_json(&format!("/api/v1/sessions/{session_id}"))
        .await
        .map_err(|error| format!("Session not found: {session_id}: {error}"))?;
    apply_chat_session_preferences(state, &session);
    Ok(())
}

fn apply_chat_session_preferences(state: &mut ChatState, session: &Value) {
    let preferences = &session["preferences"];
    if let Some(capability) = preferences["capability"]
        .as_str()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        state.capability = capability.to_string();
    }
    let tools = string_array(&preferences["tools"]);
    if !tools.is_empty() {
        state.tools = tools;
    }
    let knowledge_bases = string_array(&preferences["knowledge_bases"]);
    if !knowledge_bases.is_empty() {
        state.knowledge_bases = knowledge_bases;
    }
    if let Some(language) = preferences["language"]
        .as_str()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        state.language = language.to_string();
    }
    let notebook_references = preferences["notebook_references"]
        .as_array()
        .map(|items| items.to_vec())
        .unwrap_or_default();
    if !notebook_references.is_empty() {
        state.notebook_references = notebook_references;
    }
    let history_references = string_array(&preferences["history_references"]);
    if !history_references.is_empty() {
        state.history_references = history_references;
    }
}

fn string_array(value: &Value) -> Vec<String> {
    value
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
        .collect()
}

fn print_chat_state(state: &ChatState) -> CliResult {
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "session_id": &state.session_id,
            "capability": &state.capability,
            "tools": &state.tools,
            "knowledge_bases": &state.knowledge_bases,
            "language": &state.language,
            "history_references": &state.history_references,
            "notebook_references": &state.notebook_references,
            "config": &state.config
        }))?
    );
    Ok(())
}

async fn book_command(api: &ApiClient, command: BookCommand) -> CliResult {
    match command {
        BookCommand::List(args) => {
            let value = api.get_json("/api/v1/book/books").await?;
            print_collection(&value, "books", args.format)
        }
        BookCommand::Health { book_id } => print_json(
            &api.get_json(&format!("/api/v1/book/books/{book_id}/health"))
                .await?,
        ),
        BookCommand::RefreshFingerprints { book_id } => print_json(
            &api.post_json(
                &format!("/api/v1/book/books/{book_id}/refresh-fingerprints"),
                &json!({}),
            )
            .await?,
        ),
    }
}

async fn bot_command(api: &ApiClient, command: BotCommand) -> CliResult {
    match command {
        BotCommand::List(args) => {
            let value = api.get_json("/api/v1/tutorbot").await?;
            print_collection(&value, "", args.format)
        }
        BotCommand::Start { name } => print_json(
            &api.post_json("/api/v1/tutorbot", &json!({ "bot_id": name }))
                .await?,
        ),
        BotCommand::Stop { name } => {
            print_json(&api.delete_json(&format!("/api/v1/tutorbot/{name}")).await?)
        }
        BotCommand::Create {
            name,
            display_name,
            persona,
            model,
        } => {
            let mut payload = json!({ "bot_id": name });
            if !display_name.is_empty() {
                payload["name"] = json!(display_name);
            }
            if !persona.is_empty() {
                payload["persona"] = json!(persona);
            }
            if !model.is_empty() {
                payload["model"] = json!(model);
            }
            print_json(&api.post_json("/api/v1/tutorbot", &payload).await?)
        }
    }
}

async fn kb_command(api: &ApiClient, command: KbCommand) -> CliResult {
    match command {
        KbCommand::List(args) => {
            let value = api.get_json("/api/v1/knowledge/list").await?;
            print_collection(&value, "knowledge_bases", args.format)
        }
        KbCommand::Info { name } => {
            print_json(&api.get_json(&format!("/api/v1/knowledge/{name}")).await?)
        }
        KbCommand::SetDefault { name } => print_json(
            &api.put_json(&format!("/api/v1/knowledge/default/{name}"), &json!({}))
                .await?,
        ),
        KbCommand::Create(args) => {
            let docs = collect_documents(&args.docs, args.docs_dir.as_deref())?;
            if docs.is_empty() {
                return Err(
                    "Provide at least one supported document with --doc or --docs-dir.".into(),
                );
            }
            let form = multipart_form_with_files(Some(("name", args.name.as_str())), &docs)?;
            print_json(&api.post_multipart("/api/v1/knowledge/create", form).await?)
        }
        KbCommand::Add(args) => {
            let docs = collect_documents(&args.docs, args.docs_dir.as_deref())?;
            if docs.is_empty() {
                return Err(
                    "Provide at least one supported document with --doc or --docs-dir.".into(),
                );
            }
            let form = multipart_form_with_files(None, &docs)?;
            print_json(
                &api.post_multipart(&format!("/api/v1/knowledge/{}/upload", args.name), form)
                    .await?,
            )
        }
        KbCommand::Delete { name, force } => {
            if !force && !confirm(&format!("Delete knowledge base '{name}'?"))? {
                return Ok(());
            }
            print_json(
                &api.delete_json(&format!("/api/v1/knowledge/{name}"))
                    .await?,
            )
        }
        KbCommand::Search {
            name,
            query,
            mode,
            format,
        } => {
            let value = api
                .post_json(
                    "/api/v1/plugins/tools/rag/execute",
                    &json!({ "params": { "query": query, "kb_name": name, "mode": mode } }),
                )
                .await?;
            print_value(&value, format)
        }
    }
}

async fn notebook_command(api: &ApiClient, command: NotebookCommand) -> CliResult {
    match command {
        NotebookCommand::List(args) => {
            let value = api.get_json("/api/v1/notebook/list").await?;
            print_collection(&value, "notebooks", args.format)
        }
        NotebookCommand::Create { name, description } => print_json(
            &api.post_json(
                "/api/v1/notebook/create",
                &json!({ "name": name, "description": description }),
            )
            .await?,
        ),
        NotebookCommand::Show {
            notebook_id,
            format,
        } => print_value(
            &api.get_json(&format!("/api/v1/notebook/{notebook_id}"))
                .await?,
            format,
        ),
        NotebookCommand::RemoveRecord {
            notebook_id,
            record_id,
        } => print_json(
            &api.delete_json(&format!(
                "/api/v1/notebook/{notebook_id}/records/{record_id}"
            ))
            .await?,
        ),
        NotebookCommand::AddMd {
            notebook_id,
            file_path,
            title,
            record_type,
        } => {
            let content = fs::read_to_string(&file_path)?;
            let record_title = if title.is_empty() {
                file_path
                    .file_stem()
                    .and_then(|name| name.to_str())
                    .unwrap_or("Markdown record")
                    .to_string()
            } else {
                title
            };
            print_json(
                &api.post_json(
                    "/api/v1/notebook/add_record",
                    &json!({
                        "notebook_ids": [notebook_id],
                        "record_type": record_type,
                        "title": record_title,
                        "user_query": "",
                        "output": content
                    }),
                )
                .await?,
            )
        }
        NotebookCommand::ReplaceMd {
            notebook_id,
            record_id,
            file_path,
        } => {
            let content = fs::read_to_string(file_path)?;
            print_json(
                &api.put_json(
                    &format!("/api/v1/notebook/{notebook_id}/records/{record_id}"),
                    &json!({ "output": content }),
                )
                .await?,
            )
        }
    }
}

async fn memory_command(api: &ApiClient, command: MemoryCommand) -> CliResult {
    match command {
        MemoryCommand::Show { file, format } => {
            let value = api.get_json("/api/v1/memory").await?;
            if matches!(format, OutputFormat::Json) || file == "all" {
                return print_value(&value, format);
            }
            if let Some(content) = value.get(&file) {
                print_value(content, format)
            } else {
                Err(format!("Unknown memory file: {file}. Use summary, profile, or all.").into())
            }
        }
        MemoryCommand::Clear { file, force } => {
            if !force {
                let target = if file == "all" {
                    "all memory files".to_string()
                } else {
                    format!("{file} memory file")
                };
                if !confirm(&format!("Clear {target}?"))? {
                    return Ok(());
                }
            }
            let body = if file == "all" {
                json!({})
            } else {
                json!({ "file": file })
            };
            print_json(&api.post_json("/api/v1/memory/clear", &body).await?)
        }
    }
}

async fn plugin_command(api: &ApiClient, command: PluginCommand) -> CliResult {
    match command {
        PluginCommand::List(args) => {
            let value = api.get_json("/api/v1/plugins/list").await?;
            print_value(&value, args.format)
        }
        PluginCommand::Info { name } => {
            let value = api.get_json("/api/v1/plugins/list").await?;
            for section in ["tools", "capabilities"] {
                if let Some(found) = value[section]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .find(|item| item["name"].as_str() == Some(name.as_str()))
                {
                    return print_json(found);
                }
            }
            Err(format!("Plugin or capability not found: {name}").into())
        }
    }
}

async fn config_command(api: &ApiClient, command: ConfigCommand) -> CliResult {
    match command {
        ConfigCommand::Show(args) => {
            if args.api {
                return print_json(&api.get_json("/api/v1/settings").await?);
            }
            print_json(&local_config_summary(args.home)?)
        }
    }
}

fn local_config_summary(home: Option<PathBuf>) -> CliResult<Value> {
    let data_root = data_root_from_home(home);
    let settings_root = data_root.join("settings");
    let catalog = read_json_or_default(&settings_root.join("catalog.json"), default_cli_catalog())?;
    let ui = read_json_or_default(&settings_root.join("ui.json"), default_cli_ui_settings())?;
    Ok(json!({
        "ports": local_ports_summary(&ui),
        "llm": local_llm_summary(&catalog),
        "embedding": local_embedding_summary(&catalog),
        "search": local_search_summary(&catalog),
        "language": ui["language"].as_str().unwrap_or("en"),
        "tools": local_tools_summary(&ui)
    }))
}

fn data_root_from_home(home: Option<PathBuf>) -> PathBuf {
    match home {
        Some(home) => home.join("data"),
        None => env::var("SOCARTES_DATA_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("data")),
    }
}

fn read_json_or_default(path: &Path, default: Value) -> CliResult<Value> {
    match fs::read_to_string(path) {
        Ok(text) => Ok(serde_json::from_str(&text)?),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(default),
        Err(error) => Err(error.into()),
    }
}

fn local_ports_summary(ui: &Value) -> Value {
    json!({
        "backend": ui.pointer("/ports/backend")
            .or_else(|| ui.get("backend_port"))
            .cloned()
            .unwrap_or_else(|| json!(8000)),
        "frontend": ui.pointer("/ports/frontend")
            .or_else(|| ui.get("frontend_port"))
            .cloned()
            .unwrap_or_else(|| json!(3000))
    })
}

fn local_llm_summary(catalog: &Value) -> Value {
    let service = &catalog["services"]["llm"];
    let profile = active_service_profile(catalog, "llm");
    let model = active_service_model(service, profile);
    let binding = profile_string(profile, "binding");
    json!({
        "binding_hint": binding,
        "provider": binding,
        "provider_mode": local_provider_mode(&binding),
        "model": model_string(model, "model")
            .or_else(|| model_string(model, "id"))
            .or_else(|| service["active_model_id"].as_str().map(str::to_string))
            .unwrap_or_default(),
        "base_url": profile_string(profile, "base_url"),
        "api_version": profile_string(profile, "api_version"),
        "extra_headers": profile
            .and_then(|value| value.get("extra_headers"))
            .cloned()
            .unwrap_or_else(|| json!({})),
        "api_key": masked_secret(profile_string(profile, "api_key"))
    })
}

fn local_embedding_summary(catalog: &Value) -> Value {
    let service = &catalog["services"]["embedding"];
    let profile = active_service_profile(catalog, "embedding");
    let model = active_service_model(service, profile);
    let binding = profile_string(profile, "binding");
    json!({
        "binding_hint": binding,
        "provider": binding,
        "provider_mode": local_provider_mode(&binding),
        "model": model_string(model, "model")
            .or_else(|| model_string(model, "id"))
            .or_else(|| service["active_model_id"].as_str().map(str::to_string))
            .unwrap_or_default(),
        "base_url": profile_string(profile, "base_url"),
        "api_version": profile_string(profile, "api_version"),
        "extra_headers": profile
            .and_then(|value| value.get("extra_headers"))
            .cloned()
            .unwrap_or_else(|| json!({})),
        "api_key": masked_secret(profile_string(profile, "api_key")),
        "dimension": model
            .and_then(|value| value.get("dimension"))
            .cloned()
            .unwrap_or(Value::Null)
    })
}

fn local_search_summary(catalog: &Value) -> Value {
    let profile = active_service_profile(catalog, "search");
    let provider = profile_string(profile, "provider");
    json!({
        "provider": if provider.is_empty() { "(optional)" } else { provider.as_str() },
        "requested_provider": if provider.is_empty() { "(optional)" } else { provider.as_str() },
        "status": if provider.is_empty() { "optional" } else { "configured" },
        "fallback_reason": Value::Null,
        "base_url": profile_string(profile, "base_url"),
        "proxy": profile_string(profile, "proxy"),
        "api_key": masked_secret(profile_string(profile, "api_key"))
    })
}

fn active_service_profile<'a>(catalog: &'a Value, service_name: &str) -> Option<&'a Value> {
    let service = &catalog["services"][service_name];
    let active_profile_id = service["active_profile_id"].as_str();
    let profiles = service["profiles"].as_array()?;
    profiles
        .iter()
        .find(|profile| profile["id"].as_str() == active_profile_id)
        .or_else(|| profiles.first())
}

fn active_service_model<'a>(service: &'a Value, profile: Option<&'a Value>) -> Option<&'a Value> {
    let active_model_id = service["active_model_id"].as_str();
    let models = profile?.get("models")?.as_array()?;
    models
        .iter()
        .find(|model| model["id"].as_str() == active_model_id)
        .or_else(|| models.first())
}

fn local_provider_mode(binding: &str) -> &str {
    match binding {
        "" => "",
        "openai" | "openai-compatible" => "openai-compatible",
        "anthropic" => "anthropic",
        "local" => "local",
        _ => "custom",
    }
}

fn profile_string(profile: Option<&Value>, key: &str) -> String {
    profile
        .and_then(|value| value[key].as_str())
        .unwrap_or("")
        .to_string()
}

fn model_string(model: Option<&Value>, key: &str) -> Option<String> {
    model.and_then(|value| value[key].as_str().map(str::to_string))
}

fn masked_secret(secret: String) -> &'static str {
    if secret.trim().is_empty() {
        "(not set)"
    } else {
        "***"
    }
}

fn local_tools_summary(ui: &Value) -> Value {
    match ui.get("tools").and_then(Value::as_object) {
        Some(tools) => Value::Array(tools.keys().map(|key| json!(key)).collect()),
        None => json!([]),
    }
}

async fn session_command(api: &ApiClient, command: SessionCommand) -> CliResult {
    match command {
        SessionCommand::List { limit, format } => {
            let mut value = api.get_json("/api/v1/sessions").await?;
            if let Some(sessions) = value["sessions"].as_array_mut() {
                sessions.truncate(limit);
            }
            print_value(&value, format)
        }
        SessionCommand::Show { session_id, format } => print_value(
            &api.get_json(&format!("/api/v1/sessions/{session_id}"))
                .await?,
            format,
        ),
        SessionCommand::Open { session_id } => {
            chat_repl(
                api,
                ChatArgs {
                    session: Some(session_id),
                    tool: Vec::new(),
                    capability: "chat".to_string(),
                    kb: Vec::new(),
                    notebook_ref: Vec::new(),
                    history_ref: Vec::new(),
                    language: "en".to_string(),
                },
            )
            .await
        }
        SessionCommand::Delete { session_id } => print_json(
            &api.delete_json(&format!("/api/v1/sessions/{session_id}"))
                .await?,
        ),
        SessionCommand::Rename { session_id, title } => print_json(
            &api.patch_json(
                &format!("/api/v1/sessions/{session_id}"),
                &json!({ "title": title }),
            )
            .await?,
        ),
    }
}

async fn provider_command(_api: &ApiClient, command: ProviderCommand) -> CliResult {
    match command {
        ProviderCommand::Login { provider } => match normalize_provider_name(&provider).as_str() {
            "openai_codex" => login_openai_codex(),
            "github_copilot" => login_github_copilot().await,
            _ => Err(format!(
                "Unknown provider `{provider}`. Supported: openai-codex, github-copilot"
            )
            .into()),
        },
    }
}

fn normalize_provider_name(provider: &str) -> String {
    provider.trim().to_lowercase().replace('-', "_")
}

fn login_openai_codex() -> CliResult {
    let token = env::var("SOCARTES_OPENAI_CODEX_ACCESS_TOKEN")
        .or_else(|_| env::var("OPENAI_CODEX_ACCESS_TOKEN"))
        .ok()
        .filter(|value| !value.trim().is_empty());
    if token.is_some() {
        println!("OpenAI Codex OAuth authentication succeeded.");
        return Ok(());
    }
    Err("OpenAI Codex OAuth authentication failed: no existing OAuth token was found. Set SOCARTES_OPENAI_CODEX_ACCESS_TOKEN or run the OpenAI Codex login flow.".into())
}

async fn login_github_copilot() -> CliResult {
    let base_url = env::var("SOCARTES_GITHUB_COPILOT_BASE_URL")
        .unwrap_or_else(|_| "https://api.githubcopilot.com".to_string());
    let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));
    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()?
        .post(url)
        .bearer_auth("copilot")
        .json(&json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1
        }))
        .send()
        .await;
    let response = match response {
        Ok(response) => response,
        Err(error) => {
            return Err(format!("GitHub Copilot auth validation failed: {error}").into());
        }
    };
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(format!(
            "GitHub Copilot auth validation failed: HTTP {}: {}",
            status.as_u16(),
            body.trim()
        )
        .into());
    }
    println!("GitHub Copilot auth validation succeeded.");
    Ok(())
}

async fn init_command(args: InitArgs) -> CliResult {
    let mut yes = args.yes;
    let mut cli_only = args.cli;
    let mut home = args.home;
    if let Some(command) = args.command {
        let InitCommand::Wizard(wizard) = command;
        yes |= wizard.yes;
        cli_only |= wizard.cli;
        home = home.or(wizard.home);
    }
    run_init_wizard(InitWizardArgs {
        yes,
        cli: cli_only,
        home,
    })
}

fn run_init_wizard(args: InitWizardArgs) -> CliResult {
    let data_root = match args.home {
        Some(home) => home.join("data"),
        None => env::var("SOCARTES_DATA_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("data")),
    };
    if !args.yes {
        println!(
            "This will create local Socartes runtime directories under {}.",
            data_root.display()
        );
        if !confirm("Continue?")? {
            return Ok(());
        }
    }

    let dirs = [
        data_root.join("knowledge"),
        data_root.join("sessions"),
        data_root.join("user").join("workspace").join("book"),
        data_root.join("user").join("workspace").join("notebook"),
        data_root
            .join("user")
            .join("workspace")
            .join("chat")
            .join("attachments"),
        data_root.join("memory"),
        data_root.join("settings"),
        data_root.join("auth").join("users"),
        data_root.join("tutorbot"),
        data_root.join("skills"),
    ];
    for dir in &dirs {
        fs::create_dir_all(dir)?;
    }

    let settings_root = data_root.join("settings");
    write_json_if_absent(&settings_root.join("catalog.json"), &default_cli_catalog())?;
    write_json_if_absent(&settings_root.join("ui.json"), &default_cli_ui_settings())?;
    write_json_if_absent(
        &data_root.join("knowledge").join("kb_config.json"),
        &json!({ "knowledge_bases": {} }),
    )?;

    print_json(&json!({
        "initialized": true,
        "cli_only": args.cli,
        "data_dir": data_root,
        "settings": settings_root,
        "created": dirs
    }))
}

fn write_json_if_absent(path: &Path, value: &Value) -> CliResult {
    if path.exists() {
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let bytes = serde_json::to_vec_pretty(value)?;
    fs::write(path, bytes)?;
    Ok(())
}

fn default_cli_ui_settings() -> Value {
    json!({
        "theme": "light",
        "language": "en",
        "sidebar_description": "Data Intelligence Lab @ HKU",
        "sidebar_nav_order": {
            "start": ["/", "/history", "/knowledge", "/notebook"],
            "learnResearch": ["/question", "/solver", "/research", "/co_writer"]
        }
    })
}

fn default_cli_catalog() -> Value {
    json!({
        "version": 1,
        "services": {
            "llm": {
                "active_profile_id": "socartes-rust",
                "active_model_id": "deterministic-agent-loop",
                "profiles": [{
                    "id": "socartes-rust",
                    "name": "Socartes Rust",
                    "binding": "openai",
                    "base_url": "http://127.0.0.1:8810/v1",
                    "api_key": "",
                    "api_version": "",
                    "extra_headers": {},
                    "models": [{
                        "id": "deterministic-agent-loop",
                        "name": "Deterministic Agent Loop",
                        "model": "deterministic-agent-loop",
                        "context_window": "8192",
                        "context_window_source": "rust-default"
                    }]
                }]
            },
            "embedding": {
                "active_profile_id": "socartes-rust-embedding",
                "active_model_id": "deterministic-embedding",
                "profiles": [{
                    "id": "socartes-rust-embedding",
                    "name": "Socartes Rust Embedding",
                    "binding": "openai",
                    "base_url": "http://127.0.0.1:8810/v1",
                    "api_key": "",
                    "api_version": "",
                    "extra_headers": {},
                    "models": [{
                        "id": "deterministic-embedding",
                        "name": "Deterministic Embedding",
                        "model": "deterministic-embedding",
                        "dimension": "3072",
                        "send_dimensions": true,
                        "supported_dimensions": "1536,3072"
                    }]
                }]
            },
            "search": {
                "active_profile_id": "duckduckgo-local",
                "profiles": [{
                    "id": "duckduckgo-local",
                    "name": "DuckDuckGo Local",
                    "provider": "duckduckgo",
                    "base_url": "",
                    "api_key": "",
                    "api_version": "",
                    "proxy": "",
                    "max_results": 5,
                    "models": []
                }]
            }
        }
    })
}

fn merged_config(raw_json: &Option<String>, items: &[String]) -> CliResult<Value> {
    let mut object = match raw_json {
        Some(raw) if !raw.trim().is_empty() => match serde_json::from_str::<Value>(raw)? {
            Value::Object(map) => map,
            _ => return Err("--config-json must be a JSON object.".into()),
        },
        _ => Map::new(),
    };
    for item in items {
        let (key, value) = parse_config_item(item)?;
        object.insert(key, value);
    }
    Ok(Value::Object(object))
}

fn parse_config_item(item: &str) -> CliResult<(String, Value)> {
    let Some((key, raw_value)) = item.split_once('=') else {
        return Err(format!("Invalid --config item `{item}`. Expected KEY=VALUE.").into());
    };
    let key = key.trim();
    if key.is_empty() {
        return Err(format!("Invalid --config item `{item}`. Expected KEY=VALUE.").into());
    }
    let raw_value = raw_value.trim();
    let value = serde_json::from_str::<Value>(raw_value).unwrap_or_else(|_| json!(raw_value));
    Ok((key.to_string(), value))
}

fn parse_notebook_refs(items: &[String]) -> CliResult<Vec<Value>> {
    let mut refs = Vec::new();
    for item in items {
        let (notebook_id, record_part) = item.split_once(':').unwrap_or((item.as_str(), ""));
        let notebook_id = notebook_id.trim();
        if notebook_id.is_empty() {
            return Err(format!("Invalid notebook reference `{item}`.").into());
        }
        let record_ids = record_part
            .split(',')
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        refs.push(json!({ "notebook_id": notebook_id, "record_ids": record_ids }));
    }
    Ok(refs)
}

fn collect_documents(docs: &[PathBuf], docs_dir: Option<&Path>) -> CliResult<Vec<PathBuf>> {
    let mut collected = BTreeMap::<String, PathBuf>::new();
    for doc in docs {
        let path = doc.canonicalize()?;
        if path.is_file() {
            collected.insert(path.to_string_lossy().to_string(), path);
        }
    }
    if let Some(dir) = docs_dir {
        let dir = dir.canonicalize()?;
        if !dir.is_dir() {
            return Err(format!("docs directory does not exist: {}", dir.display()).into());
        }
        collect_files_recursively(&dir, &mut collected)?;
    }
    Ok(collected.into_values().collect())
}

fn collect_files_recursively(dir: &Path, collected: &mut BTreeMap<String, PathBuf>) -> CliResult {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_files_recursively(&path, collected)?;
        } else if path.is_file() {
            let path = path.canonicalize()?;
            collected.insert(path.to_string_lossy().to_string(), path);
        }
    }
    Ok(())
}

fn multipart_form_with_files(
    text_field: Option<(&str, &str)>,
    docs: &[PathBuf],
) -> CliResult<multipart::Form> {
    let mut form = multipart::Form::new();
    if let Some((name, value)) = text_field {
        form = form.text(name.to_string(), value.to_string());
    }
    for path in docs {
        let bytes = fs::read(path)?;
        let filename = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("document")
            .to_string();
        form = form.part("files", multipart::Part::bytes(bytes).file_name(filename));
    }
    Ok(form)
}

fn confirm(prompt: &str) -> CliResult<bool> {
    print!("{prompt} [y/N] ");
    io::stdout().flush()?;
    let mut answer = String::new();
    io::stdin().read_line(&mut answer)?;
    Ok(matches!(
        answer.trim().to_ascii_lowercase().as_str(),
        "y" | "yes"
    ))
}

fn print_collection(value: &Value, key: &str, format: OutputFormat) -> CliResult {
    if matches!(format, OutputFormat::Json) {
        return print_json(value);
    }
    let collection = if key.is_empty() {
        value.as_array()
    } else {
        value[key].as_array()
    };
    if let Some(items) = collection {
        if items.is_empty() {
            println!("No items.");
            return Ok(());
        }
        for item in items {
            let name = item["name"]
                .as_str()
                .or_else(|| item["title"].as_str())
                .or_else(|| item["id"].as_str())
                .or_else(|| item["bot_id"].as_str())
                .unwrap_or("(unnamed)");
            let id = item["id"]
                .as_str()
                .or_else(|| item["bot_id"].as_str())
                .unwrap_or("");
            let status = item["status"]
                .as_str()
                .or_else(|| {
                    item["running"]
                        .as_bool()
                        .map(|running| if running { "running" } else { "stopped" })
                })
                .unwrap_or("");
            println!("{name}\t{id}\t{status}");
        }
        Ok(())
    } else {
        print_json(value)
    }
}

fn print_value(value: &Value, format: OutputFormat) -> CliResult {
    match format {
        OutputFormat::Json => print_json(value),
        OutputFormat::Rich => {
            if let Some(text) = value.as_str() {
                println!("{text}");
            } else {
                print_json(value)?;
            }
            Ok(())
        }
    }
}

fn print_json(value: &Value) -> CliResult {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}
