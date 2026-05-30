use std::{
    collections::BTreeMap,
    env, fs,
    io::{self, Write},
    net::SocketAddr,
    path::{Path, PathBuf},
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
    #[command(subcommand, about = "Initialize local Socartes runtime files.")]
    Init(InitCommand),
}

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
}

#[derive(Debug, Args)]
struct ServeArgs {
    #[arg(long, default_value = "0.0.0.0")]
    host: String,
    #[arg(long, default_value_t = 8000)]
    port: u16,
    #[arg(long, action = ArgAction::SetTrue)]
    reload: bool,
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
    Show,
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
    Wizard {
        #[arg(long, action = ArgAction::SetTrue)]
        yes: bool,
    },
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
        Command::Serve(args) => serve(args.host, args.port, args.reload).await,
        Command::Start(args) => serve(args.host, args.port, false).await,
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
        Command::Init(command) => init_command(command).await,
        Command::Serve(_) | Command::Start(_) => {
            unreachable!("serve/start handled before API dispatch")
        }
    }
}

async fn serve(host: String, port: u16, reload: bool) -> CliResult {
    if reload {
        eprintln!(
            "--reload is accepted for Python CLI parity; the Rust binary runs without hot reload."
        );
    }
    let address: SocketAddr = format!("{host}:{port}").parse()?;
    let listener = TcpListener::bind(address).await?;
    eprintln!("Socartes Rust API listening on http://{address}");
    axum::serve(listener, socartes_backend::app()).await?;
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
    .await
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

async fn execute_capability(api: &ApiClient, turn: CapabilityTurn<'_>) -> CliResult {
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
    render_sse(&text, turn.format)
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
        "tool_call" | "tool_result" | "sources" => {
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
            if apply_chat_command(line, &mut state)? {
                continue;
            }
            break;
        }
        execute_capability(
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
        println!();
    }
    Ok(())
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
        ["/session"] => {
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "session_id": &state.session_id,
                    "capability": &state.capability,
                    "tools": &state.tools,
                    "knowledge_bases": &state.knowledge_bases,
                    "history_references": &state.history_references,
                    "notebook_references": &state.notebook_references,
                    "config": &state.config
                }))?
            );
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
        ConfigCommand::Show => print_json(&api.get_json("/api/v1/settings").await?),
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

async fn provider_command(api: &ApiClient, command: ProviderCommand) -> CliResult {
    match command {
        ProviderCommand::Login { provider } => {
            let settings = api.get_json("/api/v1/settings").await?;
            println!(
                "Provider `{provider}` login is handled by configured environment/OAuth tokens. Current settings:"
            );
            print_json(&settings)
        }
    }
}

async fn init_command(command: InitCommand) -> CliResult {
    match command {
        InitCommand::Wizard { yes } => {
            let root = env::var("SOCARTES_DATA_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from("data"));
            let dirs = [
                root.join("knowledge_bases"),
                root.join("sessions"),
                root.join("notebooks"),
                root.join("books"),
                root.join("memory"),
                root.join("outputs"),
            ];
            if !yes {
                println!(
                    "This will create local Socartes runtime directories under {}.",
                    root.display()
                );
                if !confirm("Continue?")? {
                    return Ok(());
                }
            }
            for dir in &dirs {
                fs::create_dir_all(dir)?;
            }
            print_json(&json!({
                "initialized": true,
                "data_dir": root,
                "created": dirs
            }))
        }
    }
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
