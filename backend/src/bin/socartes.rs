use std::{
    collections::BTreeMap,
    env, fs,
    io::{self, Write},
    net::{SocketAddr, TcpStream},
    path::{Path, PathBuf},
    sync::{LazyLock, Mutex},
    time::{Duration, Instant},
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
        default_value = "http://127.0.0.1:8001"
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
    Init(Box<InitArgs>),
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
    #[arg(long)]
    port: Option<u16>,
    #[arg(long)]
    frontend_port: Option<u16>,
    #[arg(long, env = "SOCARTES_FRONTEND_DIR")]
    frontend_dir: Option<PathBuf>,
    #[arg(long, action = ArgAction::SetTrue)]
    dry_run: bool,
    #[arg(long)]
    home: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct ServeArgs {
    #[arg(long, default_value = "0.0.0.0")]
    host: String,
    #[arg(long, default_value_t = 8001)]
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
    #[command(flatten)]
    runtime: RuntimeInitOptions,
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
    #[command(flatten)]
    runtime: RuntimeInitOptions,
}

#[derive(Debug, Clone, Default, Args)]
struct RuntimeInitOptions {
    #[arg(long)]
    llm_binding: Option<String>,
    #[arg(long)]
    llm_base_url: Option<String>,
    #[arg(long)]
    llm_api_key: Option<String>,
    #[arg(long)]
    llm_model: Option<String>,
    #[arg(long)]
    embedding_binding: Option<String>,
    #[arg(long)]
    embedding_base_url: Option<String>,
    #[arg(long)]
    embedding_api_key: Option<String>,
    #[arg(long)]
    embedding_model: Option<String>,
    #[arg(long)]
    embedding_dimension: Option<u32>,
    #[arg(long)]
    search_provider: Option<String>,
    #[arg(long)]
    search_base_url: Option<String>,
    #[arg(long)]
    search_api_key: Option<String>,
    #[arg(long)]
    backend_port: Option<u16>,
    #[arg(long)]
    frontend_port: Option<u16>,
    #[arg(long, short = 'l')]
    language: Option<String>,
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
        Command::Start(args) => start_launcher(args).await,
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
        Command::Init(args) => init_command(*args).await,
        Command::Serve(_) | Command::Start(_) => {
            unreachable!("serve/start handled before API dispatch")
        }
    }
}

async fn start_launcher(args: StartArgs) -> CliResult {
    let plan = build_start_plan(&args)?;
    if args.dry_run {
        return print_json(&plan);
    }
    cleanup_previous_start_state(&plan);
    ensure_start_ports_available(&plan)?;

    let frontend_dir = PathBuf::from(plan["frontend"]["cwd"].as_str().unwrap_or_default());
    let api_base = plan["frontend"]["env"]["NEXT_PUBLIC_API_BASE"]
        .as_str()
        .unwrap_or_default()
        .to_string();
    let auth_enabled = plan["frontend"]["env"]["NEXT_PUBLIC_AUTH_ENABLED"]
        .as_str()
        .unwrap_or("false")
        .to_string();
    write_frontend_env_local(&frontend_dir, &api_base, &auth_enabled)?;

    let backend_command = string_array(&plan["backend"]["command"]);
    let frontend_command = string_array(&plan["frontend"]["command"]);
    if backend_command.is_empty() || frontend_command.is_empty() {
        return Err("Invalid start plan: missing backend or frontend command.".into());
    }
    ensure_command_available(&backend_command[0])?;
    ensure_command_available(&frontend_command[0])?;

    eprintln!(
        "Starting Socartes backend at {}",
        plan["backend"]["url"].as_str().unwrap_or("")
    );
    let mut backend = spawn_command(
        &backend_command,
        env_pairs(&[
            ("BACKEND_PORT", plan["backend"]["port"].to_string()),
            ("FRONTEND_PORT", plan["frontend"]["port"].to_string()),
        ]),
        None,
    )?;
    if let Err(error) = write_start_state(&plan, Some(&backend), None) {
        cleanup_started_processes(&plan, Some(&mut backend), None);
        return Err(error);
    }
    if let Err(error) = wait_for_http(
        plan["backend"]["url"].as_str().unwrap_or_default(),
        "backend",
        &mut backend,
        readiness_timeout(&plan, "backend_timeout_ms", Duration::from_secs(60)),
    )
    .await
    {
        cleanup_started_processes(&plan, Some(&mut backend), None);
        return Err(error);
    }

    eprintln!(
        "Starting Socartes frontend at {}",
        plan["frontend"]["url"].as_str().unwrap_or("")
    );
    let mut frontend = match spawn_command(
        &frontend_command,
        env_pairs(&[
            ("BACKEND_PORT", plan["backend"]["port"].to_string()),
            ("FRONTEND_PORT", plan["frontend"]["port"].to_string()),
            ("NEXT_PUBLIC_API_BASE", api_base.clone()),
            (
                "AUTH_ENABLED",
                plan["frontend"]["env"]["AUTH_ENABLED"]
                    .as_str()
                    .unwrap_or("false")
                    .to_string(),
            ),
            (
                "NEXT_PUBLIC_AUTH_ENABLED",
                plan["frontend"]["env"]["NEXT_PUBLIC_AUTH_ENABLED"]
                    .as_str()
                    .unwrap_or("false")
                    .to_string(),
            ),
        ]),
        Some(&frontend_dir),
    ) {
        Ok(frontend) => frontend,
        Err(error) => {
            cleanup_started_processes(&plan, Some(&mut backend), None);
            return Err(error);
        }
    };
    if let Err(error) = write_start_state(&plan, Some(&backend), Some(&frontend)) {
        cleanup_started_processes(&plan, Some(&mut backend), Some(&mut frontend));
        return Err(error);
    }
    if let Err(error) = wait_for_http(
        plan["frontend"]["url"].as_str().unwrap_or_default(),
        "frontend",
        &mut frontend,
        readiness_timeout(&plan, "frontend_timeout_ms", Duration::from_secs(120)),
    )
    .await
    {
        cleanup_started_processes(&plan, Some(&mut backend), Some(&mut frontend));
        return Err(error);
    }

    println!(
        "Open {}",
        plan["frontend"]["url"].as_str().unwrap_or_default()
    );
    let exit_code = monitor_started_processes(&mut backend, &mut frontend);
    remove_start_state(&plan);
    if exit_code == 0 {
        Ok(())
    } else {
        Err(format!("Socartes start exited with code {exit_code}.").into())
    }
}

fn build_start_plan(args: &StartArgs) -> CliResult<Value> {
    let frontend_dir = resolve_frontend_dir(args.frontend_dir.as_deref())?;
    let settings = resolve_start_settings(args, &frontend_dir)?;
    let current_exe = env::current_exe()?;
    let mut backend_command = vec![
        current_exe.display().to_string(),
        "serve".to_string(),
        "--host".to_string(),
        args.host.clone(),
        "--port".to_string(),
        settings.backend_port.to_string(),
    ];
    if let Some(home) = &args.home {
        backend_command.push("--home".to_string());
        backend_command.push(home.display().to_string());
    }
    let backend_command =
        command_override("SOCARTES_START_BACKEND_COMMAND").unwrap_or(backend_command);
    let frontend_command = vec![
        "npm".to_string(),
        "run".to_string(),
        "dev".to_string(),
        "--".to_string(),
        "--port".to_string(),
        settings.frontend_port.to_string(),
    ];
    let frontend_command =
        command_override("SOCARTES_START_FRONTEND_COMMAND").unwrap_or(frontend_command);
    let state_path = start_data_root_from_home(args.home.as_deref())?
        .join("user")
        .join("settings")
        .join("start_web_state.json");
    let backend_url = format!(
        "http://{}:{}",
        normalized_host_for_url(&args.host),
        settings.backend_port
    );
    let api_base = format!("http://localhost:{}", settings.backend_port);
    let auth_enabled = if settings.auth_enabled {
        "true"
    } else {
        "false"
    };
    Ok(json!({
        "backend": {
            "host": args.host,
            "port": settings.backend_port,
            "url": backend_url,
            "command": backend_command
        },
        "frontend": {
            "port": settings.frontend_port,
            "url": format!("http://localhost:{}", settings.frontend_port),
            "cwd": frontend_dir.display().to_string(),
            "command": frontend_command,
            "env": {
                "BACKEND_PORT": settings.backend_port.to_string(),
                "FRONTEND_PORT": settings.frontend_port.to_string(),
                "NEXT_PUBLIC_API_BASE": api_base,
                "AUTH_ENABLED": auth_enabled,
                "NEXT_PUBLIC_AUTH_ENABLED": auth_enabled
            }
        },
        "state_path": state_path.display().to_string(),
        "state": start_state_template(settings.backend_port, settings.frontend_port),
        "readiness": {
            "backend_timeout_ms": env_u64("SOCARTES_START_BACKEND_READY_TIMEOUT_MS").unwrap_or(60_000),
            "frontend_timeout_ms": env_u64("SOCARTES_START_FRONTEND_READY_TIMEOUT_MS").unwrap_or(120_000)
        }
    }))
}

fn command_override(name: &str) -> Option<Vec<String>> {
    let raw = env::var(name).ok()?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.starts_with('[')
        && let Ok(Value::Array(items)) = serde_json::from_str::<Value>(trimmed)
    {
        let command = items
            .into_iter()
            .filter_map(|item| item.as_str().map(str::to_string))
            .filter(|item| !item.trim().is_empty())
            .collect::<Vec<_>>();
        return (!command.is_empty()).then_some(command);
    }
    Some(vec![trimmed.to_string()])
}

struct StartSettings {
    backend_port: u16,
    frontend_port: u16,
    auth_enabled: bool,
}

fn resolve_start_settings(args: &StartArgs, frontend_dir: &Path) -> CliResult<StartSettings> {
    let project_root = start_project_root(frontend_dir)?;
    let dotenv = read_env_file(&project_root.join(".env"))?;
    let backend_port = args
        .port
        .or_else(|| env_u16_from_map(&dotenv, "BACKEND_PORT"))
        .or_else(|| env_u16("BACKEND_PORT"))
        .unwrap_or(8001);
    let frontend_port = args
        .frontend_port
        .or_else(|| env_u16_from_map(&dotenv, "FRONTEND_PORT"))
        .or_else(|| env_u16("FRONTEND_PORT"))
        .unwrap_or(3782);
    let auth_enabled = env_string_from_map(&dotenv, "NEXT_PUBLIC_AUTH_ENABLED")
        .or_else(|| env_string_from_map(&dotenv, "AUTH_ENABLED"))
        .or_else(|| env::var("NEXT_PUBLIC_AUTH_ENABLED").ok())
        .or_else(|| env::var("AUTH_ENABLED").ok())
        .map(|value| truthy_env(&value))
        .unwrap_or(false);
    Ok(StartSettings {
        backend_port,
        frontend_port,
        auth_enabled,
    })
}

fn start_project_root(frontend_dir: &Path) -> CliResult<PathBuf> {
    if frontend_dir.file_name().and_then(|value| value.to_str()) == Some("web")
        && let Some(parent) = frontend_dir.parent()
    {
        return Ok(parent.to_path_buf());
    }
    Ok(env::current_dir()?)
}

fn start_data_root_from_home(home: Option<&Path>) -> CliResult<PathBuf> {
    match home {
        Some(home) => Ok(home.join("data")),
        None => env::var("SOCARTES_DATA_DIR")
            .map(PathBuf::from)
            .or_else(|_| env::current_dir().map(|cwd| cwd.join("data")))
            .map_err(Into::into),
    }
}

fn read_env_file(path: &Path) -> CliResult<BTreeMap<String, String>> {
    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(BTreeMap::new()),
        Err(error) => return Err(error.into()),
    };
    let mut values = BTreeMap::new();
    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        values.insert(
            key.trim().to_string(),
            value
                .trim()
                .trim_matches('"')
                .trim_matches('\'')
                .to_string(),
        );
    }
    Ok(values)
}

fn env_string_from_map(map: &BTreeMap<String, String>, key: &str) -> Option<String> {
    map.get(key)
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn env_u16_from_map(map: &BTreeMap<String, String>, key: &str) -> Option<u16> {
    env_string_from_map(map, key).and_then(|value| value.parse::<u16>().ok())
}

fn env_u16(key: &str) -> Option<u16> {
    env::var(key)
        .ok()
        .and_then(|value| value.trim().parse::<u16>().ok())
}

fn env_u64(key: &str) -> Option<u64> {
    env::var(key)
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
}

fn truthy_env(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn normalized_host_for_url(host: &str) -> String {
    match host {
        "0.0.0.0" | "::" => "127.0.0.1".to_string(),
        value => value.to_string(),
    }
}

fn resolve_frontend_dir(explicit: Option<&Path>) -> CliResult<PathBuf> {
    let candidates = explicit
        .map(|path| vec![path.to_path_buf()])
        .unwrap_or_else(default_frontend_dir_candidates);
    candidates
        .into_iter()
        .find(|path| path.join("package.json").is_file() || path.is_dir())
        .ok_or_else(|| {
            "Frontend directory not found. Pass --frontend-dir or set SOCARTES_FRONTEND_DIR.".into()
        })
}

fn default_frontend_dir_candidates() -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Ok(cwd) = env::current_dir() {
        candidates.push(cwd.join("web"));
        if let Some(parent) = cwd.parent() {
            candidates.push(parent.join("web"));
            candidates.push(parent.join("DeepTutor").join("web"));
        }
    }
    candidates.push(PathBuf::from("/home/coobabm/DeepTutor/web"));
    candidates.push(PathBuf::from("/home/coobabm/.gitnexus/repos/DeepTutor/web"));
    candidates
}

fn write_frontend_env_local(frontend_dir: &Path, api_base: &str, auth_enabled: &str) -> CliResult {
    fs::create_dir_all(frontend_dir)?;
    fs::write(
        frontend_dir.join(".env.local"),
        format!(
            "# Auto-generated by socartes start - do not edit manually\nNEXT_PUBLIC_API_BASE={api_base}\nNEXT_PUBLIC_AUTH_ENABLED={auth_enabled}\n"
        ),
    )?;
    Ok(())
}

fn ensure_command_available(command: &str) -> CliResult {
    if command.contains('/') || command.contains('\\') {
        if Path::new(command).is_file() {
            return Ok(());
        }
        return Err(format!("Command not found: {command}").into());
    }
    let Some(path) = env::var_os("PATH") else {
        return Err(format!("Command not found on PATH: {command}").into());
    };
    for dir in env::split_paths(&path) {
        if dir.join(command).is_file() {
            return Ok(());
        }
    }
    Err(format!("Command not found on PATH: {command}").into())
}

fn env_pairs(pairs: &[(&str, String)]) -> Vec<(String, String)> {
    pairs
        .iter()
        .map(|(key, value)| ((*key).to_string(), value.clone()))
        .collect()
}

fn spawn_command(
    command: &[String],
    envs: Vec<(String, String)>,
    cwd: Option<&Path>,
) -> CliResult<std::process::Child> {
    let mut process = std::process::Command::new(&command[0]);
    process.args(&command[1..]).envs(envs);
    if let Some(cwd) = cwd {
        process.current_dir(cwd);
    }
    configure_child_process_group(&mut process);
    Ok(process
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .spawn()?)
}

fn readiness_timeout(plan: &Value, key: &str, fallback: Duration) -> Duration {
    plan["readiness"][key]
        .as_u64()
        .map(Duration::from_millis)
        .unwrap_or(fallback)
}

#[cfg(unix)]
fn configure_child_process_group(process: &mut std::process::Command) {
    use std::os::unix::process::CommandExt;
    process.process_group(0);
}

#[cfg(not(unix))]
fn configure_child_process_group(_process: &mut std::process::Command) {}

async fn wait_for_http(
    url: &str,
    name: &str,
    process: &mut std::process::Child,
    timeout: Duration,
) -> CliResult {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(1))
        .build()?;
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if let Some(status) = process.try_wait()? {
            return Err(format!("{name} exited before readiness: {status}").into());
        }
        match client.get(url).send().await {
            Ok(response) if response.status().as_u16() < 500 => return Ok(()),
            Ok(_) | Err(_) => {
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
        }
    }
    Err(format!("{name} did not become ready at {url} within {timeout:?}.").into())
}

struct StartPortOwner {
    command: String,
    pid: Option<u32>,
}

struct StartPortConflict {
    name: &'static str,
    port: u16,
    owners: Vec<StartPortOwner>,
}

fn ensure_start_ports_available(plan: &Value) -> CliResult {
    let ports = [
        ("Backend", value_u16(&plan["backend"]["port"])),
        ("Frontend", value_u16(&plan["frontend"]["port"])),
    ];
    let conflicts = ports
        .into_iter()
        .filter_map(|(name, port)| port.map(|port| (name, port)))
        .filter_map(|(name, port)| {
            let owners = listening_port_owners(port);
            (!owners.is_empty() || port_accepts_connection(port)).then_some(StartPortConflict {
                name,
                port,
                owners,
            })
        })
        .collect::<Vec<_>>();
    if conflicts.is_empty() {
        return Ok(());
    }

    for conflict in conflicts {
        eprintln!(
            "{} port {} is already in use.",
            conflict.name, conflict.port
        );
        if conflict.owners.is_empty() {
            eprintln!("owner: unknown process");
        } else {
            for owner in conflict.owners {
                match owner.pid {
                    Some(pid) => eprintln!("owner: {} (PID {})", owner.command, pid),
                    None => eprintln!("owner: {} (PID unknown)", owner.command),
                }
            }
        }
    }
    eprintln!(
        "Stop the existing process or run `python scripts/stop_web.py` if it is a stale Socartes launch."
    );
    Err("Port conflict detected.".into())
}

fn value_u16(value: &Value) -> Option<u16> {
    value.as_u64().and_then(|value| u16::try_from(value).ok())
}

fn port_accepts_connection(port: u16) -> bool {
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    TcpStream::connect_timeout(&addr, Duration::from_millis(250)).is_ok()
}

fn listening_port_owners(port: u16) -> Vec<StartPortOwner> {
    let output = match std::process::Command::new("lsof")
        .args(["-n", "-P", &format!("-iTCP:{port}"), "-sTCP:LISTEN"])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output()
    {
        Ok(output) => output,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Vec::new(),
        Err(_) => return Vec::new(),
    };
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .skip(1)
        .filter_map(|line| {
            let mut parts = line.split_whitespace();
            let command = parts.next()?.to_string();
            let pid = parts.next().and_then(|value| value.parse::<u32>().ok());
            Some(StartPortOwner { command, pid })
        })
        .collect()
}

fn write_start_state(
    plan: &Value,
    backend: Option<&std::process::Child>,
    frontend: Option<&std::process::Child>,
) -> CliResult {
    let path = PathBuf::from(plan["state_path"].as_str().unwrap_or_default());
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temp_path = path.with_extension("json.tmp");
    fs::write(
        &temp_path,
        serde_json::to_vec_pretty(&start_state_value(
            plan["backend"]["port"].as_u64().unwrap_or_default(),
            plan["frontend"]["port"].as_u64().unwrap_or_default(),
            backend,
            frontend,
        ))?,
    )?;
    fs::rename(temp_path, path)?;
    Ok(())
}

fn start_state_template(backend_port: u16, frontend_port: u16) -> Value {
    json!({
        "version": 1,
        "created_at": Value::Null,
        "backend_port": backend_port,
        "frontend_port": frontend_port,
        "processes": {
            "backend": {"pid": Value::Null, "pgid": Value::Null},
            "frontend": {"pid": Value::Null, "pgid": Value::Null}
        }
    })
}

fn start_state_value(
    backend_port: u64,
    frontend_port: u64,
    backend: Option<&std::process::Child>,
    frontend: Option<&std::process::Child>,
) -> Value {
    json!({
        "version": 1,
        "created_at": chrono::Utc::now().to_rfc3339(),
        "backend_port": backend_port,
        "frontend_port": frontend_port,
        "processes": {
            "backend": child_process_state(backend),
            "frontend": child_process_state(frontend)
        }
    })
}

fn child_process_state(child: Option<&std::process::Child>) -> Value {
    match child {
        Some(child) => {
            let pid = child.id();
            json!({
                "pid": pid,
                "pgid": child_process_group_id(pid)
            })
        }
        None => json!({"pid": Value::Null, "pgid": Value::Null}),
    }
}

#[cfg(unix)]
fn child_process_group_id(pid: u32) -> Value {
    json!(pid)
}

#[cfg(not(unix))]
fn child_process_group_id(_pid: u32) -> Value {
    Value::Null
}

fn remove_start_state(plan: &Value) {
    if let Some(path) = plan["state_path"].as_str() {
        let _ = fs::remove_file(path);
    }
}

fn cleanup_previous_start_state(plan: &Value) {
    let Some(path) = plan["state_path"].as_str() else {
        return;
    };
    let path = PathBuf::from(path);
    let state = fs::read_to_string(&path)
        .ok()
        .and_then(|text| serde_json::from_str::<Value>(&text).ok());
    let Some(state) = state else {
        let _ = fs::remove_file(path);
        return;
    };
    let records = start_state_process_records(&state);
    if !records.is_empty() {
        eprintln!("Found a stale Socartes launch state; cleaning it up first ...");
        terminate_start_state_records(&records, Duration::from_secs(1));
    }
    let _ = fs::remove_file(path);
}

fn start_state_process_records(state: &Value) -> Vec<(Option<u32>, Option<u32>)> {
    state["processes"]
        .as_object()
        .into_iter()
        .flat_map(|processes| processes.values())
        .filter_map(|record| {
            let pid = record["pid"]
                .as_u64()
                .and_then(|value| u32::try_from(value).ok());
            let pgid = record["pgid"]
                .as_u64()
                .and_then(|value| u32::try_from(value).ok());
            (pid.is_some() || pgid.is_some()).then_some((pid, pgid))
        })
        .collect()
}

fn terminate_start_state_records(records: &[(Option<u32>, Option<u32>)], grace: Duration) {
    let mut seen = Vec::<(Option<u32>, Option<u32>)>::new();
    for record in records {
        if !seen.contains(record) {
            seen.push(*record);
        }
    }
    for (pid, pgid) in &seen {
        signal_recorded_process(*pid, *pgid, "TERM");
    }
    let deadline = Instant::now() + grace;
    while Instant::now() < deadline {
        if seen.iter().all(|(pid, _)| !recorded_pid_alive(*pid)) {
            return;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    for (pid, pgid) in &seen {
        if recorded_pid_alive(*pid) {
            signal_recorded_process(*pid, *pgid, "KILL");
        }
    }
}

#[cfg(unix)]
fn recorded_pid_alive(pid: Option<u32>) -> bool {
    let Some(pid) = pid else {
        return false;
    };
    std::process::Command::new("kill")
        .args(["-0", &pid.to_string()])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

#[cfg(not(unix))]
fn recorded_pid_alive(_pid: Option<u32>) -> bool {
    false
}

#[cfg(unix)]
fn signal_recorded_process(pid: Option<u32>, pgid: Option<u32>, signal: &str) {
    let current_pid = std::process::id();
    if let Some(pgid) = pgid.filter(|pgid| *pgid != current_pid) {
        let status = std::process::Command::new("kill")
            .args([format!("-{signal}"), format!("-{pgid}")])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
        if matches!(status, Ok(status) if status.success()) {
            return;
        }
    }
    if let Some(pid) = pid.filter(|pid| *pid != current_pid) {
        let _ = std::process::Command::new("kill")
            .args([format!("-{signal}"), pid.to_string()])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
    }
}

#[cfg(not(unix))]
fn signal_recorded_process(_pid: Option<u32>, _pgid: Option<u32>, _signal: &str) {}

fn cleanup_started_processes(
    plan: &Value,
    backend: Option<&mut std::process::Child>,
    frontend: Option<&mut std::process::Child>,
) {
    if let Some(frontend) = frontend {
        terminate_child_tree(frontend, Duration::from_secs(5));
    }
    if let Some(backend) = backend {
        terminate_child_tree(backend, Duration::from_secs(5));
    }
    remove_start_state(plan);
}

fn terminate_child_tree(child: &mut std::process::Child, grace: Duration) {
    if child.try_wait().ok().flatten().is_some() {
        let _ = child.wait();
        return;
    }
    signal_child_tree(child, "TERM");
    if wait_child_until(child, grace) {
        return;
    }
    signal_child_tree(child, "KILL");
    let _ = child.wait();
}

fn wait_child_until(child: &mut std::process::Child, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => {
                let _ = child.wait();
                return true;
            }
            Ok(None) => {
                if Instant::now() >= deadline {
                    return false;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(_) => return true,
        }
    }
}

#[cfg(unix)]
fn signal_child_tree(child: &mut std::process::Child, signal: &str) {
    let pgid = child.id().to_string();
    let status = std::process::Command::new("kill")
        .args([format!("-{signal}"), format!("-{pgid}")])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
    if !matches!(status, Ok(status) if status.success()) {
        let _ = child.kill();
    }
}

#[cfg(not(unix))]
fn signal_child_tree(child: &mut std::process::Child, _signal: &str) {
    let _ = child.kill();
}

fn monitor_started_processes(
    backend: &mut std::process::Child,
    frontend: &mut std::process::Child,
) -> i32 {
    loop {
        match backend.try_wait() {
            Ok(Some(status)) => {
                terminate_child_tree(frontend, Duration::from_secs(5));
                return status.code().unwrap_or(1);
            }
            Ok(None) => {}
            Err(_) => return 1,
        }
        match frontend.try_wait() {
            Ok(Some(status)) => {
                terminate_child_tree(backend, Duration::from_secs(5));
                return status.code().unwrap_or(1);
            }
            Ok(None) => {}
            Err(_) => return 1,
        }
        std::thread::sleep(Duration::from_secs(1));
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
            print_chat_state(state)?;
            Ok(true)
        }
        ["/tool", "off", name] => {
            state.tools.retain(|tool| tool != name);
            print_chat_state(state)?;
            Ok(true)
        }
        ["/cap", name] => {
            state.capability = (*name).to_string();
            print_chat_state(state)?;
            Ok(true)
        }
        ["/kb", "none"] => {
            state.knowledge_bases.clear();
            print_chat_state(state)?;
            Ok(true)
        }
        ["/kb", name] => {
            state.knowledge_bases = vec![(*name).to_string()];
            print_chat_state(state)?;
            Ok(true)
        }
        ["/history", "add", id] => {
            state.history_references.push((*id).to_string());
            print_chat_state(state)?;
            Ok(true)
        }
        ["/history", "clear"] => {
            state.history_references.clear();
            print_chat_state(state)?;
            Ok(true)
        }
        ["/notebook", "add", reference] => {
            state
                .notebook_references
                .extend(parse_notebook_refs(&[(*reference).to_string()])?);
            print_chat_state(state)?;
            Ok(true)
        }
        ["/notebook", "clear"] => {
            state.notebook_references.clear();
            print_chat_state(state)?;
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
            print_chat_state(state)?;
            Ok(true)
        }
        ["/config", "set", item] => {
            let mut object = state.config.as_object().cloned().unwrap_or_default();
            let (key, value) = parse_config_item(item)?;
            object.insert(key, value);
            state.config = Value::Object(object);
            print_chat_state(state)?;
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
            let value = match api.get_json("/api/v1/memory").await {
                Ok(value) => value,
                Err(error) if should_use_local_memory(error.as_ref()) => local_memory_snapshot()?,
                Err(error) => return Err(error),
            };
            print_memory_value(&value, &file, format)
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
            let value = match api.post_json("/api/v1/memory/clear", &body).await {
                Ok(value) => value,
                Err(error) if should_use_local_memory(error.as_ref()) => {
                    clear_local_memory(&file)?;
                    let mut snapshot = local_memory_snapshot()?;
                    snapshot["cleared"] = json!(true);
                    snapshot
                }
                Err(error) => return Err(error),
            };
            print_json(&value)
        }
    }
}

fn print_memory_value(value: &Value, file: &str, format: OutputFormat) -> CliResult {
    if matches!(format, OutputFormat::Json) || file == "all" {
        return print_value(value, format);
    }
    if let Some(content) = value.get(file) {
        print_value(content, format)
    } else {
        Err(format!("Unknown memory file: {file}. Use summary, profile, or all.").into())
    }
}

fn should_use_local_memory(error: &(dyn std::error::Error + 'static)) -> bool {
    let mut current = Some(error);
    while let Some(error) = current {
        if let Some(reqwest_error) = error.downcast_ref::<reqwest::Error>()
            && (reqwest_error.is_connect() || reqwest_error.is_timeout())
        {
            return true;
        }
        current = error.source();
    }
    false
}

fn local_memory_snapshot() -> CliResult<Value> {
    let root = local_memory_root();
    Ok(json!({
        "summary": read_local_memory_file(&root, "SUMMARY.md")?,
        "profile": read_local_memory_file(&root, "PROFILE.md")?,
        "summary_updated_at": Value::Null,
        "profile_updated_at": Value::Null
    }))
}

fn local_memory_root() -> PathBuf {
    env::var_os("SOCARTES_MEMORY_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| data_root_from_home(None).join("memory"))
}

fn read_local_memory_file(root: &Path, filename: &str) -> CliResult<String> {
    let path = root.join(filename);
    if !path.exists() {
        return Ok(String::new());
    }
    Ok(fs::read_to_string(path)?.trim().to_string())
}

fn clear_local_memory(file: &str) -> CliResult {
    match file {
        "all" => {
            clear_local_memory_file("SUMMARY.md")?;
            clear_local_memory_file("PROFILE.md")
        }
        "summary" => clear_local_memory_file("SUMMARY.md"),
        "profile" => clear_local_memory_file("PROFILE.md"),
        _ => Err(format!("Unknown file: {file}. Use summary, profile, or all.").into()),
    }
}

fn clear_local_memory_file(filename: &str) -> CliResult {
    let root = local_memory_root();
    fs::create_dir_all(&root)?;
    let path = root.join(filename);
    if path.exists() {
        fs::remove_file(path)?;
    }
    Ok(())
}

async fn plugin_command(api: &ApiClient, command: PluginCommand) -> CliResult {
    match command {
        PluginCommand::List(args) => {
            let value = api.get_json("/api/v1/plugins/list").await?;
            print_value(&value, args.format)
        }
        PluginCommand::Info { name } => {
            let value = api.get_json("/api/v1/plugins/list").await?;
            for section in ["plugins", "tools", "capabilities"] {
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
    let paths = local_config_paths(home.as_deref())?;
    let env_values = read_env_file(&paths.env_file)?;
    let catalog = read_first_json_or_default(
        &[
            paths.user_settings_root.join("model_catalog.json"),
            paths.settings_root.join("model_catalog.json"),
            paths.settings_root.join("catalog.json"),
        ],
        default_cli_catalog(),
    )?;
    let ui = read_json_or_default(
        &paths.settings_root.join("ui.json"),
        default_cli_ui_settings(),
    )?;
    let main_yaml = read_main_yaml_summary(&[
        paths.user_settings_root.join("main.yaml"),
        paths.settings_root.join("main.yaml"),
    ])?;
    Ok(json!({
        "ports": local_ports_summary(&ui, &env_values),
        "llm": local_llm_summary(&catalog, &env_values),
        "embedding": local_embedding_summary(&catalog, &env_values),
        "search": local_search_summary(&catalog, &env_values),
        "language": local_language_summary(&ui, main_yaml.as_ref()),
        "tools": local_tools_summary(&ui, main_yaml.as_ref())
    }))
}

struct LocalConfigPaths {
    env_file: PathBuf,
    settings_root: PathBuf,
    user_settings_root: PathBuf,
}

fn local_config_paths(home: Option<&Path>) -> CliResult<LocalConfigPaths> {
    let data_root = match home {
        Some(home) => home.join("data"),
        None => env::var("SOCARTES_DATA_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("data")),
    };
    let project_root = match home {
        Some(home) => home.to_path_buf(),
        None => data_root
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from(".")),
    };
    Ok(LocalConfigPaths {
        env_file: project_root.join(".env"),
        settings_root: data_root.join("settings"),
        user_settings_root: data_root.join("user").join("settings"),
    })
}

fn data_root_from_home(home: Option<PathBuf>) -> PathBuf {
    match home {
        Some(home) => home.join("data"),
        None => env::var("SOCARTES_DATA_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("data")),
    }
}

fn read_first_json_or_default(paths: &[PathBuf], default: Value) -> CliResult<Value> {
    for path in paths {
        match fs::read_to_string(path) {
            Ok(text) => return Ok(serde_json::from_str(&text)?),
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
    }
    Ok(default)
}

fn read_json_or_default(path: &Path, default: Value) -> CliResult<Value> {
    match fs::read_to_string(path) {
        Ok(text) => Ok(serde_json::from_str(&text)?),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(default),
        Err(error) => Err(error.into()),
    }
}

#[derive(Debug, Default)]
struct MainYamlSummary {
    language: Option<String>,
    tools: Option<Vec<String>>,
}

fn read_main_yaml_summary(paths: &[PathBuf]) -> CliResult<Option<MainYamlSummary>> {
    for path in paths {
        match fs::read_to_string(path) {
            Ok(text) => return Ok(Some(parse_main_yaml_summary(&text))),
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
    }
    Ok(None)
}

fn parse_main_yaml_summary(text: &str) -> MainYamlSummary {
    let mut summary = MainYamlSummary::default();
    let mut in_system = false;
    let mut in_tools = false;
    let mut tools = Vec::new();
    for raw_line in text.lines() {
        let without_comment = raw_line.split_once('#').map_or(raw_line, |(head, _)| head);
        let trimmed = without_comment.trim();
        if trimmed.is_empty() {
            continue;
        }
        let indent = without_comment
            .chars()
            .take_while(|value| value.is_whitespace())
            .count();
        if indent == 0 {
            in_tools = false;
            in_system = false;
            if let Some((key, _)) = trimmed.split_once(':') {
                let key = clean_yaml_scalar(key);
                in_system = key == "system";
                if key == "tools" {
                    in_tools = true;
                    summary.tools = Some(Vec::new());
                }
            }
            continue;
        }
        if in_system && let Some(value) = trimmed.strip_prefix("language:") {
            let language = clean_yaml_scalar(value);
            if !language.is_empty() {
                summary.language = Some(language);
            }
            continue;
        }
        if in_tools
            && indent > 0
            && let Some((key, _)) = trimmed.split_once(':')
        {
            let key = clean_yaml_scalar(key);
            if !key.is_empty() {
                tools.push(key);
            }
            summary.tools = Some(tools.clone());
        }
    }
    summary
}

fn clean_yaml_scalar(value: &str) -> String {
    value
        .trim()
        .trim_matches('"')
        .trim_matches('\'')
        .to_string()
}

fn local_ports_summary(ui: &Value, env_values: &BTreeMap<String, String>) -> Value {
    json!({
        "backend": env_u16_from_config(env_values, "BACKEND_PORT")
            .map(Value::from)
            .or_else(|| ui.pointer("/ports/backend").cloned())
            .or_else(|| ui.get("backend_port").cloned())
            .unwrap_or_else(|| json!(8001)),
        "frontend": env_u16_from_config(env_values, "FRONTEND_PORT")
            .map(Value::from)
            .or_else(|| ui.pointer("/ports/frontend").cloned())
            .or_else(|| ui.get("frontend_port").cloned())
            .unwrap_or_else(|| json!(3782))
    })
}

fn env_config_value(env_values: &BTreeMap<String, String>, key: &str) -> Option<String> {
    if let Some(value) = env_values.get(key) {
        return Some(value.trim().to_string());
    }
    env::var(key).ok().map(|value| value.trim().to_string())
}

fn env_non_empty(env_values: &BTreeMap<String, String>, key: &str) -> Option<String> {
    env_config_value(env_values, key).filter(|value| !value.is_empty())
}

fn env_u16_from_config(env_values: &BTreeMap<String, String>, key: &str) -> Option<u16> {
    env_config_value(env_values, key).and_then(|value| value.parse::<u16>().ok())
}

fn local_llm_summary(catalog: &Value, env_values: &BTreeMap<String, String>) -> Value {
    let service = &catalog["services"]["llm"];
    let profile = active_service_profile(catalog, "llm");
    let model = active_service_model(service, profile);
    let binding = profile_string_option(profile, "binding")
        .or_else(|| env_config_value(env_values, "LLM_BINDING"))
        .map(|value| canonical_provider_name(&value))
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "openai".to_string());
    let model_name = model_string(model, "model")
        .or_else(|| model_string(model, "id"))
        .or_else(|| env_non_empty(env_values, "LLM_MODEL"))
        .or_else(|| service["active_model_id"].as_str().map(str::to_string))
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "gpt-4o-mini".to_string());
    let api_key = profile_string_option(profile, "api_key")
        .or_else(|| env_config_value(env_values, "LLM_API_KEY"))
        .unwrap_or_default();
    let configured_base_url = profile_string_option(profile, "base_url")
        .or_else(|| env_non_empty(env_values, "LLM_HOST"))
        .or_else(|| env_non_empty(env_values, "LLM_BASE_URL"));
    let provider = resolve_llm_provider(
        &binding,
        &model_name,
        &api_key,
        configured_base_url.as_deref(),
    );
    let base_url = configured_base_url
        .or_else(|| default_llm_base_url(&provider).map(str::to_string))
        .unwrap_or_default();
    json!({
        "binding_hint": binding,
        "provider": provider,
        "provider_mode": local_provider_mode(&provider),
        "model": model_name,
        "base_url": base_url,
        "api_version": profile_string_option(profile, "api_version")
            .or_else(|| env_non_empty(env_values, "LLM_API_VERSION"))
            .unwrap_or_default(),
        "extra_headers": profile
            .and_then(|value| value.get("extra_headers"))
            .cloned()
            .unwrap_or_else(|| json!({})),
        "api_key": masked_secret(api_key)
    })
}

fn local_embedding_summary(catalog: &Value, env_values: &BTreeMap<String, String>) -> Value {
    let service = &catalog["services"]["embedding"];
    let profile = active_service_profile(catalog, "embedding");
    let model = active_service_model(service, profile);
    let binding = profile_string_option(profile, "binding")
        .or_else(|| env_config_value(env_values, "EMBEDDING_BINDING"))
        .map(|value| canonical_provider_name(&value))
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "openai".to_string());
    let model_name = model_string(model, "model")
        .or_else(|| model_string(model, "id"))
        .or_else(|| env_non_empty(env_values, "EMBEDDING_MODEL"))
        .or_else(|| service["active_model_id"].as_str().map(str::to_string))
        .unwrap_or_default();
    let api_key = profile_string_option(profile, "api_key")
        .or_else(|| env_config_value(env_values, "EMBEDDING_API_KEY"))
        .unwrap_or_default();
    let base_url = profile_string_option(profile, "base_url")
        .or_else(|| env_non_empty(env_values, "EMBEDDING_HOST"))
        .or_else(|| env_non_empty(env_values, "EMBEDDING_BASE_URL"))
        .unwrap_or_default();
    let dimension = model
        .and_then(|value| value.get("dimension"))
        .and_then(json_i64)
        .or_else(|| {
            env_config_value(env_values, "EMBEDDING_DIMENSION").and_then(|value| value.parse().ok())
        })
        .map(Value::from)
        .unwrap_or(Value::Null);
    let provider = resolve_embedding_provider(&binding, &model_name, &base_url);
    json!({
        "binding_hint": binding,
        "provider": provider,
        "provider_mode": local_provider_mode(&provider),
        "model": model_name,
        "base_url": base_url,
        "api_version": profile_string_option(profile, "api_version")
            .or_else(|| env_non_empty(env_values, "EMBEDDING_API_VERSION"))
            .unwrap_or_default(),
        "extra_headers": profile
            .and_then(|value| value.get("extra_headers"))
            .cloned()
            .unwrap_or_else(|| json!({})),
        "api_key": masked_secret(api_key),
        "dimension": dimension
    })
}

fn local_search_summary(catalog: &Value, env_values: &BTreeMap<String, String>) -> Value {
    let profile = active_service_profile(catalog, "search");
    let requested_provider = profile_string_option(profile, "provider")
        .or_else(|| env_config_value(env_values, "SEARCH_PROVIDER"))
        .map(|value| canonical_provider_name(&value))
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "brave".to_string());
    let api_key = profile_string_option(profile, "api_key")
        .or_else(|| env_config_value(env_values, "SEARCH_API_KEY"))
        .or_else(|| search_provider_env_key(&requested_provider))
        .unwrap_or_default();
    let base_url = profile_string_option(profile, "base_url")
        .or_else(|| env_non_empty(env_values, "SEARCH_BASE_URL"))
        .unwrap_or_default();
    let proxy = profile_string_option(profile, "proxy")
        .or_else(|| env_non_empty(env_values, "SEARCH_PROXY"))
        .unwrap_or_default();
    let (provider, status, fallback_reason) =
        resolve_search_provider(&requested_provider, &api_key, &base_url);
    json!({
        "provider": provider,
        "requested_provider": requested_provider,
        "status": status,
        "fallback_reason": fallback_reason.map(Value::from).unwrap_or(Value::Null),
        "base_url": base_url,
        "proxy": proxy,
        "api_key": masked_secret(api_key)
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

fn canonical_provider_name(value: &str) -> String {
    match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "openai_compatible" | "openai_compat" => "openai".to_string(),
        value => value.to_string(),
    }
}

fn resolve_llm_provider(
    binding: &str,
    model: &str,
    api_key: &str,
    base_url: Option<&str>,
) -> String {
    let base_url = base_url.unwrap_or("").to_ascii_lowercase();
    if binding == "openrouter"
        || api_key.starts_with("sk-or-")
        || base_url.contains("openrouter.ai")
    {
        return "openrouter".to_string();
    }
    let model_lower = model.to_ascii_lowercase();
    if binding == "openai" && model_lower.starts_with("claude") {
        return "anthropic".to_string();
    }
    if binding == "openai" && _is_local_url(&base_url) {
        return if base_url.contains("11434") {
            "ollama".to_string()
        } else {
            "vllm".to_string()
        };
    }
    binding.to_string()
}

fn default_llm_base_url(provider: &str) -> Option<&'static str> {
    match provider {
        "openrouter" => Some("https://openrouter.ai/api/v1"),
        "anthropic" => Some("https://api.anthropic.com"),
        _ => None,
    }
}

fn resolve_embedding_provider(binding: &str, model: &str, base_url: &str) -> String {
    let model = model.to_ascii_lowercase();
    let base_url = base_url.to_ascii_lowercase();
    if binding != "openai" {
        return binding.to_string();
    }
    if model.contains("gemini") {
        return "gemini".to_string();
    }
    if model.contains("cohere") || model.contains("embed-v4") {
        return "cohere".to_string();
    }
    if model.contains("jina") {
        return "jina".to_string();
    }
    if _is_local_url(&base_url) {
        return if base_url.contains("11434") {
            "ollama".to_string()
        } else {
            "vllm".to_string()
        };
    }
    "openai".to_string()
}

fn _is_local_url(value: &str) -> bool {
    value.contains("localhost")
        || value.contains("127.0.0.1")
        || value.contains("::1")
        || value.contains(".local")
}

fn search_provider_env_key(provider: &str) -> Option<String> {
    let key = match provider {
        "brave" => "BRAVE_API_KEY",
        "tavily" => "TAVILY_API_KEY",
        "jina" => "JINA_API_KEY",
        "perplexity" => "PERPLEXITY_API_KEY",
        "serper" => "SERPER_API_KEY",
        _ => return None,
    };
    env::var(key)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn resolve_search_provider(
    requested_provider: &str,
    api_key: &str,
    base_url: &str,
) -> (String, String, Option<String>) {
    if matches!(requested_provider, "exa" | "baidu" | "openrouter") {
        return (
            requested_provider.to_string(),
            "unsupported".to_string(),
            Some(format!(
                "{requested_provider} is deprecated/unsupported. Switch to brave/tavily/jina/searxng/duckduckgo/perplexity/serper."
            )),
        );
    }
    if !matches!(
        requested_provider,
        "brave" | "tavily" | "jina" | "searxng" | "duckduckgo" | "perplexity" | "serper"
    ) {
        return (
            requested_provider.to_string(),
            "unsupported".to_string(),
            Some(format!("Unsupported search provider: {requested_provider}")),
        );
    }
    if matches!(requested_provider, "brave" | "tavily" | "jina") && api_key.is_empty() {
        return (
            "duckduckgo".to_string(),
            "fallback".to_string(),
            Some(format!(
                "{requested_provider} requires api_key, falling back to duckduckgo"
            )),
        );
    }
    if requested_provider == "searxng" && base_url.is_empty() {
        return (
            "duckduckgo".to_string(),
            "fallback".to_string(),
            Some("searxng requires base_url, falling back to duckduckgo".to_string()),
        );
    }
    if matches!(requested_provider, "perplexity" | "serper") && api_key.is_empty() {
        return (
            requested_provider.to_string(),
            "not_configured".to_string(),
            Some(format!("{requested_provider} requires api_key")),
        );
    }
    (
        requested_provider.to_string(),
        "configured".to_string(),
        None,
    )
}

fn local_provider_mode(binding: &str) -> &str {
    match binding {
        "" => "",
        "openrouter" => "gateway",
        "openai" | "openai-compatible" => "openai-compatible",
        "anthropic" => "anthropic",
        "local" | "ollama" | "vllm" => "local",
        _ => "custom",
    }
}

fn profile_string_option(profile: Option<&Value>, key: &str) -> Option<String> {
    profile
        .and_then(|value| value[key].as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn model_string(model: Option<&Value>, key: &str) -> Option<String> {
    model
        .and_then(|value| value[key].as_str().map(str::trim))
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn json_i64(value: &Value) -> Option<i64> {
    value
        .as_i64()
        .or_else(|| value.as_str().and_then(|value| value.parse::<i64>().ok()))
}

fn masked_secret(secret: String) -> &'static str {
    if secret.trim().is_empty() {
        "(not set)"
    } else {
        "***"
    }
}

fn local_language_summary(ui: &Value, main_yaml: Option<&MainYamlSummary>) -> String {
    main_yaml
        .and_then(|summary| summary.language.clone())
        .or_else(|| ui["language"].as_str().map(str::to_string))
        .unwrap_or_else(|| "en".to_string())
}

fn local_tools_summary(ui: &Value, main_yaml: Option<&MainYamlSummary>) -> Value {
    if let Some(tools) = main_yaml.and_then(|summary| summary.tools.as_ref()) {
        return Value::Array(tools.iter().map(|key| json!(key)).collect());
    }
    match ui.get("tools").and_then(Value::as_object) {
        Some(tools) => Value::Array(tools.keys().map(|key| json!(key)).collect()),
        None => json!([]),
    }
}

async fn session_command(api: &ApiClient, command: SessionCommand) -> CliResult {
    match command {
        SessionCommand::List { limit, format } => {
            let mut value = api
                .get_json(&format!("/api/v1/sessions?limit={limit}"))
                .await?;
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
    if openai_codex_access_token().is_some() {
        println!("OpenAI Codex OAuth authentication succeeded.");
        return Ok(());
    }
    if run_openai_codex_helper()? {
        println!("OpenAI Codex OAuth authentication succeeded.");
        return Ok(());
    }
    Err("OpenAI Codex OAuth authentication failed: no existing OAuth token was found. Set SOCARTES_OPENAI_CODEX_ACCESS_TOKEN or run `codex login` / the OpenAI Codex login flow.".into())
}

fn openai_codex_access_token() -> Option<String> {
    env_token([
        "SOCARTES_OPENAI_CODEX_ACCESS_TOKEN",
        "OPENAI_CODEX_ACCESS_TOKEN",
    ])
    .or_else(|| read_codex_auth_access_token(&codex_auth_file()?))
}

fn env_token<const N: usize>(names: [&str; N]) -> Option<String> {
    names
        .iter()
        .find_map(|name| env::var(name).ok())
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn codex_auth_file() -> Option<PathBuf> {
    let home = env::var_os("CODEX_HOME")
        .map(PathBuf::from)
        .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join(".codex")))?;
    Some(home.join("auth.json"))
}

fn read_codex_auth_access_token(path: &Path) -> Option<String> {
    let text = fs::read_to_string(path).ok()?;
    let value = serde_json::from_str::<Value>(&text).ok()?;
    value
        .pointer("/tokens/access_token")
        .and_then(Value::as_str)
        .or_else(|| value.get("access_token").and_then(Value::as_str))
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .map(ToString::to_string)
}

fn run_openai_codex_helper() -> CliResult<bool> {
    let Some(helper) = env::var_os("SOCARTES_OPENAI_CODEX_HELPER")
        .map(PathBuf::from)
        .filter(|path| !path.as_os_str().is_empty())
    else {
        return Ok(false);
    };
    let output = std::process::Command::new(&helper)
        .output()
        .map_err(|error| {
            format!(
                "OpenAI Codex OAuth helper `{}` failed to start: {error}",
                helper.display()
            )
        })?;
    if output.status.success() {
        return Ok(true);
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    Err(format!(
        "OpenAI Codex OAuth helper `{}` failed with status {}: {}{}{}",
        helper.display(),
        output
            .status
            .code()
            .map(|code| code.to_string())
            .unwrap_or_else(|| "terminated by signal".to_string()),
        stderr.trim(),
        if stdout.trim().is_empty() || stderr.trim().is_empty() {
            ""
        } else {
            "\n"
        },
        stdout.trim()
    )
    .into())
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
    let mut runtime = args.runtime;
    let mut prompt_for_runtime = false;
    if let Some(command) = args.command {
        let InitCommand::Wizard(wizard) = command;
        yes |= wizard.yes;
        cli_only |= wizard.cli;
        home = home.or(wizard.home);
        runtime.merge(wizard.runtime);
        prompt_for_runtime = !yes;
    }
    run_init_wizard(
        InitWizardArgs {
            yes,
            cli: cli_only,
            home,
            runtime,
        },
        prompt_for_runtime,
    )
}

impl RuntimeInitOptions {
    fn merge(&mut self, other: RuntimeInitOptions) {
        if other.llm_binding.is_some() {
            self.llm_binding = other.llm_binding;
        }
        if other.llm_base_url.is_some() {
            self.llm_base_url = other.llm_base_url;
        }
        if other.llm_api_key.is_some() {
            self.llm_api_key = other.llm_api_key;
        }
        if other.llm_model.is_some() {
            self.llm_model = other.llm_model;
        }
        if other.embedding_binding.is_some() {
            self.embedding_binding = other.embedding_binding;
        }
        if other.embedding_base_url.is_some() {
            self.embedding_base_url = other.embedding_base_url;
        }
        if other.embedding_api_key.is_some() {
            self.embedding_api_key = other.embedding_api_key;
        }
        if other.embedding_model.is_some() {
            self.embedding_model = other.embedding_model;
        }
        if other.embedding_dimension.is_some() {
            self.embedding_dimension = other.embedding_dimension;
        }
        if other.search_provider.is_some() {
            self.search_provider = other.search_provider;
        }
        if other.search_base_url.is_some() {
            self.search_base_url = other.search_base_url;
        }
        if other.search_api_key.is_some() {
            self.search_api_key = other.search_api_key;
        }
        if other.backend_port.is_some() {
            self.backend_port = other.backend_port;
        }
        if other.frontend_port.is_some() {
            self.frontend_port = other.frontend_port;
        }
        if other.language.is_some() {
            self.language = other.language;
        }
    }

    fn has_catalog_options(&self) -> bool {
        self.llm_binding.is_some()
            || self.llm_base_url.is_some()
            || self.llm_api_key.is_some()
            || self.llm_model.is_some()
            || self.embedding_binding.is_some()
            || self.embedding_base_url.is_some()
            || self.embedding_api_key.is_some()
            || self.embedding_model.is_some()
            || self.embedding_dimension.is_some()
            || self.search_provider.is_some()
            || self.search_base_url.is_some()
            || self.search_api_key.is_some()
    }
}

fn run_init_wizard(mut args: InitWizardArgs, prompt_for_runtime: bool) -> CliResult {
    let data_root = match args.home.as_deref() {
        Some(home) => home.join("data"),
        None => env::var("SOCARTES_DATA_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("data")),
    };
    let project_root = init_project_root(args.home.as_deref(), &data_root);
    if !args.yes {
        println!(
            "This will create local Socartes runtime directories under {}.",
            data_root.display()
        );
        if !confirm("Continue?")? {
            return Ok(());
        }
    }
    if prompt_for_runtime {
        prompt_runtime_options(&mut args.runtime)?;
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
        data_root.join("user").join("settings"),
        data_root.join("auth").join("users"),
        data_root.join("tutorbot"),
        data_root.join("skills"),
    ];
    for dir in &dirs {
        fs::create_dir_all(dir)?;
    }

    let settings_root = data_root.join("settings");
    write_json_if_absent(
        &settings_root.join("catalog.json"),
        &configured_cli_catalog(&args.runtime),
    )?;
    write_json_if_absent(
        &settings_root.join("ui.json"),
        &configured_cli_ui_settings(&args.runtime),
    )?;
    write_init_env_file(&project_root.join(".env"), &args.runtime)?;
    write_interface_settings(
        &data_root
            .join("user")
            .join("settings")
            .join("interface.json"),
        &args.runtime,
    )?;
    write_json_if_absent(
        &data_root.join("knowledge").join("kb_config.json"),
        &json!({ "knowledge_bases": {} }),
    )?;

    print_json(&json!({
        "initialized": true,
        "cli_only": args.cli,
        "data_dir": data_root,
        "settings": settings_root,
        "env_file": project_root.join(".env"),
        "interface_settings": data_root.join("user").join("settings").join("interface.json"),
        "created": dirs
    }))
}

fn init_project_root(home: Option<&Path>, data_root: &Path) -> PathBuf {
    match home {
        Some(home) => home.to_path_buf(),
        None => data_root
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from(".")),
    }
}

fn prompt_runtime_options(options: &mut RuntimeInitOptions) -> CliResult {
    println!("Configure Socartes runtime. Press Enter to keep defaults.");
    prompt_option("LLM binding", &mut options.llm_binding)?;
    prompt_option("LLM base URL", &mut options.llm_base_url)?;
    prompt_option("LLM API key", &mut options.llm_api_key)?;
    prompt_option("LLM model", &mut options.llm_model)?;
    prompt_option("Embedding binding", &mut options.embedding_binding)?;
    prompt_option("Embedding base URL", &mut options.embedding_base_url)?;
    prompt_option("Embedding API key", &mut options.embedding_api_key)?;
    prompt_option("Embedding model", &mut options.embedding_model)?;
    options.embedding_dimension =
        prompt_parse_option("Embedding dimension", options.embedding_dimension)?;
    prompt_option("Search provider", &mut options.search_provider)?;
    prompt_option("Search base URL", &mut options.search_base_url)?;
    prompt_option("Search API key", &mut options.search_api_key)?;
    options.backend_port = prompt_parse_option("Backend port", options.backend_port)?;
    options.frontend_port = prompt_parse_option("Frontend port", options.frontend_port)?;
    prompt_option("Language", &mut options.language)?;
    Ok(())
}

fn prompt_option(label: &str, value: &mut Option<String>) -> CliResult {
    if let Some(answer) = prompt_text(label, value.as_deref())? {
        *value = Some(answer);
    }
    Ok(())
}

fn prompt_parse_option<T>(label: &str, value: Option<T>) -> CliResult<Option<T>>
where
    T: std::str::FromStr + Copy + std::fmt::Display,
    T::Err: std::fmt::Display,
{
    let default = value.as_ref().map(ToString::to_string);
    match prompt_text(label, default.as_deref())? {
        Some(answer) => answer
            .parse::<T>()
            .map(Some)
            .map_err(|error| format!("Invalid {label}: {error}").into()),
        None => Ok(value),
    }
}

fn prompt_text(label: &str, default: Option<&str>) -> CliResult<Option<String>> {
    match default.map(str::trim).filter(|value| !value.is_empty()) {
        Some(default) => print!("{label} [{default}]: "),
        None => print!("{label}: "),
    }
    io::stdout().flush()?;
    let mut answer = String::new();
    io::stdin().read_line(&mut answer)?;
    let answer = answer.trim().to_string();
    if answer.is_empty() {
        Ok(None)
    } else {
        Ok(Some(answer))
    }
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

fn write_init_env_file(path: &Path, options: &RuntimeInitOptions) -> CliResult {
    let mut current = read_env_file(path)?;
    for (key, value) in configured_cli_env(options) {
        current.insert(key.to_string(), value);
    }
    let key_order = [
        "BACKEND_PORT",
        "FRONTEND_PORT",
        "LLM_BINDING",
        "LLM_MODEL",
        "LLM_API_KEY",
        "LLM_HOST",
        "LLM_API_VERSION",
        "EMBEDDING_BINDING",
        "EMBEDDING_MODEL",
        "EMBEDDING_API_KEY",
        "EMBEDDING_HOST",
        "EMBEDDING_DIMENSION",
        "EMBEDDING_SEND_DIMENSIONS",
        "EMBEDDING_API_VERSION",
        "SEARCH_PROVIDER",
        "SEARCH_API_KEY",
        "SEARCH_BASE_URL",
        "SEARCH_PROXY",
    ];
    let mut rendered = String::new();
    for key in key_order {
        if key == "SEARCH_BASE_URL"
            && current
                .get(key)
                .map(|value| value.is_empty())
                .unwrap_or(true)
        {
            continue;
        }
        if key == "EMBEDDING_SEND_DIMENSIONS"
            && current
                .get(key)
                .map(|value| value.is_empty())
                .unwrap_or(true)
        {
            continue;
        }
        rendered.push_str(key);
        rendered.push('=');
        rendered.push_str(current.get(key).map(String::as_str).unwrap_or(""));
        rendered.push('\n');
    }
    for (key, value) in current {
        if !key_order.contains(&key.as_str()) {
            rendered.push_str(&key);
            rendered.push('=');
            rendered.push_str(&value);
            rendered.push('\n');
        }
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, rendered)?;
    Ok(())
}

fn configured_cli_env(options: &RuntimeInitOptions) -> Vec<(&'static str, String)> {
    vec![
        (
            "BACKEND_PORT",
            options.backend_port.unwrap_or(8001).to_string(),
        ),
        (
            "FRONTEND_PORT",
            options.frontend_port.unwrap_or(3782).to_string(),
        ),
        (
            "LLM_BINDING",
            option_or_default(&options.llm_binding, "openai").to_string(),
        ),
        (
            "LLM_MODEL",
            option_or_default(&options.llm_model, "").to_string(),
        ),
        (
            "LLM_API_KEY",
            option_or_default(&options.llm_api_key, "").to_string(),
        ),
        (
            "LLM_HOST",
            option_or_default(&options.llm_base_url, "").to_string(),
        ),
        ("LLM_API_VERSION", String::new()),
        (
            "EMBEDDING_BINDING",
            option_or_default(&options.embedding_binding, "openai").to_string(),
        ),
        (
            "EMBEDDING_MODEL",
            option_or_default(&options.embedding_model, "").to_string(),
        ),
        (
            "EMBEDDING_API_KEY",
            option_or_default(&options.embedding_api_key, "").to_string(),
        ),
        (
            "EMBEDDING_HOST",
            option_or_default(&options.embedding_base_url, "").to_string(),
        ),
        (
            "EMBEDDING_DIMENSION",
            options
                .embedding_dimension
                .map(|value| value.to_string())
                .unwrap_or_default(),
        ),
        ("EMBEDDING_SEND_DIMENSIONS", String::new()),
        ("EMBEDDING_API_VERSION", String::new()),
        (
            "SEARCH_PROVIDER",
            option_or_default(&options.search_provider, "").to_string(),
        ),
        (
            "SEARCH_API_KEY",
            option_or_default(&options.search_api_key, "").to_string(),
        ),
        (
            "SEARCH_BASE_URL",
            option_or_default(&options.search_base_url, "").to_string(),
        ),
        ("SEARCH_PROXY", String::new()),
    ]
}

fn write_interface_settings(path: &Path, options: &RuntimeInitOptions) -> CliResult {
    let mut payload = match fs::read_to_string(path) {
        Ok(text) => serde_json::from_str::<Value>(&text).unwrap_or_else(|_| json!({})),
        Err(error) if error.kind() == io::ErrorKind::NotFound => json!({}),
        Err(error) => return Err(error.into()),
    };
    if !payload.is_object() {
        payload = json!({});
    }
    payload["theme"] = payload
        .get("theme")
        .cloned()
        .filter(|value| value.is_string())
        .unwrap_or_else(|| json!("light"));
    payload["language"] = json!(
        options
            .language
            .as_deref()
            .map(normalize_init_language)
            .unwrap_or("en")
    );
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_vec_pretty(&payload)?)?;
    Ok(())
}

fn normalize_init_language(language: &str) -> &str {
    match language.trim().to_ascii_lowercase().as_str() {
        "zh" | "cn" | "chinese" => "zh",
        "ko" | "kr" | "korean" => "ko",
        _ => "en",
    }
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

fn configured_cli_ui_settings(options: &RuntimeInitOptions) -> Value {
    let mut ui = default_cli_ui_settings();
    if let Some(language) = &options.language {
        ui["language"] = json!(language);
    }
    if options.backend_port.is_some() || options.frontend_port.is_some() {
        ui["ports"] = json!({
            "backend": options.backend_port.unwrap_or(8001),
            "frontend": options.frontend_port.unwrap_or(3782)
        });
    }
    ui
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

fn configured_cli_catalog(options: &RuntimeInitOptions) -> Value {
    if !options.has_catalog_options() {
        return default_cli_catalog();
    }
    let llm_model = option_or_default(&options.llm_model, "socartes-llm");
    let embedding_model = option_or_default(&options.embedding_model, "socartes-embedding");
    let embedding_dimension = options.embedding_dimension.unwrap_or(3072);
    json!({
        "version": 1,
        "services": {
            "llm": {
                "active_profile_id": "llm-main",
                "active_model_id": llm_model,
                "profiles": [{
                    "id": "llm-main",
                    "name": "Socartes LLM",
                    "binding": option_or_default(&options.llm_binding, "openai"),
                    "base_url": option_or_default(&options.llm_base_url, "http://127.0.0.1:8810/v1"),
                    "api_key": option_or_default(&options.llm_api_key, ""),
                    "api_version": "",
                    "extra_headers": {},
                    "models": [{
                        "id": llm_model,
                        "name": llm_model,
                        "model": llm_model,
                        "context_window": "8192",
                        "context_window_source": "cli-init"
                    }]
                }]
            },
            "embedding": {
                "active_profile_id": "embedding-main",
                "active_model_id": embedding_model,
                "profiles": [{
                    "id": "embedding-main",
                    "name": "Socartes Embedding",
                    "binding": option_or_default(&options.embedding_binding, "openai"),
                    "base_url": option_or_default(&options.embedding_base_url, "http://127.0.0.1:8810/v1"),
                    "api_key": option_or_default(&options.embedding_api_key, ""),
                    "api_version": "",
                    "extra_headers": {},
                    "models": [{
                        "id": embedding_model,
                        "name": embedding_model,
                        "model": embedding_model,
                        "dimension": embedding_dimension,
                        "send_dimensions": true,
                        "supported_dimensions": embedding_dimension.to_string()
                    }]
                }]
            },
            "search": {
                "active_profile_id": "search-main",
                "profiles": [{
                    "id": "search-main",
                    "name": "Socartes Search",
                    "provider": option_or_default(&options.search_provider, "duckduckgo"),
                    "base_url": option_or_default(&options.search_base_url, ""),
                    "api_key": option_or_default(&options.search_api_key, ""),
                    "api_version": "",
                    "proxy": "",
                    "max_results": 5,
                    "models": []
                }]
            }
        }
    })
}

fn option_or_default<'a>(value: &'a Option<String>, default: &'a str) -> &'a str {
    value
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or(default)
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
        collect_supported_files_recursively(&dir, &mut collected)?;
    }
    Ok(collected.into_values().collect())
}

fn collect_supported_files_recursively(
    dir: &Path,
    collected: &mut BTreeMap<String, PathBuf>,
) -> CliResult {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_supported_files_recursively(&path, collected)?;
        } else if path.is_file() && is_python_supported_kb_file(&path) {
            let path = path.canonicalize()?;
            collected.insert(path.to_string_lossy().to_string(), path);
        }
    }
    Ok(())
}

fn is_python_supported_kb_file(path: &Path) -> bool {
    path.extension()
        .and_then(|value| value.to_str())
        .map(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "pdf"
                    | "docx"
                    | "xlsx"
                    | "pptx"
                    | "txt"
                    | "text"
                    | "log"
                    | "md"
                    | "markdown"
                    | "rst"
                    | "asciidoc"
                    | "json"
                    | "jsonc"
                    | "json5"
                    | "yaml"
                    | "yml"
                    | "toml"
                    | "csv"
                    | "tsv"
                    | "ini"
                    | "cfg"
                    | "conf"
                    | "env"
                    | "properties"
                    | "tex"
                    | "latex"
                    | "bib"
                    | "js"
                    | "mjs"
                    | "cjs"
                    | "ts"
                    | "mts"
                    | "cts"
                    | "jsx"
                    | "tsx"
                    | "vue"
                    | "svelte"
                    | "py"
                    | "java"
                    | "kt"
                    | "kts"
                    | "scala"
                    | "groovy"
                    | "gradle"
                    | "c"
                    | "h"
                    | "cpp"
                    | "cc"
                    | "cxx"
                    | "hpp"
                    | "hh"
                    | "hxx"
                    | "cs"
                    | "go"
                    | "rs"
                    | "zig"
                    | "nim"
                    | "swift"
                    | "m"
                    | "mm"
                    | "rb"
                    | "php"
                    | "pl"
                    | "pm"
                    | "lua"
                    | "r"
                    | "jl"
                    | "dart"
                    | "hs"
                    | "clj"
                    | "cljs"
                    | "cljc"
                    | "ex"
                    | "exs"
                    | "erl"
                    | "ml"
                    | "mli"
                    | "fs"
                    | "fsx"
                    | "lisp"
                    | "lsp"
                    | "scm"
                    | "rkt"
                    | "html"
                    | "htm"
                    | "xml"
                    | "svg"
                    | "css"
                    | "scss"
                    | "sass"
                    | "less"
                    | "sol"
                    | "sh"
                    | "bash"
                    | "zsh"
                    | "fish"
                    | "ps1"
                    | "vim"
                    | "sql"
                    | "graphql"
                    | "gql"
                    | "proto"
                    | "cmake"
                    | "mk"
                    | "tf"
                    | "hcl"
                    | "nginxconf"
                    | "dockerfile"
            )
        })
        .unwrap_or(false)
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
        value[key].as_array().or_else(|| value.as_array())
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
