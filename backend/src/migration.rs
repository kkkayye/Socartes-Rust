use std::{
    collections::{HashMap, HashSet},
    env, fs, io,
    path::{Path, PathBuf},
    sync::Arc,
};

use arc_swap::ArcSwap;
use axum::{
    body::Body,
    extract::{
        Request,
        ws::{Message, WebSocket, WebSocketUpgrade},
    },
    http::{
        HeaderMap, HeaderName, HeaderValue, StatusCode, Uri,
        header::{
            CACHE_CONTROL, CONNECTION, CONTENT_TYPE, HOST, PROXY_AUTHENTICATE, PROXY_AUTHORIZATION,
            TE, TRAILER, TRANSFER_ENCODING, UPGRADE,
        },
    },
    response::{IntoResponse, Response},
};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio_tungstenite::{
    connect_async,
    tungstenite::{Message as TungsteniteMessage, client::IntoClientRequest},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum MigrationMode {
    #[default]
    Native,
    Proxy,
    Shadow,
}

impl MigrationMode {
    pub fn should_proxy(self) -> bool {
        matches!(self, Self::Proxy | Self::Shadow)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MigrationConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_python_base_url")]
    pub python_base_url: String,
    #[serde(default = "default_python_ws_base_url")]
    pub python_ws_base_url: String,
    #[serde(default = "default_fallback_mode")]
    pub fallback: MigrationMode,
    #[serde(default)]
    pub routes: HashMap<String, MigrationMode>,
}

impl Default for MigrationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            python_base_url: default_python_base_url(),
            python_ws_base_url: default_python_ws_base_url(),
            fallback: default_fallback_mode(),
            routes: HashMap::new(),
        }
    }
}

impl MigrationConfig {
    pub fn from_toml_str(input: &str) -> Result<Self, toml::de::Error> {
        let mut config: Self = toml::from_str(input)?;
        config.normalize();
        Ok(config)
    }

    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, MigrationConfigError> {
        let content = fs::read_to_string(path.as_ref()).map_err(MigrationConfigError::Io)?;
        Self::from_toml_str(&content).map_err(MigrationConfigError::Parse)
    }

    pub fn normalize(&mut self) {
        self.python_base_url = normalize_base_url(&self.python_base_url, "http");
        self.python_ws_base_url = normalize_base_url(&self.python_ws_base_url, "ws");
        self.routes = self
            .routes
            .iter()
            .map(|(key, value)| (key.trim().to_ascii_lowercase(), *value))
            .collect();
    }

    pub fn mode_for_capability(&self, capability: &str) -> MigrationMode {
        if !self.enabled {
            return MigrationMode::Native;
        }
        self.routes
            .get(&capability.to_ascii_lowercase())
            .copied()
            .unwrap_or(self.fallback)
    }

    pub fn mode_for_path(&self, path: &str) -> MigrationMode {
        if path == "/api/v1/admin/migration/reload" {
            return MigrationMode::Native;
        }
        if !self.enabled {
            return MigrationMode::Native;
        }
        capability_for_path(path)
            .map(|capability| self.mode_for_capability(capability))
            .unwrap_or(self.fallback)
    }

    pub fn fallback_should_proxy(&self) -> bool {
        self.enabled && self.fallback.should_proxy()
    }
}

#[derive(Debug)]
pub enum MigrationConfigError {
    Io(io::Error),
    Parse(toml::de::Error),
}

impl std::fmt::Display for MigrationConfigError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "failed to read migration config: {error}"),
            Self::Parse(error) => write!(formatter, "failed to parse migration config: {error}"),
        }
    }
}

impl std::error::Error for MigrationConfigError {}

pub struct MigrationRuntime {
    config: ArcSwap<MigrationConfig>,
    config_path: PathBuf,
    client: reqwest::Client,
}

impl std::fmt::Debug for MigrationRuntime {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MigrationRuntime")
            .field("config_path", &self.config_path)
            .field("config", &self.config())
            .finish_non_exhaustive()
    }
}

impl MigrationRuntime {
    pub fn from_env() -> Self {
        Self::from_config_path(default_config_path())
    }

    pub fn from_config_path(config_path: PathBuf) -> Self {
        let config = if config_path.exists() {
            MigrationConfig::from_path(&config_path).unwrap_or_default()
        } else {
            MigrationConfig::default()
        };
        Self::new(config_path, config)
    }

    pub fn from_config_for_tests(config: MigrationConfig) -> Self {
        Self::new(PathBuf::from("migration.toml"), config)
    }

    fn new(config_path: PathBuf, mut config: MigrationConfig) -> Self {
        config.normalize();
        let client = reqwest::Client::builder()
            .no_proxy()
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self {
            config: ArcSwap::from_pointee(config),
            config_path,
            client,
        }
    }

    pub fn config(&self) -> Arc<MigrationConfig> {
        self.config.load_full()
    }

    pub fn reload_from_disk(&self) -> Result<Arc<MigrationConfig>, MigrationConfigError> {
        let config = if self.config_path.exists() {
            MigrationConfig::from_path(&self.config_path)?
        } else {
            MigrationConfig::default()
        };
        let config = Arc::new(config);
        self.config.store(config.clone());
        Ok(config)
    }

    pub fn mode_for_path(&self, path: &str) -> MigrationMode {
        self.config().mode_for_path(path)
    }

    pub fn fallback_should_proxy(&self) -> bool {
        self.config().fallback_should_proxy()
    }
}

pub fn capability_for_path(path: &str) -> Option<&'static str> {
    match path {
        "/api/v1/ws" | "/api/v1/chat" => Some("chat"),
        "/api/v1/solve" => Some("solve"),
        "/api/v1/book/ws" => Some("book"),
        _ if path.starts_with("/api/v1/chat/") => Some("chat"),
        _ if path.starts_with("/api/v1/sessions") => Some("chat"),
        _ if path.starts_with("/api/v1/solve/") => Some("solve"),
        _ if path.starts_with("/api/v1/book/") => Some("book"),
        _ if path.starts_with("/api/v1/knowledge/")
            || path == "/api/v1/knowledge"
            || path.starts_with("/api/v1/courses")
            || path.starts_with("/api/v1/course") =>
        {
            Some("knowledge")
        }
        _ if path.starts_with("/api/v1/tutorbot") => Some("tutorbot"),
        _ if path.starts_with("/api/v1/notebook") => Some("notebook"),
        _ if path.starts_with("/api/v1/question") => Some("quiz"),
        _ if path.starts_with("/api/v1/plugins") || path.starts_with("/api/v1/skills") => {
            Some("tools")
        }
        _ if path.starts_with("/api/v1/vision") => Some("vision"),
        _ if path.starts_with("/api/v1/co_writer") => Some("co_writer"),
        _ => None,
    }
}

pub fn is_websocket_upgrade_request(headers: &HeaderMap) -> bool {
    let has_websocket_upgrade = headers
        .get(UPGRADE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.eq_ignore_ascii_case("websocket"));
    if !has_websocket_upgrade {
        return false;
    }
    headers.get_all(CONNECTION).iter().any(|value| {
        value.to_str().ok().is_some_and(|value| {
            value
                .split(',')
                .any(|token| token.trim().eq_ignore_ascii_case("upgrade"))
        })
    })
}

pub async fn proxy_to_python(runtime: Arc<MigrationRuntime>, request: Request) -> Response {
    match proxy_to_python_inner(runtime, request).await {
        Ok(response) => response,
        Err(error) => proxy_error(error),
    }
}

pub async fn proxy_fallback_or_404(runtime: Arc<MigrationRuntime>, request: Request) -> Response {
    if runtime.fallback_should_proxy() {
        proxy_to_python(runtime, request).await
    } else {
        StatusCode::NOT_FOUND.into_response()
    }
}

pub fn proxy_ws_to_python(
    runtime: Arc<MigrationRuntime>,
    path_and_query: String,
    headers: HeaderMap,
    ws: WebSocketUpgrade,
) -> Response {
    ws.on_upgrade(move |socket| async move {
        if let Err(error) = proxy_ws_inner(runtime, path_and_query, headers, socket).await {
            eprintln!("socartes migration websocket proxy failed: {error}");
        }
    })
    .into_response()
}

async fn proxy_to_python_inner(
    runtime: Arc<MigrationRuntime>,
    request: Request,
) -> Result<Response, ProxyError> {
    let config = runtime.config();
    let uri = upstream_http_uri(&config.python_base_url, request.uri())?;
    let (parts, body) = request.into_parts();
    let request_body = reqwest::Body::wrap_stream(body.into_data_stream());

    let mut upstream = runtime
        .client
        .request(parts.method, uri)
        .version(parts.version)
        .body(request_body);
    let forwarded_headers = strip_hop_by_hop_headers(&parts.headers, true);
    for (name, value) in &forwarded_headers {
        upstream = upstream.header(name, value);
    }

    let upstream = upstream.send().await.map_err(ProxyError::Upstream)?;
    let status = upstream.status();
    let mut headers = strip_hop_by_hop_headers(upstream.headers(), false);
    if is_sse(&headers) {
        headers
            .entry(HeaderName::from_static("x-accel-buffering"))
            .or_insert(HeaderValue::from_static("no"));
        headers
            .entry(CACHE_CONTROL)
            .or_insert(HeaderValue::from_static("no-cache"));
    }
    let mut response = Response::new(Body::from_stream(upstream.bytes_stream()));
    *response.status_mut() = status;
    *response.headers_mut() = headers;
    Ok(response)
}

async fn proxy_ws_inner(
    runtime: Arc<MigrationRuntime>,
    path_and_query: String,
    headers: HeaderMap,
    client_socket: WebSocket,
) -> Result<(), ProxyError> {
    let config = runtime.config();
    let url = upstream_ws_url(&config.python_ws_base_url, &path_and_query)?;
    let mut upstream_request = url
        .into_client_request()
        .map_err(|error| ProxyError::WebSocket(error.to_string()))?;
    let forwarded_headers = strip_hop_by_hop_headers(&headers, true);
    for (name, value) in &forwarded_headers {
        if is_websocket_handshake_header(name) {
            continue;
        }
        upstream_request
            .headers_mut()
            .append(name.clone(), value.clone());
    }

    let (upstream_socket, _) = connect_async(upstream_request)
        .await
        .map_err(|error| ProxyError::WebSocket(error.to_string()))?;
    let (mut client_sender, mut client_receiver) = client_socket.split();
    let (mut upstream_sender, mut upstream_receiver) = upstream_socket.split();

    let client_to_upstream = async {
        while let Some(message) = client_receiver.next().await {
            let Ok(message) = message else {
                break;
            };
            let Some(message) = axum_to_tungstenite(message) else {
                break;
            };
            if upstream_sender.send(message).await.is_err() {
                break;
            }
        }
    };

    let upstream_to_client = async {
        while let Some(message) = upstream_receiver.next().await {
            let Ok(message) = message else {
                break;
            };
            let Some(message) = tungstenite_to_axum(message) else {
                break;
            };
            if client_sender.send(message).await.is_err() {
                break;
            }
        }
    };

    tokio::select! {
        _ = client_to_upstream => {}
        _ = upstream_to_client => {}
    }

    Ok(())
}

fn upstream_http_uri(base_url: &str, uri: &Uri) -> Result<String, ProxyError> {
    let path_and_query = uri
        .path_and_query()
        .map(|value| value.as_str())
        .unwrap_or("/");
    Ok(format!(
        "{}{}",
        base_url.trim_end_matches('/'),
        path_and_query
    ))
}

fn upstream_ws_url(base_url: &str, path_and_query: &str) -> Result<String, ProxyError> {
    let path = if path_and_query.starts_with('/') {
        path_and_query.to_string()
    } else {
        format!("/{path_and_query}")
    };
    Ok(format!("{}{}", base_url.trim_end_matches('/'), path))
}

fn strip_hop_by_hop_headers(headers: &HeaderMap, strip_host: bool) -> HeaderMap {
    let mut connection_tokens = HashSet::new();
    for value in headers.get_all(CONNECTION) {
        if let Ok(value) = value.to_str() {
            connection_tokens.extend(
                value
                    .split(',')
                    .map(|token| token.trim().to_ascii_lowercase())
                    .filter(|token| !token.is_empty()),
            );
        }
    }

    let mut filtered = HeaderMap::new();
    for (name, value) in headers {
        if is_hop_by_hop_header(name, &connection_tokens) {
            continue;
        }
        if strip_host && *name == HOST {
            continue;
        }
        filtered.append(name.clone(), value.clone());
    }
    filtered
}

fn is_hop_by_hop_header(name: &HeaderName, connection_tokens: &HashSet<String>) -> bool {
    *name == CONNECTION
        || name.as_str() == "keep-alive"
        || *name == PROXY_AUTHENTICATE
        || *name == PROXY_AUTHORIZATION
        || *name == TE
        || *name == TRAILER
        || *name == TRANSFER_ENCODING
        || *name == UPGRADE
        || connection_tokens.contains(name.as_str())
}

fn is_websocket_handshake_header(name: &HeaderName) -> bool {
    matches!(
        name.as_str(),
        "connection"
            | "upgrade"
            | "host"
            | "sec-websocket-key"
            | "sec-websocket-version"
            | "sec-websocket-accept"
            | "sec-websocket-protocol"
            | "sec-websocket-extensions"
    )
}

fn is_sse(headers: &HeaderMap) -> bool {
    headers
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.starts_with("text/event-stream"))
}

fn normalize_base_url(value: &str, default_scheme: &str) -> String {
    let trimmed = value.trim().trim_end_matches('/');
    if trimmed.is_empty() {
        return match default_scheme {
            "ws" => default_python_ws_base_url(),
            _ => default_python_base_url(),
        };
    }
    if trimmed.contains("://") {
        trimmed.to_string()
    } else {
        format!("{default_scheme}://{trimmed}")
    }
}

fn default_python_base_url() -> String {
    "http://127.0.0.1:8001".to_string()
}

fn default_python_ws_base_url() -> String {
    "ws://127.0.0.1:8001".to_string()
}

fn default_fallback_mode() -> MigrationMode {
    MigrationMode::Native
}

fn default_config_path() -> PathBuf {
    if let Some(path) = env::var_os("SOCARTES_MIGRATION_CONFIG") {
        return PathBuf::from(path);
    }
    for candidate in ["migration.toml", "../migration.toml"] {
        let path = PathBuf::from(candidate);
        if path.exists() {
            return path;
        }
    }
    PathBuf::from("migration.toml")
}

fn axum_to_tungstenite(message: Message) -> Option<TungsteniteMessage> {
    match message {
        Message::Text(text) => Some(TungsteniteMessage::Text(text.to_string().into())),
        Message::Binary(bytes) => Some(TungsteniteMessage::Binary(bytes.to_vec().into())),
        Message::Ping(bytes) => Some(TungsteniteMessage::Ping(bytes.to_vec().into())),
        Message::Pong(bytes) => Some(TungsteniteMessage::Pong(bytes.to_vec().into())),
        Message::Close(_) => None,
    }
}

fn tungstenite_to_axum(message: TungsteniteMessage) -> Option<Message> {
    match message {
        TungsteniteMessage::Text(text) => Some(Message::Text(text.to_string().into())),
        TungsteniteMessage::Binary(bytes) => Some(Message::Binary(bytes.to_vec().into())),
        TungsteniteMessage::Ping(bytes) => Some(Message::Ping(bytes.to_vec().into())),
        TungsteniteMessage::Pong(bytes) => Some(Message::Pong(bytes.to_vec().into())),
        TungsteniteMessage::Close(_) => None,
        TungsteniteMessage::Frame(_) => None,
    }
}

#[derive(Debug)]
enum ProxyError {
    Upstream(reqwest::Error),
    WebSocket(String),
}

impl std::fmt::Display for ProxyError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Upstream(error) => write!(formatter, "upstream request failed: {error}"),
            Self::WebSocket(error) => write!(formatter, "websocket proxy failed: {error}"),
        }
    }
}

fn proxy_error(error: ProxyError) -> Response {
    (
        StatusCode::BAD_GATEWAY,
        [(CONTENT_TYPE, "application/json")],
        Body::from(
            json!({
                "detail": "Python upstream is unavailable",
                "error": error.to_string()
            })
            .to_string(),
        ),
    )
        .into_response()
}
