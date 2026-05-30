use std::{
    convert::Infallible,
    fs,
    net::SocketAddr,
    path::PathBuf,
    sync::{
        Arc, OnceLock,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use axum::{
    Router,
    body::{Body, Bytes},
    extract::{OriginalUri, ws::WebSocketUpgrade},
    http::{HeaderMap, Request, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{any, get},
};
use futures_util::{SinkExt, StreamExt, stream};
use http_body_util::BodyExt;
use socartes_backend::app_with_knowledge_root_and_auth;
use socartes_backend::migration::{
    MigrationConfig, MigrationMode, MigrationRuntime, is_shadow_native_ws_request,
    is_websocket_upgrade_request, proxy_to_python, proxy_ws_to_python, shadow_to_python,
    shadow_ws_to_python,
};
use tokio::{
    net::TcpListener,
    sync::{Mutex, oneshot},
};
use tokio_tungstenite::tungstenite::Message as TungsteniteMessage;

static MIGRATION_ENV_MUTEX: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();

struct MigrationEnvGuard {
    saved: Vec<(&'static str, Option<String>)>,
    root: PathBuf,
    _guard: tokio::sync::MutexGuard<'static, ()>,
}

impl MigrationEnvGuard {
    async fn with_config(config: &str, native_ws_base_url: &str) -> Self {
        let guard = MIGRATION_ENV_MUTEX
            .get_or_init(|| tokio::sync::Mutex::new(()))
            .lock()
            .await;
        let root = unique_temp_dir("socartes-migration-test");
        fs::create_dir_all(&root).expect("create migration test root");
        let config_path = root.join("migration.toml");
        fs::write(&config_path, config).expect("write migration config");
        let saved = [
            "SOCARTES_MIGRATION_CONFIG",
            "SOCARTES_NATIVE_WS_BASE_URL",
            "SOCARTES_SHADOW_NATIVE_WS_TOKEN",
        ]
        .into_iter()
        .map(|key| (key, std::env::var(key).ok()))
        .collect::<Vec<_>>();
        // SAFETY: this guard serializes mutation of migration-related env vars in this test binary.
        unsafe {
            std::env::set_var("SOCARTES_MIGRATION_CONFIG", &config_path);
            std::env::set_var("SOCARTES_NATIVE_WS_BASE_URL", native_ws_base_url);
            std::env::set_var(
                "SOCARTES_SHADOW_NATIVE_WS_TOKEN",
                "socartes-test-shadow-token",
            );
        }
        Self {
            saved,
            root,
            _guard: guard,
        }
    }

    fn data_root(&self) -> PathBuf {
        self.root.join("data")
    }
}

impl Drop for MigrationEnvGuard {
    fn drop(&mut self) {
        // SAFETY: MIGRATION_ENV_MUTEX is still held while this guard restores variables.
        unsafe {
            for (key, value) in &self.saved {
                match value {
                    Some(value) => std::env::set_var(key, value),
                    None => std::env::remove_var(key),
                }
            }
        }
        let _ = fs::remove_dir_all(&self.root);
    }
}

#[test]
fn migration_toml_maps_capabilities_to_modes() {
    let config = MigrationConfig::from_toml_str(
        r#"
enabled = true
python_base_url = "http://127.0.0.1:8001/"
python_ws_base_url = "ws://127.0.0.1:8001/"
fallback = "proxy"

[routes]
chat = "shadow"
book = "native"
knowledge = "proxy"
"#,
    )
    .expect("migration config should parse");

    assert!(config.enabled);
    assert_eq!(config.python_base_url, "http://127.0.0.1:8001");
    assert_eq!(config.python_ws_base_url, "ws://127.0.0.1:8001");
    assert_eq!(config.fallback, MigrationMode::Proxy);
    assert_eq!(config.mode_for_capability("chat"), MigrationMode::Shadow);
    assert_eq!(config.mode_for_path("/api/v1/ws"), MigrationMode::Shadow);
    assert_eq!(
        config.mode_for_path("/api/v1/book/ws"),
        MigrationMode::Native
    );
    assert_eq!(
        config.mode_for_path("/api/v1/courses/linear-algebra/files"),
        MigrationMode::Proxy
    );
    assert_eq!(
        config.mode_for_path("/api/v1/not-yet-mapped"),
        MigrationMode::Proxy
    );
    assert_eq!(
        config.mode_for_path("/api/v1/admin/migration/reload"),
        MigrationMode::Native
    );
}

#[test]
fn websocket_upgrade_requests_bypass_http_proxy_gate() {
    let headers = HeaderMap::from_iter([
        (header::CONNECTION, "keep-alive, Upgrade".parse().unwrap()),
        (header::UPGRADE, "websocket".parse().unwrap()),
    ]);

    assert!(is_websocket_upgrade_request(&headers));

    let plain_headers = HeaderMap::from_iter([(header::CONNECTION, "keep-alive".parse().unwrap())]);
    assert!(!is_websocket_upgrade_request(&plain_headers));
}

#[test]
fn shadow_native_ws_bypass_requires_runtime_token() {
    let mut headers = HeaderMap::new();
    headers.insert("x-socartes-migration-shadow-native", "1".parse().unwrap());
    assert!(!is_shadow_native_ws_request(
        &headers,
        "socartes-test-shadow-token"
    ));

    headers.insert(
        "x-socartes-migration-shadow-native",
        "wrong-token".parse().unwrap(),
    );
    assert!(!is_shadow_native_ws_request(
        &headers,
        "socartes-test-shadow-token"
    ));

    headers.insert(
        "x-socartes-migration-shadow-native",
        "socartes-test-shadow-token".parse().unwrap(),
    );
    assert!(is_shadow_native_ws_request(
        &headers,
        "socartes-test-shadow-token"
    ));
    assert!(!is_shadow_native_ws_request(&headers, ""));
}

#[tokio::test]
async fn proxy_preserves_sse_headers_and_streams_chunks() {
    let upstream = Router::new().route("/api/v1/stream", get(delayed_sse));
    let upstream_addr = spawn_app(upstream).await;

    let config = MigrationConfig {
        enabled: true,
        python_base_url: format!("http://{upstream_addr}"),
        python_ws_base_url: format!("ws://{upstream_addr}"),
        fallback: MigrationMode::Proxy,
        routes: Default::default(),
    };
    let runtime = Arc::new(MigrationRuntime::from_config_for_tests(config));
    let proxy = Router::new().fallback(any({
        let runtime = runtime.clone();
        move |request| {
            let runtime = runtime.clone();
            async move { proxy_to_python(runtime, request).await }
        }
    }));
    let proxy_addr = spawn_app(proxy).await;

    let started = std::time::Instant::now();
    let response = reqwest::Client::new()
        .get(format!("http://{proxy_addr}/api/v1/stream?topic=rag"))
        .send()
        .await
        .expect("proxy request should complete");
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get(header::CONTENT_TYPE).unwrap(),
        "text/event-stream"
    );
    assert_eq!(
        response
            .headers()
            .get("x-upstream-path")
            .and_then(|value| value.to_str().ok()),
        Some("/api/v1/stream?topic=rag")
    );

    let mut chunks = response.bytes_stream();
    let first = chunks
        .next()
        .await
        .expect("first SSE chunk should arrive")
        .expect("first SSE chunk should be ok");
    assert!(
        started.elapsed() < Duration::from_millis(200),
        "first chunk should be streamed before upstream finishes"
    );
    assert_eq!(&first[..], b"event: thinking\ndata: first\n\n");

    let second = chunks
        .next()
        .await
        .expect("second SSE chunk should arrive")
        .expect("second SSE chunk should be ok");
    assert_eq!(&second[..], b"event: content\ndata: second\n\n");
    assert!(chunks.next().await.is_none());
}

#[tokio::test]
async fn shadow_returns_python_stream_and_runs_native_copy_in_background() {
    let upstream = Router::new().route("/api/v1/stream", any(delayed_sse));
    let upstream_addr = spawn_app(upstream).await;

    let config = MigrationConfig {
        enabled: true,
        python_base_url: format!("http://{upstream_addr}"),
        python_ws_base_url: format!("ws://{upstream_addr}"),
        fallback: MigrationMode::Proxy,
        routes: Default::default(),
    };
    let runtime = Arc::new(MigrationRuntime::from_config_for_tests(config));
    let native_calls = Arc::new(AtomicUsize::new(0));
    let native_bodies = Arc::new(Mutex::new(Vec::<String>::new()));
    let proxy = Router::new().fallback(any({
        let runtime = runtime.clone();
        let native_calls = native_calls.clone();
        let native_bodies = native_bodies.clone();
        move |request| {
            let runtime = runtime.clone();
            let native_calls = native_calls.clone();
            let native_bodies = native_bodies.clone();
            async move {
                shadow_to_python(runtime, "chat", request, move |native_request| {
                    let native_calls = native_calls.clone();
                    let native_bodies = native_bodies.clone();
                    async move {
                        native_calls.fetch_add(1, Ordering::SeqCst);
                        let bytes = native_request
                            .into_body()
                            .collect()
                            .await
                            .expect("native request body should collect")
                            .to_bytes();
                        native_bodies
                            .lock()
                            .await
                            .push(String::from_utf8_lossy(&bytes).to_string());
                        (
                            [(header::CONTENT_TYPE, "text/event-stream")],
                            Body::from("event: content\ndata: native\n\n"),
                        )
                            .into_response()
                    }
                })
                .await
            }
        }
    }));
    let proxy_addr = spawn_app(proxy).await;

    let started = std::time::Instant::now();
    let response = reqwest::Client::new()
        .post(format!("http://{proxy_addr}/api/v1/stream?topic=rag"))
        .body("shadow payload")
        .send()
        .await
        .expect("shadow request should complete");
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get("x-socartes-migration-mode")
            .and_then(|value| value.to_str().ok()),
        Some("shadow")
    );

    let mut chunks = response.bytes_stream();
    let first = chunks
        .next()
        .await
        .expect("first shadow chunk should arrive")
        .expect("first shadow chunk should be ok");
    assert!(
        started.elapsed() < Duration::from_millis(200),
        "shadow must not buffer Python SSE before returning it"
    );
    assert_eq!(&first[..], b"event: thinking\ndata: first\n\n");
    let second = chunks
        .next()
        .await
        .expect("second shadow chunk should arrive")
        .expect("second shadow chunk should be ok");
    assert_eq!(&second[..], b"event: content\ndata: second\n\n");
    assert!(chunks.next().await.is_none());

    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if native_calls.load(Ordering::SeqCst) == 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("native copy should run in the background");
    assert_eq!(native_bodies.lock().await.as_slice(), ["shadow payload"]);
}

#[tokio::test]
async fn shadow_native_copy_preserves_request_extensions() {
    #[derive(Clone)]
    struct ShadowMarker(&'static str);

    let upstream = Router::new().route("/api/v1/stream", any(delayed_sse));
    let upstream_addr = spawn_app(upstream).await;
    let config = MigrationConfig {
        enabled: true,
        python_base_url: format!("http://{upstream_addr}"),
        python_ws_base_url: format!("ws://{upstream_addr}"),
        fallback: MigrationMode::Proxy,
        routes: Default::default(),
    };
    let runtime = Arc::new(MigrationRuntime::from_config_for_tests(config));
    let (marker_tx, marker_rx) = oneshot::channel();
    let request = Request::builder()
        .method("POST")
        .uri("/api/v1/stream")
        .extension(ShadowMarker("kept"))
        .body(Body::from("shadow payload"))
        .expect("shadow request should build");

    let response = shadow_to_python(runtime, "chat", request, move |native_request| async move {
        let marker = native_request
            .extensions()
            .get::<ShadowMarker>()
            .map(|marker| marker.0.to_string());
        let _ = marker_tx.send(marker);
        StatusCode::OK.into_response()
    })
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let _ = response
        .into_body()
        .collect()
        .await
        .expect("shadow Python stream should drain");

    assert_eq!(
        marker_rx
            .await
            .expect("native copy should report marker extension"),
        Some("kept".to_string())
    );
}

#[tokio::test]
async fn proxy_splices_websocket_messages_in_both_directions() {
    let upstream = Router::new().route("/api/v1/book/ws", get(echo_ws));
    let upstream_addr = spawn_app(upstream).await;

    let config = MigrationConfig {
        enabled: true,
        python_base_url: format!("http://{upstream_addr}"),
        python_ws_base_url: format!("ws://{upstream_addr}"),
        fallback: MigrationMode::Proxy,
        routes: Default::default(),
    };
    let runtime = Arc::new(MigrationRuntime::from_config_for_tests(config));
    let proxy = Router::new().route(
        "/api/v1/book/ws",
        get({
            let runtime = runtime.clone();
            move |OriginalUri(uri): OriginalUri, headers: HeaderMap, ws: WebSocketUpgrade| {
                let runtime = runtime.clone();
                async move {
                    proxy_ws_to_python(
                        runtime,
                        uri.path_and_query()
                            .map(|value| value.as_str().to_string())
                            .unwrap_or_else(|| uri.path().to_string()),
                        headers,
                        ws,
                    )
                }
            }
        }),
    );
    let proxy_addr = spawn_app(proxy).await;

    let (mut socket, _) =
        tokio_tungstenite::connect_async(format!("ws://{proxy_addr}/api/v1/book/ws?turn=1"))
            .await
            .expect("websocket should connect through proxy");
    socket
        .send(TungsteniteMessage::Text("hello book".into()))
        .await
        .expect("client message should send");
    let echoed = socket
        .next()
        .await
        .expect("echo should arrive")
        .expect("echo should be ok");
    assert_eq!(echoed, TungsteniteMessage::Text("python:hello book".into()));
}

#[tokio::test]
async fn shadow_ws_returns_python_frames_and_tees_client_frames_to_native_ws() {
    let upstream = Router::new().route("/api/v1/ws", get(echo_ws));
    let upstream_addr = spawn_app(upstream).await;

    let native_messages = Arc::new(Mutex::new(Vec::<String>::new()));
    let native_header_seen = Arc::new(Mutex::new(Vec::<bool>::new()));
    let native = Router::new().route(
        "/api/v1/ws",
        get({
            let native_messages = native_messages.clone();
            let native_header_seen = native_header_seen.clone();
            move |headers: HeaderMap, ws: WebSocketUpgrade| {
                let native_messages = native_messages.clone();
                let native_header_seen = native_header_seen.clone();
                async move {
                    native_header_seen
                        .lock()
                        .await
                        .push(is_shadow_native_ws_request(
                            &headers,
                            "socartes-test-shadow-token",
                        ));
                    observed_native_ws(ws, native_messages).await
                }
            }
        }),
    );
    let native_addr = spawn_app(native).await;

    let config = MigrationConfig {
        enabled: true,
        python_base_url: format!("http://{upstream_addr}"),
        python_ws_base_url: format!("ws://{upstream_addr}"),
        fallback: MigrationMode::Proxy,
        routes: Default::default(),
    };
    let runtime = Arc::new(MigrationRuntime::from_config_for_tests(config));
    let proxy = Router::new().route(
        "/api/v1/ws",
        get({
            let runtime = runtime.clone();
            move |OriginalUri(uri): OriginalUri, headers: HeaderMap, ws: WebSocketUpgrade| {
                let runtime = runtime.clone();
                async move {
                    shadow_ws_to_python(
                        runtime,
                        "chat",
                        uri.path_and_query()
                            .map(|value| value.as_str().to_string())
                            .unwrap_or_else(|| uri.path().to_string()),
                        headers,
                        ws,
                        format!("ws://{native_addr}"),
                    )
                }
            }
        }),
    );
    let proxy_addr = spawn_app(proxy).await;

    let (mut socket, _) =
        tokio_tungstenite::connect_async(format!("ws://{proxy_addr}/api/v1/ws?turn=1"))
            .await
            .expect("websocket should connect through shadow proxy");
    socket
        .send(TungsteniteMessage::Text("hello shadow".into()))
        .await
        .expect("client message should send");
    let echoed = socket
        .next()
        .await
        .expect("python echo should arrive")
        .expect("python echo should be ok");
    assert_eq!(
        echoed,
        TungsteniteMessage::Text("python:hello shadow".into())
    );
    assert!(
        tokio::time::timeout(Duration::from_millis(150), socket.next())
            .await
            .is_err(),
        "native shadow frames must not be forwarded to the client"
    );

    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if native_messages.lock().await.as_slice() == ["hello shadow"] {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("native websocket should receive the client frame");
    assert_eq!(native_header_seen.lock().await.as_slice(), [true]);
}

#[tokio::test]
async fn book_ws_shadow_mode_returns_python_frames_and_tees_to_native_ws() {
    let upstream = Router::new().route("/api/v1/book/ws", get(echo_ws));
    let upstream_addr = spawn_app(upstream).await;

    let native_messages = Arc::new(Mutex::new(Vec::<String>::new()));
    let native_header_seen = Arc::new(Mutex::new(Vec::<bool>::new()));
    let native = Router::new().route(
        "/api/v1/book/ws",
        get({
            let native_messages = native_messages.clone();
            let native_header_seen = native_header_seen.clone();
            move |headers: HeaderMap, ws: WebSocketUpgrade| {
                let native_messages = native_messages.clone();
                let native_header_seen = native_header_seen.clone();
                async move {
                    native_header_seen
                        .lock()
                        .await
                        .push(is_shadow_native_ws_request(
                            &headers,
                            "socartes-test-shadow-token",
                        ));
                    observed_native_ws(ws, native_messages).await
                }
            }
        }),
    );
    let native_addr = spawn_app(native).await;
    let config = format!(
        r#"
enabled = true
python_base_url = "http://{upstream_addr}"
python_ws_base_url = "ws://{upstream_addr}"
fallback = "proxy"

[routes]
book = "shadow"
"#
    );
    let env_guard = MigrationEnvGuard::with_config(&config, &format!("ws://{native_addr}")).await;
    let app =
        app_with_knowledge_root_and_auth(env_guard.data_root().join("knowledge_bases"), false);
    let app_addr = spawn_app(app).await;

    let (mut socket, _) =
        tokio_tungstenite::connect_async(format!("ws://{app_addr}/api/v1/book/ws?turn=1"))
            .await
            .expect("book websocket should connect through app router");
    socket
        .send(TungsteniteMessage::Text("hello book shadow".into()))
        .await
        .expect("client message should send");
    let echoed = socket
        .next()
        .await
        .expect("python echo should arrive")
        .expect("python echo should be ok");
    assert_eq!(
        echoed,
        TungsteniteMessage::Text("python:hello book shadow".into())
    );
    assert!(
        tokio::time::timeout(Duration::from_millis(150), socket.next())
            .await
            .is_err(),
        "native shadow frames must not be forwarded to the client"
    );

    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if native_messages.lock().await.as_slice() == ["hello book shadow"] {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("native websocket should receive the client frame");
    assert_eq!(native_header_seen.lock().await.as_slice(), [true]);
}

#[tokio::test]
async fn tutorbot_ws_shadow_mode_returns_python_frames_and_tees_to_native_ws() {
    let upstream = Router::new().route("/api/v1/tutorbot/test-bot/ws", get(echo_ws));
    let upstream_addr = spawn_app(upstream).await;

    let native_messages = Arc::new(Mutex::new(Vec::<String>::new()));
    let native_header_seen = Arc::new(Mutex::new(Vec::<bool>::new()));
    let native = Router::new().route(
        "/api/v1/tutorbot/test-bot/ws",
        get({
            let native_messages = native_messages.clone();
            let native_header_seen = native_header_seen.clone();
            move |headers: HeaderMap, ws: WebSocketUpgrade| {
                let native_messages = native_messages.clone();
                let native_header_seen = native_header_seen.clone();
                async move {
                    native_header_seen
                        .lock()
                        .await
                        .push(is_shadow_native_ws_request(
                            &headers,
                            "socartes-test-shadow-token",
                        ));
                    observed_native_ws(ws, native_messages).await
                }
            }
        }),
    );
    let native_addr = spawn_app(native).await;
    let config = format!(
        r#"
enabled = true
python_base_url = "http://{upstream_addr}"
python_ws_base_url = "ws://{upstream_addr}"
fallback = "proxy"

[routes]
tutorbot = "shadow"
"#
    );
    let env_guard = MigrationEnvGuard::with_config(&config, &format!("ws://{native_addr}")).await;
    let app =
        app_with_knowledge_root_and_auth(env_guard.data_root().join("knowledge_bases"), false);
    let app_addr = spawn_app(app).await;

    let (mut socket, _) = tokio_tungstenite::connect_async(format!(
        "ws://{app_addr}/api/v1/tutorbot/test-bot/ws?turn=1"
    ))
    .await
    .expect("tutorbot websocket should connect through app router");
    socket
        .send(TungsteniteMessage::Text("hello tutorbot shadow".into()))
        .await
        .expect("client message should send");
    let echoed = socket
        .next()
        .await
        .expect("python echo should arrive")
        .expect("python echo should be ok");
    assert_eq!(
        echoed,
        TungsteniteMessage::Text("python:hello tutorbot shadow".into())
    );
    assert!(
        tokio::time::timeout(Duration::from_millis(150), socket.next())
            .await
            .is_err(),
        "native shadow frames must not be forwarded to the client"
    );

    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if native_messages.lock().await.as_slice() == ["hello tutorbot shadow"] {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("native websocket should receive the client frame");
    assert_eq!(native_header_seen.lock().await.as_slice(), [true]);
}

#[tokio::test]
async fn solve_ws_shadow_mode_returns_python_frames_and_tees_to_native_ws() {
    let upstream = Router::new().route("/api/v1/solve", get(echo_ws));
    let upstream_addr = spawn_app(upstream).await;

    let native_messages = Arc::new(Mutex::new(Vec::<String>::new()));
    let native_header_seen = Arc::new(Mutex::new(Vec::<bool>::new()));
    let native = Router::new().route(
        "/api/v1/solve",
        get({
            let native_messages = native_messages.clone();
            let native_header_seen = native_header_seen.clone();
            move |headers: HeaderMap, ws: WebSocketUpgrade| {
                let native_messages = native_messages.clone();
                let native_header_seen = native_header_seen.clone();
                async move {
                    native_header_seen
                        .lock()
                        .await
                        .push(is_shadow_native_ws_request(
                            &headers,
                            "socartes-test-shadow-token",
                        ));
                    observed_native_ws(ws, native_messages).await
                }
            }
        }),
    );
    let native_addr = spawn_app(native).await;
    let config = format!(
        r#"
enabled = true
python_base_url = "http://{upstream_addr}"
python_ws_base_url = "ws://{upstream_addr}"
fallback = "proxy"

[routes]
solve = "shadow"
"#
    );
    let env_guard = MigrationEnvGuard::with_config(&config, &format!("ws://{native_addr}")).await;
    let app =
        app_with_knowledge_root_and_auth(env_guard.data_root().join("knowledge_bases"), false);
    let app_addr = spawn_app(app).await;

    let (mut socket, _) =
        tokio_tungstenite::connect_async(format!("ws://{app_addr}/api/v1/solve?turn=1"))
            .await
            .expect("solve websocket should connect through app router");
    socket
        .send(TungsteniteMessage::Text("hello solve shadow".into()))
        .await
        .expect("client message should send");
    let echoed = socket
        .next()
        .await
        .expect("python echo should arrive")
        .expect("python echo should be ok");
    assert_eq!(
        echoed,
        TungsteniteMessage::Text("python:hello solve shadow".into())
    );
    assert!(
        tokio::time::timeout(Duration::from_millis(150), socket.next())
            .await
            .is_err(),
        "native shadow frames must not be forwarded to the client"
    );

    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if native_messages.lock().await.as_slice() == ["hello solve shadow"] {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("native websocket should receive the client frame");
    assert_eq!(native_header_seen.lock().await.as_slice(), [true]);
}

#[tokio::test]
async fn question_generate_ws_shadow_mode_returns_python_frames_and_tees_to_native_ws() {
    assert_quiz_ws_shadow_route(
        "/api/v1/question/generate",
        "hello question generate shadow",
    )
    .await;
}

#[tokio::test]
async fn question_mimic_ws_shadow_mode_returns_python_frames_and_tees_to_native_ws() {
    assert_quiz_ws_shadow_route("/api/v1/question/mimic", "hello question mimic shadow").await;
}

#[tokio::test]
async fn vision_solve_ws_shadow_mode_returns_python_frames_and_tees_to_native_ws() {
    assert_app_ws_shadow_route(
        "vision",
        "/api/v1/vision/solve",
        "hello vision solve shadow",
    )
    .await;
}

#[tokio::test]
async fn knowledge_progress_ws_shadow_mode_returns_python_frames_and_tees_to_native_ws() {
    assert_app_ws_shadow_route(
        "knowledge",
        "/api/v1/knowledge/test-kb/progress/ws",
        "hello knowledge progress shadow",
    )
    .await;
}

#[tokio::test]
async fn legacy_chat_ws_shadow_mode_returns_python_frames_and_tees_to_native_ws() {
    assert_app_ws_shadow_route("chat", "/api/v1/chat", "hello legacy chat shadow").await;
}

#[tokio::test]
async fn disabled_migration_fallback_returns_404_without_python() {
    let runtime = Arc::new(MigrationRuntime::from_config_for_tests(
        MigrationConfig::default(),
    ));
    let response = socartes_backend::migration::proxy_fallback_or_404(
        runtime,
        Request::builder()
            .uri("/api/v1/unmapped")
            .body(Body::empty())
            .unwrap(),
    )
    .await;

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body = response.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(&body[..], b"");
}

async fn assert_quiz_ws_shadow_route(path: &'static str, payload: &'static str) {
    assert_app_ws_shadow_route("quiz", path, payload).await;
}

async fn assert_app_ws_shadow_route(
    capability: &'static str,
    path: &'static str,
    payload: &'static str,
) {
    let upstream = Router::new().route(path, get(echo_ws));
    let upstream_addr = spawn_app(upstream).await;

    let native_messages = Arc::new(Mutex::new(Vec::<String>::new()));
    let native_header_seen = Arc::new(Mutex::new(Vec::<bool>::new()));
    let native = Router::new().route(
        path,
        get({
            let native_messages = native_messages.clone();
            let native_header_seen = native_header_seen.clone();
            move |headers: HeaderMap, ws: WebSocketUpgrade| {
                let native_messages = native_messages.clone();
                let native_header_seen = native_header_seen.clone();
                async move {
                    native_header_seen
                        .lock()
                        .await
                        .push(is_shadow_native_ws_request(
                            &headers,
                            "socartes-test-shadow-token",
                        ));
                    observed_native_ws(ws, native_messages).await
                }
            }
        }),
    );
    let native_addr = spawn_app(native).await;
    let config = format!(
        r#"
enabled = true
python_base_url = "http://{upstream_addr}"
python_ws_base_url = "ws://{upstream_addr}"
fallback = "proxy"

[routes]
{capability} = "shadow"
"#
    );
    let env_guard = MigrationEnvGuard::with_config(&config, &format!("ws://{native_addr}")).await;
    let app =
        app_with_knowledge_root_and_auth(env_guard.data_root().join("knowledge_bases"), false);
    let app_addr = spawn_app(app).await;

    let (mut socket, _) = tokio_tungstenite::connect_async(format!("ws://{app_addr}{path}?turn=1"))
        .await
        .expect("quiz websocket should connect through app router");
    socket
        .send(TungsteniteMessage::Text(payload.into()))
        .await
        .expect("client message should send");
    let echoed = socket
        .next()
        .await
        .expect("python echo should arrive")
        .expect("python echo should be ok");
    assert_eq!(
        echoed,
        TungsteniteMessage::Text(format!("python:{payload}").into())
    );
    assert!(
        tokio::time::timeout(Duration::from_millis(150), socket.next())
            .await
            .is_err(),
        "native shadow frames must not be forwarded to the client"
    );

    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if native_messages.lock().await.as_slice() == [payload] {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("native websocket should receive the client frame");
    assert_eq!(native_header_seen.lock().await.as_slice(), [true]);
}

async fn observed_native_ws(ws: WebSocketUpgrade, messages: Arc<Mutex<Vec<String>>>) -> Response {
    ws.on_upgrade(move |mut socket| async move {
        while let Some(Ok(message)) = socket.recv().await {
            let axum::extract::ws::Message::Text(text) = message else {
                continue;
            };
            messages.lock().await.push(text.to_string());
            let _ = socket
                .send(axum::extract::ws::Message::Text(
                    format!("rust:{text}").into(),
                ))
                .await;
        }
    })
    .into_response()
}

async fn echo_ws(ws: WebSocketUpgrade) -> Response {
    ws.on_upgrade(|mut socket| async move {
        while let Some(Ok(message)) = socket.recv().await {
            let axum::extract::ws::Message::Text(text) = message else {
                continue;
            };
            let _ = socket
                .send(axum::extract::ws::Message::Text(
                    format!("python:{text}").into(),
                ))
                .await;
        }
    })
    .into_response()
}

async fn delayed_sse(request: Request<Body>) -> Response {
    let path = request
        .uri()
        .path_and_query()
        .map(|value| value.as_str().to_string())
        .unwrap_or_default();
    let stream = stream::unfold(0, |index| async move {
        match index {
            0 => Some((
                Ok::<_, Infallible>(Bytes::from_static(b"event: thinking\ndata: first\n\n")),
                1,
            )),
            1 => {
                tokio::time::sleep(Duration::from_millis(350)).await;
                Some((
                    Ok::<_, Infallible>(Bytes::from_static(b"event: content\ndata: second\n\n")),
                    2,
                ))
            }
            _ => None,
        }
    });
    (
        [
            (header::CONTENT_TYPE, "text/event-stream"),
            (header::CACHE_CONTROL, "no-cache"),
            (
                header::HeaderName::from_static("x-upstream-path"),
                path.as_str(),
            ),
        ],
        Body::from_stream(stream),
    )
        .into_response()
}

async fn spawn_app(app: Router) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    addr
}

fn unique_temp_dir(prefix: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    std::env::temp_dir().join(format!("{prefix}-{}-{nanos}", std::process::id()))
}
