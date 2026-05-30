use std::{convert::Infallible, net::SocketAddr, sync::Arc, time::Duration};

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
use socartes_backend::migration::{
    MigrationConfig, MigrationMode, MigrationRuntime, is_websocket_upgrade_request,
    proxy_to_python, proxy_ws_to_python,
};
use tokio::net::TcpListener;
use tokio_tungstenite::tungstenite::Message as TungsteniteMessage;

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
