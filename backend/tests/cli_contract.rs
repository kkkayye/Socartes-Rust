use assert_cmd::Command;
use axum::{Json, Router, extract::State, response::IntoResponse, routing::post};
use serde_json::{Value, json};
use std::sync::Arc;
use tokio::{net::TcpListener, sync::Mutex};

fn socartes_cmd() -> Command {
    Command::cargo_bin("socartes").expect("socartes binary should build")
}

fn stdout_for(args: &[&str]) -> String {
    let output = socartes_cmd()
        .args(args)
        .output()
        .expect("socartes command should run");
    assert!(
        output.status.success(),
        "socartes {args:?} failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("stdout should be utf-8")
}

fn assert_contains_all(stdout: &str, needles: &[&str]) {
    for needle in needles {
        assert!(
            stdout.contains(needle),
            "expected stdout to contain `{needle}`:\n{stdout}"
        );
    }
}

async fn capture_capability_stream(
    State(captured): State<Arc<Mutex<Option<Value>>>>,
    Json(payload): Json<Value>,
) -> impl IntoResponse {
    *captured.lock().await = Some(payload);
    (
        [("content-type", "text/event-stream")],
        "event: stream\ndata: {\"type\":\"content\",\"stage\":\"executor\",\"content\":\"hello back\"}\n\nevent: result\ndata: {\"success\":true,\"data\":{\"turn_id\":\"turn-1\",\"result\":{\"content\":\"hello back\"}},\"elapsed_ms\":1}\n\n",
    )
}

#[test]
fn top_level_help_exposes_python_cli_command_surface() {
    let stdout = stdout_for(&["--help"]);
    assert_contains_all(
        &stdout,
        &[
            "Socartes CLI",
            "run",
            "start",
            "serve",
            "chat",
            "book",
            "bot",
            "kb",
            "notebook",
            "memory",
            "plugin",
            "config",
            "session",
            "provider",
            "init",
        ],
    );
}

#[test]
fn run_help_matches_agent_first_python_options() {
    let stdout = stdout_for(&["run", "--help"]);
    assert_contains_all(
        &stdout,
        &[
            "--session",
            "--tool",
            "--kb",
            "--notebook-ref",
            "--history-ref",
            "--language",
            "--config",
            "--config-json",
            "--format",
        ],
    );
}

#[test]
fn grouped_help_exposes_python_subcommands() {
    let groups = [
        ("book", vec!["list", "health", "refresh-fingerprints"]),
        ("bot", vec!["list", "start", "stop", "create"]),
        (
            "kb",
            vec![
                "list",
                "info",
                "set-default",
                "create",
                "add",
                "delete",
                "search",
            ],
        ),
        (
            "notebook",
            vec![
                "list",
                "create",
                "show",
                "remove-record",
                "add-md",
                "replace-md",
            ],
        ),
        ("memory", vec!["show", "clear"]),
        ("plugin", vec!["list", "info"]),
        ("config", vec!["show"]),
        ("provider", vec!["login"]),
        ("session", vec!["list", "show", "open", "delete", "rename"]),
        ("init", vec!["wizard"]),
    ];

    for (group, subcommands) in groups {
        let stdout = stdout_for(&[group, "--help"]);
        assert_contains_all(&stdout, &[group]);
        assert_contains_all(&stdout, &subcommands);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_json_posts_capability_stream_payload_and_prints_sse_payloads() {
    let captured = Arc::new(Mutex::new(None));
    let app = Router::new()
        .route(
            "/api/v1/plugins/capabilities/chat/execute-stream",
            post(capture_capability_stream),
        )
        .with_state(captured.clone());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let output = socartes_cmd()
        .env("SOCARTES_API_URL", format!("http://{address}"))
        .args([
            "run",
            "chat",
            "hello",
            "--tool",
            "rag",
            "--kb",
            "course-ai",
            "--notebook-ref",
            "nb1:r1,r2",
            "--history-ref",
            "session-old",
            "--language",
            "zh",
            "--config",
            "temperature=0",
            "--config-json",
            r#"{"render_mode":"auto"}"#,
            "--format",
            "json",
        ])
        .output()
        .expect("socartes run should execute");
    server.abort();

    assert!(
        output.status.success(),
        "socartes run failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert_contains_all(&stdout, &["hello back", "\"turn_id\":\"turn-1\""]);

    let payload = captured
        .lock()
        .await
        .clone()
        .expect("server should capture request payload");
    assert_eq!(payload["content"], "hello");
    assert_eq!(payload["tools"], json!(["rag"]));
    assert_eq!(payload["knowledge_bases"], json!(["course-ai"]));
    assert_eq!(
        payload["notebook_references"],
        json!([{"notebook_id":"nb1","record_ids":["r1","r2"]}])
    );
    assert_eq!(payload["history_references"], json!(["session-old"]));
    assert_eq!(payload["language"], "zh");
    assert_eq!(payload["config"]["temperature"], 0);
    assert_eq!(payload["config"]["render_mode"], "auto");
}
