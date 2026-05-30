use assert_cmd::Command;
use axum::{
    Json, Router,
    extract::State,
    response::IntoResponse,
    routing::{get, post},
};
use serde_json::{Value, json};
use std::{
    fs,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};
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

async fn capture_new_session_stream(
    State(captured): State<Arc<Mutex<Vec<Value>>>>,
    Json(payload): Json<Value>,
) -> impl IntoResponse {
    captured.lock().await.push(payload);
    (
        [("content-type", "text/event-stream")],
        "event: stream\ndata: {\"type\":\"session\",\"session_id\":\"session-new\",\"turn_id\":\"turn-new\",\"metadata\":{\"session_id\":\"session-new\",\"turn_id\":\"turn-new\"}}\n\nevent: stream\ndata: {\"type\":\"content\",\"stage\":\"executor\",\"content\":\"first answer\"}\n\nevent: result\ndata: {\"success\":true,\"data\":{\"session_id\":\"session-new\",\"turn_id\":\"turn-new\",\"result\":{\"content\":\"first answer\"}},\"elapsed_ms\":1}\n\n",
    )
}

async fn capture_tool_result_stream(
    State(captured): State<Arc<Mutex<Vec<Value>>>>,
    Json(payload): Json<Value>,
) -> impl IntoResponse {
    captured.lock().await.push(payload);
    (
        [("content-type", "text/event-stream")],
        "event: stream\ndata: {\"type\":\"session\",\"session_id\":\"session-tools\",\"turn_id\":\"turn-tools\"}\n\nevent: stream\ndata: {\"type\":\"tool_result\",\"metadata\":{\"tool\":\"rag\"},\"content\":\"line 1\\nline 2\\nline 3\\nline 4\\nline 5\\nline 6\\nline 7\\nline 8\\nline 9\\nline 10\\nline 11\\nline 12\"}\n\nevent: result\ndata: {\"success\":true,\"data\":{\"session_id\":\"session-tools\",\"turn_id\":\"turn-tools\",\"result\":{\"content\":\"done\"}},\"elapsed_ms\":1}\n\n",
    )
}

fn unique_temp_dir(label: &str) -> std::path::PathBuf {
    let suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("socartes-cli-{label}-{suffix}"))
}

async fn existing_cli_session() -> Json<Value> {
    Json(json!({
        "id": "session-existing",
        "title": "Existing CLI session",
        "preferences": {
            "capability": "deep_solve",
            "tools": ["rag", "web_search"],
            "knowledge_bases": ["course-ai"],
            "language": "zh",
            "notebook_references": [
                {"notebook_id": "nb1", "record_ids": ["r1"]}
            ],
            "history_references": ["history-1"]
        },
        "messages": []
    }))
}

async fn capture_regenerate_stream(
    State(captured): State<Arc<Mutex<Vec<Value>>>>,
    Json(payload): Json<Value>,
) -> impl IntoResponse {
    captured.lock().await.push(payload);
    (
        [("content-type", "text/event-stream")],
        "event: stream\ndata: {\"type\":\"content\",\"stage\":\"executor\",\"content\":\"regenerated answer\"}\n\nevent: result\ndata: {\"success\":true,\"data\":{\"turn_id\":\"turn-regenerated\",\"result\":{\"content\":\"regenerated answer\"}},\"elapsed_ms\":1}\n\n",
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
    ];

    for (group, subcommands) in groups {
        let stdout = stdout_for(&[group, "--help"]);
        assert_contains_all(&stdout, &[group]);
        assert_contains_all(&stdout, &subcommands);
    }

    let init_help = stdout_for(&["init", "--help"]);
    assert_contains_all(&init_help, &["--cli", "--home", "--yes", "wizard"]);
}

#[test]
fn start_help_keeps_python_home_option() {
    let stdout = stdout_for(&["start", "--help"]);
    assert_contains_all(&stdout, &["--home", "--host", "--port"]);
}

#[test]
fn init_top_level_wizard_creates_runtime_layout_and_settings() {
    let home = unique_temp_dir("init");
    let output = socartes_cmd()
        .args(["init", "--yes", "--cli", "--home"])
        .arg(&home)
        .output()
        .expect("socartes init should execute");

    assert!(
        output.status.success(),
        "socartes init failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert_contains_all(&stdout, &["\"initialized\": true", "\"cli_only\": true"]);

    for relative in [
        "data/knowledge",
        "data/sessions",
        "data/user/workspace/book",
        "data/user/workspace/notebook",
        "data/user/workspace/chat/attachments",
        "data/memory",
        "data/settings",
    ] {
        assert!(
            home.join(relative).is_dir(),
            "expected init to create {}",
            home.join(relative).display()
        );
    }

    let catalog_text = fs::read_to_string(home.join("data/settings/catalog.json"))
        .expect("catalog.json should be written");
    let catalog: Value = serde_json::from_str(&catalog_text).unwrap();
    assert_eq!(
        catalog["services"]["llm"]["active_profile_id"],
        "socartes-rust"
    );
    assert!(
        home.join("data/settings/ui.json").is_file(),
        "ui.json should be written"
    );

    let _ = fs::remove_dir_all(home);
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_first_message_updates_session_and_enables_retry_like_python() {
    let captured_turns = Arc::new(Mutex::new(Vec::new()));
    let captured_regenerates = Arc::new(Mutex::new(Vec::new()));
    let app = Router::new()
        .route(
            "/api/v1/plugins/capabilities/chat/execute-stream",
            post(capture_new_session_stream),
        )
        .with_state(captured_turns.clone())
        .route(
            "/api/v1/sessions/session-new/regenerate-stream",
            post(capture_regenerate_stream),
        )
        .with_state(captured_regenerates.clone());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let output = socartes_cmd()
        .env("SOCARTES_API_URL", format!("http://{address}"))
        .args(["chat"])
        .write_stdin("hello\n/session\n/retry\n/quit\n")
        .output()
        .expect("socartes chat should execute");
    server.abort();

    assert!(
        output.status.success(),
        "socartes chat failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert_contains_all(
        &stdout,
        &["first answer", "session-new", "regenerated answer"],
    );
    assert!(
        !stdout.contains("No active session yet"),
        "retry should use the session created by the first chat turn:\n{stdout}"
    );
    let turns = captured_turns.lock().await;
    assert_eq!(turns.len(), 1);
    assert!(turns[0].get("session_id").is_none());
    let regenerates = captured_regenerates.lock().await;
    assert_eq!(regenerates.len(), 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_regenerate_and_retry_match_python_repl_commands() {
    let captured = Arc::new(Mutex::new(Vec::new()));
    let app = Router::new()
        .route(
            "/api/v1/sessions/session-existing",
            get(existing_cli_session),
        )
        .route(
            "/api/v1/sessions/session-existing/regenerate-stream",
            post(capture_regenerate_stream),
        )
        .with_state(captured.clone());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let output = socartes_cmd()
        .env("SOCARTES_API_URL", format!("http://{address}"))
        .args(["chat", "--session", "session-existing"])
        .write_stdin("/regenerate\n/retry\n/quit\n")
        .output()
        .expect("socartes chat should execute");
    server.abort();

    assert!(
        output.status.success(),
        "socartes chat failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert_contains_all(&stdout, &["regenerated answer", "turn-regenerated"]);
    assert!(
        !String::from_utf8_lossy(&output.stderr).contains("Unknown chat command"),
        "/regenerate and /retry must be first-class Python-compatible chat commands"
    );
    let calls = captured.lock().await;
    assert_eq!(
        calls.len(),
        2,
        "both /regenerate and /retry should call the backend"
    );
    for payload in calls.iter() {
        assert_eq!(payload["overrides"]["capability"], "deep_solve");
        assert_eq!(payload["overrides"]["tools"], json!(["rag", "web_search"]));
        assert_eq!(
            payload["overrides"]["knowledge_bases"],
            json!(["course-ai"])
        );
        assert_eq!(payload["overrides"]["language"], "zh");
        assert_eq!(
            payload["overrides"]["history_references"],
            json!(["history-1"])
        );
        assert_eq!(
            payload["overrides"]["notebook_references"],
            json!([{"notebook_id":"nb1","record_ids":["r1"]}])
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_session_loads_python_repl_preferences_and_refs_command() {
    let app = Router::new().route(
        "/api/v1/sessions/session-existing",
        get(existing_cli_session),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let output = socartes_cmd()
        .env("SOCARTES_API_URL", format!("http://{address}"))
        .args(["chat", "--session", "session-existing"])
        .write_stdin("/session\n/refs\n/quit\n")
        .output()
        .expect("socartes chat should execute");
    server.abort();

    assert!(
        output.status.success(),
        "socartes chat failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert_contains_all(
        &stdout,
        &[
            "session-existing",
            "deep_solve",
            "rag",
            "web_search",
            "course-ai",
            "zh",
            "history-1",
            "nb1",
            "r1",
        ],
    );
    assert!(
        !String::from_utf8_lossy(&output.stderr).contains("Unknown chat command"),
        "/refs must be a first-class Python-compatible chat command"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_show_expands_recent_tool_result_like_python_repl() {
    let captured_turns = Arc::new(Mutex::new(Vec::new()));
    let app = Router::new()
        .route(
            "/api/v1/plugins/capabilities/chat/execute-stream",
            post(capture_tool_result_stream),
        )
        .with_state(captured_turns.clone());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let output = socartes_cmd()
        .env("SOCARTES_API_URL", format!("http://{address}"))
        .args(["chat"])
        .write_stdin("use rag\n/show last\n/quit\n")
        .output()
        .expect("socartes chat should execute");
    server.abort();

    assert!(
        output.status.success(),
        "socartes chat failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert_contains_all(&stdout, &["#1 rag", "+2 more lines", "line 12"]);
    assert!(
        !String::from_utf8_lossy(&output.stderr).contains("Unknown chat command"),
        "/show must be a first-class Python-compatible chat command"
    );
}
