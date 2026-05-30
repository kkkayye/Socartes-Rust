use assert_cmd::Command;
use axum::{
    Json, Router,
    body::to_bytes,
    extract::State,
    http::HeaderMap,
    response::IntoResponse,
    routing::{get, post},
};
use serde_json::{Value, json};
use std::{
    fs,
    os::unix::fs::PermissionsExt,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::{net::TcpListener, sync::Mutex};

fn socartes_cmd() -> Command {
    Command::cargo_bin("socartes").expect("socartes binary should build")
}

fn socartes_cli_cmd() -> Command {
    Command::cargo_bin("socartes-cli").expect("socartes-cli binary alias should build")
}

fn socartes_cli_underscore_cmd() -> Command {
    Command::cargo_bin("socartes_cli").expect("socartes_cli binary alias should build")
}

fn deeptutor_cmd() -> Command {
    Command::cargo_bin("deeptutor").expect("deeptutor compatibility binary should build")
}

fn deeptutor_cli_cmd() -> Command {
    Command::cargo_bin("deeptutor-cli").expect("deeptutor-cli compatibility binary should build")
}

fn deeptutor_cli_underscore_cmd() -> Command {
    Command::cargo_bin("deeptutor_cli").expect("deeptutor_cli compatibility binary should build")
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

async fn capture_github_copilot_validation(
    State(captured): State<Arc<Mutex<Option<Value>>>>,
    headers: HeaderMap,
    Json(payload): Json<Value>,
) -> impl IntoResponse {
    let auth = headers
        .get("authorization")
        .and_then(|value| value.to_str().ok())
        .unwrap_or("");
    *captured.lock().await = Some(json!({
        "authorization": auth,
        "payload": payload
    }));
    Json(json!({
        "id": "copilot-validation",
        "choices": [{
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop"
        }]
    }))
}

async fn capture_session_list_uri(
    State(captured): State<Arc<Mutex<Option<String>>>>,
    request: axum::extract::Request,
) -> impl IntoResponse {
    *captured.lock().await = Some(
        request
            .uri()
            .path_and_query()
            .map(|value| value.as_str().to_string())
            .unwrap_or_else(|| request.uri().path().to_string()),
    );
    let _ = to_bytes(request.into_body(), usize::MAX).await;
    Json(json!({
        "sessions": [
            {"id": "s1", "title": "First"},
            {"id": "s2", "title": "Second"}
        ]
    }))
}

async fn capture_json_request(
    State(captured): State<Arc<Mutex<Vec<Value>>>>,
    request: axum::extract::Request,
) -> impl IntoResponse {
    let method = request.method().as_str().to_string();
    let uri = request
        .uri()
        .path_and_query()
        .map(|value| value.as_str().to_string())
        .unwrap_or_else(|| request.uri().path().to_string());
    let body = to_bytes(request.into_body(), usize::MAX).await.unwrap();
    let payload = if body.is_empty() {
        json!({})
    } else {
        serde_json::from_slice(&body).expect("request body should be JSON")
    };
    captured.lock().await.push(json!({
        "method": method,
        "uri": uri,
        "body": payload
    }));
    Json(json!({
        "ok": true,
        "received": payload
    }))
}

async fn capture_multipart_request(
    State(captured): State<Arc<Mutex<Vec<Value>>>>,
    headers: HeaderMap,
    request: axum::extract::Request,
) -> impl IntoResponse {
    let method = request.method().as_str().to_string();
    let uri = request
        .uri()
        .path_and_query()
        .map(|value| value.as_str().to_string())
        .unwrap_or_else(|| request.uri().path().to_string());
    let content_type = headers
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .unwrap_or("")
        .to_string();
    let body = to_bytes(request.into_body(), usize::MAX).await.unwrap();
    let body_text = String::from_utf8_lossy(&body);
    let filenames = body_text
        .split("filename=\"")
        .skip(1)
        .filter_map(|tail| tail.split('"').next())
        .map(str::to_string)
        .collect::<Vec<_>>();
    captured.lock().await.push(json!({
        "method": method,
        "uri": uri,
        "content_type": content_type,
        "filenames": filenames,
        "body": body_text.to_string()
    }));
    Json(json!({
        "ok": true,
        "task_id": "kb_init_20260530_120000_cli"
    }))
}

fn make_executable(path: &std::path::Path) {
    let mut permissions = fs::metadata(path).unwrap().permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(path, permissions).unwrap();
}

fn read_pid(path: &std::path::Path) -> u32 {
    fs::read_to_string(path)
        .expect("pid file should exist")
        .trim()
        .parse()
        .expect("pid file should contain a pid")
}

fn process_is_alive(pid: u32) -> bool {
    std::process::Command::new("kill")
        .args(["-0", &pid.to_string()])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn assert_process_stopped(pid: u32) {
    for _ in 0..20 {
        if !process_is_alive(pid) {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    assert!(!process_is_alive(pid), "process {pid} should be stopped");
}

fn free_tcp_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .expect("should bind an ephemeral port")
        .local_addr()
        .expect("listener should have local addr")
        .port()
}

async fn plugins_list_response() -> Json<Value> {
    Json(json!({
        "plugins": [{
            "name": "deep_research",
            "version": "0.1.0",
            "type": "playground",
            "description": "Multi-agent research and reporting",
            "stages": ["plan", "research", "write"]
        }],
        "tools": [{
            "name": "rag",
            "description": "Retrieval-Augmented Generation",
            "schema": {"type": "object"}
        }],
        "capabilities": [{
            "name": "deep_solve",
            "description": "Multi-step reasoning",
            "stages": ["planner", "executor", "critic"],
            "tools_used": ["rag"],
            "config_defaults": {}
        }]
    }))
}

struct RealCliServer {
    api_url: String,
    data_root: std::path::PathBuf,
    server: tokio::task::JoinHandle<()>,
}

async fn spawn_real_cli_server(label: &str) -> RealCliServer {
    let data_root = unique_temp_dir(label).join("data");
    let knowledge_root = data_root.join("knowledge");
    fs::create_dir_all(&knowledge_root).expect("knowledge root should be created");
    let app = socartes_backend::app_with_knowledge_root_and_auth(knowledge_root, false);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    RealCliServer {
        api_url: format!("http://{address}"),
        data_root,
        server,
    }
}

async fn wait_for_knowledge_task(api_url: &str, task_id: &str) -> String {
    tokio::time::timeout(std::time::Duration::from_secs(10), async {
        reqwest::get(format!(
            "{}/api/v1/knowledge/tasks/{}/stream",
            api_url.trim_end_matches('/'),
            task_id
        ))
        .await
        .expect("task stream request should complete")
        .text()
        .await
        .expect("task stream should be utf-8")
    })
    .await
    .expect("knowledge task stream should finish")
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
fn socartes_cli_binary_alias_exposes_same_python_cli_surface() {
    let output = socartes_cli_cmd()
        .arg("--help")
        .output()
        .expect("socartes-cli alias should run");

    assert!(
        output.status.success(),
        "socartes-cli --help failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("stdout should be utf-8");
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
fn deeptutor_compatibility_aliases_expose_same_cli_surface() {
    for mut command in [deeptutor_cmd(), deeptutor_cli_cmd()] {
        let output = command
            .arg("--help")
            .output()
            .expect("deeptutor compatibility alias should run");

        assert!(
            output.status.success(),
            "deeptutor compatibility alias --help failed:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8(output.stdout).expect("stdout should be utf-8");
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
}

#[test]
fn python_module_style_underscore_aliases_expose_same_cli_surface() {
    for mut command in [
        socartes_cli_underscore_cmd(),
        deeptutor_cli_underscore_cmd(),
    ] {
        let output = command
            .arg("--help")
            .output()
            .expect("underscore CLI compatibility alias should run");

        assert!(
            output.status.success(),
            "underscore CLI compatibility alias --help failed:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8(output.stdout).expect("stdout should be utf-8");
        assert_contains_all(
            &stdout,
            &[
                "Socartes CLI",
                "run",
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
    assert_contains_all(
        &stdout,
        &[
            "--home",
            "--host",
            "--port",
            "--frontend-dir",
            "--frontend-port",
            "--dry-run",
        ],
    );
}

#[test]
fn cli_default_ports_match_python_launcher_defaults() {
    let top_help = stdout_for(&["--help"]);
    assert_contains_all(&top_help, &["http://127.0.0.1:8001"]);

    let serve_help = stdout_for(&["serve", "--help"]);
    assert_contains_all(&serve_help, &["--port", "8001"]);

    let home = unique_temp_dir("default-ports");
    let output = socartes_cmd()
        .args(["config", "show", "--home"])
        .arg(&home)
        .output()
        .expect("socartes config show should execute");
    assert!(
        output.status.success(),
        "config show failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let value: Value =
        serde_json::from_slice(&output.stdout).expect("config output should be JSON");
    assert_eq!(value["ports"]["backend"], 8001);
    assert_eq!(value["ports"]["frontend"], 3782);
    let _ = fs::remove_dir_all(home);
}

#[test]
fn start_dry_run_plans_backend_and_frontend_like_python_launcher() {
    let home = unique_temp_dir("start-home");
    let frontend = unique_temp_dir("start-frontend");
    fs::create_dir_all(&frontend).expect("frontend dir should be created");

    let output = socartes_cmd()
        .args(["start", "--dry-run", "--home"])
        .arg(&home)
        .args([
            "--host",
            "127.0.0.1",
            "--port",
            "8123",
            "--frontend-port",
            "3123",
            "--frontend-dir",
        ])
        .arg(&frontend)
        .output()
        .expect("socartes start dry-run should execute");

    let _ = fs::remove_dir_all(&home);
    let _ = fs::remove_dir_all(&frontend);

    assert!(
        output.status.success(),
        "start dry-run failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let plan: Value =
        serde_json::from_slice(&output.stdout).expect("start dry-run should print JSON");
    assert_eq!(plan["backend"]["host"], "127.0.0.1");
    assert_eq!(plan["backend"]["port"], 8123);
    assert_eq!(plan["backend"]["url"], "http://127.0.0.1:8123");
    assert_eq!(plan["backend"]["command"][1], "serve");
    assert!(
        plan["backend"]["command"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| item.as_str() == Some("8123")),
        "backend command should include the configured backend port: {plan}"
    );
    assert_eq!(plan["frontend"]["port"], 3123);
    assert_eq!(plan["frontend"]["url"], "http://localhost:3123");
    assert_eq!(
        plan["frontend"]["cwd"].as_str(),
        Some(frontend.to_string_lossy().as_ref())
    );
    assert_eq!(plan["frontend"]["command"][0], "npm");
    assert_eq!(plan["frontend"]["command"][1], "run");
    assert_eq!(plan["frontend"]["command"][2], "dev");
    assert_eq!(
        plan["frontend"]["env"]["NEXT_PUBLIC_API_BASE"],
        "http://localhost:8123"
    );
    assert_eq!(
        plan["state_path"].as_str(),
        Some(
            home.join("data/user/settings/start_web_state.json")
                .to_string_lossy()
                .as_ref()
        )
    );
    assert_eq!(plan["state"]["version"], 1);
    assert_eq!(plan["state"]["backend_port"], 8123);
    assert_eq!(plan["state"]["frontend_port"], 3123);
    assert_eq!(plan["state"]["processes"]["backend"]["pid"], Value::Null);
    assert_eq!(plan["state"]["processes"]["backend"]["pgid"], Value::Null);
    assert_eq!(plan["state"]["processes"]["frontend"]["pid"], Value::Null);
    assert_eq!(plan["state"]["processes"]["frontend"]["pgid"], Value::Null);
}

#[test]
fn start_dry_run_reads_env_file_ports_and_auth_like_python_launcher() {
    let project = unique_temp_dir("start-project");
    let frontend = project.join("web");
    fs::create_dir_all(&frontend).expect("frontend dir should be created");
    fs::write(
        project.join(".env"),
        "BACKEND_PORT=8124\nFRONTEND_PORT=3124\nAUTH_ENABLED=true\n",
    )
    .expect(".env should be written");

    let output = socartes_cmd()
        .current_dir(&project)
        .args(["start", "--dry-run"])
        .output()
        .expect("socartes start dry-run should execute");

    let _ = fs::remove_dir_all(&project);

    assert!(
        output.status.success(),
        "start dry-run with .env failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let plan: Value =
        serde_json::from_slice(&output.stdout).expect("start dry-run should print JSON");
    assert_eq!(plan["backend"]["port"], 8124);
    assert_eq!(plan["frontend"]["port"], 3124);
    assert_eq!(
        plan["frontend"]["cwd"].as_str(),
        Some(frontend.to_string_lossy().as_ref())
    );
    assert_eq!(
        plan["frontend"]["env"]["NEXT_PUBLIC_API_BASE"],
        "http://localhost:8124"
    );
    assert_eq!(plan["frontend"]["env"]["AUTH_ENABLED"], "true");
    assert_eq!(plan["frontend"]["env"]["NEXT_PUBLIC_AUTH_ENABLED"], "true");
    assert_eq!(
        plan["state_path"].as_str(),
        Some(
            project
                .join("data/user/settings/start_web_state.json")
                .to_string_lossy()
                .as_ref()
        )
    );
}

#[test]
fn start_dry_run_uses_launcher_overrides_for_cleanup_contract_tests() {
    let project = unique_temp_dir("start-overrides");
    let frontend = project.join("web");
    let backend_stub = project.join("backend-stub.sh");
    let frontend_stub = project.join("frontend-stub.sh");
    fs::create_dir_all(&frontend).expect("frontend dir should be created");
    fs::write(&backend_stub, "#!/bin/sh\nsleep 1\n").expect("backend stub should be written");
    fs::write(&frontend_stub, "#!/bin/sh\nsleep 1\n").expect("frontend stub should be written");

    let output = socartes_cmd()
        .current_dir(&project)
        .env("SOCARTES_START_BACKEND_COMMAND", &backend_stub)
        .env("SOCARTES_START_FRONTEND_COMMAND", &frontend_stub)
        .env("SOCARTES_START_BACKEND_READY_TIMEOUT_MS", "25")
        .env("SOCARTES_START_FRONTEND_READY_TIMEOUT_MS", "50")
        .args(["start", "--dry-run"])
        .output()
        .expect("socartes start dry-run should execute");

    let _ = fs::remove_dir_all(&project);

    assert!(
        output.status.success(),
        "start dry-run with launcher overrides failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let plan: Value =
        serde_json::from_slice(&output.stdout).expect("start dry-run should print JSON");
    assert_eq!(
        plan["backend"]["command"],
        json!([backend_stub.to_string_lossy()])
    );
    assert_eq!(
        plan["frontend"]["command"],
        json!([frontend_stub.to_string_lossy()])
    );
    assert_eq!(plan["readiness"]["backend_timeout_ms"], 25);
    assert_eq!(plan["readiness"]["frontend_timeout_ms"], 50);
}

#[test]
fn start_cleans_backend_and_state_when_backend_readiness_times_out() {
    let project = unique_temp_dir("start-backend-timeout");
    let frontend = project.join("web");
    let backend_stub = project.join("backend-stub.sh");
    let frontend_stub = project.join("frontend-stub.sh");
    let backend_pid = project.join("backend.pid");
    let state_path = project.join("data/user/settings/start_web_state.json");
    let backend_port = free_tcp_port();
    let frontend_port = free_tcp_port();
    let backend_port_arg = backend_port.to_string();
    let frontend_port_arg = frontend_port.to_string();
    fs::create_dir_all(&frontend).expect("frontend dir should be created");
    fs::write(
        &backend_stub,
        format!(
            "#!/bin/sh\nexec >/dev/null 2>/dev/null\nprintf '%s' \"$$\" > '{}'\nsleep 30\n",
            backend_pid.display()
        ),
    )
    .expect("backend stub should be written");
    fs::write(
        &frontend_stub,
        "#!/bin/sh\nexec >/dev/null 2>/dev/null\nsleep 30\n",
    )
    .expect("frontend stub should be written");
    make_executable(&backend_stub);
    make_executable(&frontend_stub);

    let output = socartes_cmd()
        .current_dir(&project)
        .env("SOCARTES_START_BACKEND_COMMAND", &backend_stub)
        .env("SOCARTES_START_FRONTEND_COMMAND", &frontend_stub)
        .env("SOCARTES_START_BACKEND_READY_TIMEOUT_MS", "25")
        .args([
            "start",
            "--port",
            &backend_port_arg,
            "--frontend-port",
            &frontend_port_arg,
        ])
        .output()
        .expect("socartes start should execute");

    let pid = read_pid(&backend_pid);
    assert!(
        !output.status.success(),
        "backend readiness timeout should fail:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_process_stopped(pid);
    assert!(
        !state_path.exists(),
        "state file should be removed on backend readiness failure"
    );

    let _ = fs::remove_dir_all(project);
}

#[test]
fn start_cleans_stale_recorded_processes_before_launch_like_python_launcher() {
    let project = unique_temp_dir("start-stale-state");
    let frontend = project.join("web");
    let backend_stub = project.join("backend-stub.sh");
    let frontend_stub = project.join("frontend-stub.sh");
    let backend_pid = project.join("backend.pid");
    let state_path = project.join("data/user/settings/start_web_state.json");
    let backend_port = free_tcp_port();
    let frontend_port = free_tcp_port();
    fs::create_dir_all(&frontend).expect("frontend dir should be created");
    fs::write(
        &backend_stub,
        format!(
            "#!/bin/sh\nexec >/dev/null 2>/dev/null\nprintf '%s' \"$$\" > '{}'\nsleep 30\n",
            backend_pid.display()
        ),
    )
    .expect("backend stub should be written");
    fs::write(
        &frontend_stub,
        "#!/bin/sh\nexec >/dev/null 2>/dev/null\nsleep 30\n",
    )
    .expect("frontend stub should be written");
    make_executable(&backend_stub);
    make_executable(&frontend_stub);

    let mut stale_backend = std::process::Command::new("sleep")
        .arg("30")
        .spawn()
        .expect("stale backend process should start");
    let mut stale_frontend = std::process::Command::new("sleep")
        .arg("30")
        .spawn()
        .expect("stale frontend process should start");
    fs::create_dir_all(state_path.parent().unwrap()).expect("state dir should be created");
    fs::write(
        &state_path,
        serde_json::to_vec_pretty(&json!({
            "version": 1,
            "created_at": "2026-05-30T00:00:00Z",
            "backend_port": backend_port,
            "frontend_port": frontend_port,
            "processes": {
                "backend": {"pid": stale_backend.id(), "pgid": null},
                "frontend": {"pid": stale_frontend.id(), "pgid": null}
            }
        }))
        .unwrap(),
    )
    .expect("stale state should be written");

    let output = socartes_cmd()
        .current_dir(&project)
        .env("SOCARTES_START_BACKEND_COMMAND", &backend_stub)
        .env("SOCARTES_START_FRONTEND_COMMAND", &frontend_stub)
        .env("SOCARTES_START_BACKEND_READY_TIMEOUT_MS", "25")
        .args([
            "start",
            "--port",
            &backend_port.to_string(),
            "--frontend-port",
            &frontend_port.to_string(),
        ])
        .output()
        .expect("socartes start should execute");

    let stale_backend_exited = stale_backend
        .try_wait()
        .expect("stale backend status should be readable")
        .is_some();
    let stale_frontend_exited = stale_frontend
        .try_wait()
        .expect("stale frontend status should be readable")
        .is_some();
    if !stale_backend_exited {
        let _ = stale_backend.kill();
    }
    if !stale_frontend_exited {
        let _ = stale_frontend.kill();
    }
    let _ = stale_backend.wait();
    let _ = stale_frontend.wait();

    assert!(
        !output.status.success(),
        "backend readiness timeout should still fail after stale cleanup:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_process_stopped(read_pid(&backend_pid));
    assert!(
        stale_backend_exited,
        "stale backend process should be stopped before new launch"
    );
    assert!(
        stale_frontend_exited,
        "stale frontend process should be stopped before new launch"
    );
    assert!(
        !state_path.exists(),
        "state file should be removed after failed launch cleanup"
    );

    let _ = fs::remove_dir_all(project);
}

#[test]
fn start_reports_port_owner_before_spawning_like_python_launcher() {
    let project = unique_temp_dir("start-port-conflict");
    let frontend = project.join("web");
    let backend_stub = project.join("backend-stub.sh");
    let frontend_stub = project.join("frontend-stub.sh");
    let backend_pid = project.join("backend.pid");
    fs::create_dir_all(&frontend).expect("frontend dir should be created");
    fs::write(
        &backend_stub,
        format!(
            "#!/bin/sh\nexec >/dev/null 2>/dev/null\nprintf '%s' \"$$\" > '{}'\nsleep 30\n",
            backend_pid.display()
        ),
    )
    .expect("backend stub should be written");
    fs::write(
        &frontend_stub,
        "#!/bin/sh\nexec >/dev/null 2>/dev/null\nsleep 30\n",
    )
    .expect("frontend stub should be written");
    make_executable(&backend_stub);
    make_executable(&frontend_stub);

    let occupied_backend =
        std::net::TcpListener::bind("127.0.0.1:0").expect("backend conflict listener should bind");
    let backend_port = occupied_backend
        .local_addr()
        .expect("backend listener should have address")
        .port();
    let frontend_port = free_tcp_port();

    let output = socartes_cmd()
        .current_dir(&project)
        .env("SOCARTES_START_BACKEND_COMMAND", &backend_stub)
        .env("SOCARTES_START_FRONTEND_COMMAND", &frontend_stub)
        .env("SOCARTES_START_BACKEND_READY_TIMEOUT_MS", "25")
        .args([
            "start",
            "--port",
            &backend_port.to_string(),
            "--frontend-port",
            &frontend_port.to_string(),
        ])
        .output()
        .expect("socartes start should execute");

    assert!(
        !output.status.success(),
        "port conflict should fail before launch:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains(&format!("Backend port {backend_port} is already in use.")),
        "start should report the occupied backend port like Python launcher:\n{stderr}"
    );
    assert!(
        stderr.contains("owner:"),
        "start should report the port owner or unknown owner:\n{stderr}"
    );
    assert!(
        stderr.contains("Stop the existing process"),
        "start should print the Python launcher conflict hint:\n{stderr}"
    );
    assert!(
        !backend_pid.exists(),
        "backend command should not be spawned when the selected port is already occupied"
    );

    drop(occupied_backend);
    let _ = fs::remove_dir_all(project);
}

#[test]
fn start_cleans_started_processes_when_frontend_readiness_times_out() {
    let project = unique_temp_dir("start-frontend-timeout");
    let frontend = project.join("web");
    let frontend_stub = project.join("frontend-stub.sh");
    let backend_pid = project.join("backend.pid");
    let frontend_pid = project.join("frontend.pid");
    let state_path = project.join("data/user/settings/start_web_state.json");
    let backend_port = free_tcp_port();
    let frontend_port = free_tcp_port();
    let backend_port_arg = backend_port.to_string();
    let frontend_port_arg = frontend_port.to_string();
    fs::create_dir_all(&frontend).expect("frontend dir should be created");
    fs::write(
        &frontend_stub,
        format!(
            "#!/bin/sh\nexec >/dev/null 2>/dev/null\nprintf '%s' \"$$\" > '{}'\nsleep 30\n",
            frontend_pid.display()
        ),
    )
    .expect("frontend stub should be written");
    make_executable(&frontend_stub);

    let backend_code = format!(
        "import http.server, os\n\
         devnull = open(os.devnull, 'w')\n\
         os.dup2(devnull.fileno(), 1)\n\
         os.dup2(devnull.fileno(), 2)\n\
         from pathlib import Path\n\
         Path(r'{pid_path}').write_text(str(os.getpid()))\n\
         server = http.server.ThreadingHTTPServer(('127.0.0.1', int(os.environ['BACKEND_PORT'])), http.server.SimpleHTTPRequestHandler)\n\
         server.serve_forever()\n",
        pid_path = backend_pid.display()
    );
    let backend_command = json!(["python3", "-c", backend_code]).to_string();

    let output = socartes_cmd()
        .current_dir(&project)
        .env("SOCARTES_START_BACKEND_COMMAND", backend_command)
        .env("SOCARTES_START_FRONTEND_COMMAND", &frontend_stub)
        .env("SOCARTES_START_BACKEND_READY_TIMEOUT_MS", "2000")
        .env("SOCARTES_START_FRONTEND_READY_TIMEOUT_MS", "25")
        .args([
            "start",
            "--port",
            &backend_port_arg,
            "--frontend-port",
            &frontend_port_arg,
        ])
        .output()
        .expect("socartes start should execute");

    let backend_pid = read_pid(&backend_pid);
    let frontend_pid = read_pid(&frontend_pid);
    assert!(
        !output.status.success(),
        "frontend readiness timeout should fail:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_process_stopped(frontend_pid);
    assert_process_stopped(backend_pid);
    assert!(
        !state_path.exists(),
        "state file should be removed on frontend readiness failure"
    );

    let _ = fs::remove_dir_all(project);
}

#[test]
fn provider_login_rejects_unknown_provider_like_python() {
    let output = socartes_cmd()
        .args(["provider", "login", "not-real"])
        .output()
        .expect("socartes provider login should execute");

    assert!(
        !output.status.success(),
        "unknown provider should fail:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_contains_all(
        &stderr,
        &[
            "Unknown provider `not-real`",
            "openai-codex",
            "github-copilot",
        ],
    );
}

#[test]
fn provider_login_openai_codex_reads_codex_auth_file_like_python_oauth_storage() {
    let codex_home = unique_temp_dir("codex-auth");
    fs::create_dir_all(&codex_home).expect("codex home should be created");
    fs::write(
        codex_home.join("auth.json"),
        serde_json::to_vec_pretty(&json!({
            "auth_mode": "chatgpt",
            "tokens": {
                "access_token": "codex-access-token",
                "refresh_token": "codex-refresh-token"
            }
        }))
        .unwrap(),
    )
    .expect("codex auth file should be written");

    let output = socartes_cmd()
        .env_remove("SOCARTES_OPENAI_CODEX_ACCESS_TOKEN")
        .env_remove("OPENAI_CODEX_ACCESS_TOKEN")
        .env("CODEX_HOME", &codex_home)
        .args(["provider", "login", "openai-codex"])
        .output()
        .expect("socartes provider login should execute");

    let _ = fs::remove_dir_all(codex_home);

    assert!(
        output.status.success(),
        "openai-codex auth file validation failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert_contains_all(&stdout, &["OpenAI Codex OAuth authentication succeeded."]);
    assert!(
        !stdout.contains("codex-access-token") && !stdout.contains("codex-refresh-token"),
        "provider login must not print OAuth token material"
    );
}

#[test]
fn provider_login_openai_codex_invokes_helper_when_token_storage_is_empty() {
    let codex_home = unique_temp_dir("codex-auth-empty");
    let helper_dir = unique_temp_dir("codex-helper");
    fs::create_dir_all(&codex_home).expect("codex home should be created");
    fs::create_dir_all(&helper_dir).expect("helper dir should be created");
    let marker = helper_dir.join("called");
    let helper = helper_dir.join("codex-oauth-helper.sh");
    fs::write(
        &helper,
        format!(
            "#!/bin/sh\nprintf called > '{}'\nexit 0\n",
            marker.display()
        ),
    )
    .expect("helper script should be written");
    let mut permissions = fs::metadata(&helper).unwrap().permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(&helper, permissions).unwrap();

    let output = socartes_cmd()
        .env_remove("SOCARTES_OPENAI_CODEX_ACCESS_TOKEN")
        .env_remove("OPENAI_CODEX_ACCESS_TOKEN")
        .env("CODEX_HOME", &codex_home)
        .env("SOCARTES_OPENAI_CODEX_HELPER", &helper)
        .args(["provider", "login", "openai-codex"])
        .output()
        .expect("socartes provider login should execute");

    let marker_exists = marker.is_file();
    let _ = fs::remove_dir_all(codex_home);
    let _ = fs::remove_dir_all(helper_dir);

    assert!(
        output.status.success(),
        "openai-codex helper fallback failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        marker_exists,
        "helper should be invoked when token storage is empty"
    );
    assert_contains_all(
        &String::from_utf8(output.stdout).unwrap(),
        &["OpenAI Codex OAuth authentication succeeded."],
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn provider_login_github_copilot_validates_existing_auth_without_socartes_api() {
    let captured = Arc::new(Mutex::new(None));
    let app = Router::new()
        .route("/chat/completions", post(capture_github_copilot_validation))
        .with_state(captured.clone());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let output = socartes_cmd()
        .env(
            "SOCARTES_GITHUB_COPILOT_BASE_URL",
            format!("http://{address}"),
        )
        .args(["provider", "login", "github-copilot"])
        .output()
        .expect("socartes provider login should execute");
    server.abort();

    assert!(
        output.status.success(),
        "github copilot validation failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_contains_all(
        &String::from_utf8(output.stdout).unwrap(),
        &["GitHub Copilot auth validation succeeded."],
    );

    let request = captured
        .lock()
        .await
        .clone()
        .expect("mock Copilot endpoint should capture validation request");
    assert_eq!(request["authorization"], "Bearer copilot");
    assert_eq!(request["payload"]["model"], "gpt-4o");
    assert_eq!(request["payload"]["max_tokens"], 1);
    assert_eq!(
        request["payload"]["messages"],
        json!([{"role":"user","content":"ping"}])
    );
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
        "data/user/settings",
    ] {
        assert!(
            home.join(relative).is_dir(),
            "expected init to create {}",
            home.join(relative).display()
        );
    }

    let catalog_text = fs::read_to_string(home.join("data/user/settings/model_catalog.json"))
        .expect("model_catalog.json should be written");
    let catalog: Value = serde_json::from_str(&catalog_text).unwrap();
    assert_eq!(
        catalog["services"]["llm"]["active_profile_id"],
        "socartes-rust"
    );
    assert!(
        home.join("data/user/settings/interface.json").is_file(),
        "interface.json should be written"
    );

    let _ = fs::remove_dir_all(home);
}

#[test]
fn init_wizard_subcommand_creates_same_runtime_layout() {
    let home = unique_temp_dir("init-wizard");
    let output = socartes_cmd()
        .args(["init", "wizard", "--yes", "--cli", "--home"])
        .arg(&home)
        .args([
            "--language",
            "zh",
            "--backend-port",
            "8125",
            "--frontend-port",
            "3125",
        ])
        .output()
        .expect("socartes init wizard should execute");

    assert!(
        output.status.success(),
        "socartes init wizard failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let value: Value =
        serde_json::from_slice(&output.stdout).expect("init wizard output should be JSON");
    assert_eq!(value["initialized"], true);
    assert_eq!(value["cli_only"], true);
    let ui: Value = serde_json::from_slice(
        &fs::read(home.join("data/user/settings/interface.json"))
            .expect("ui settings should exist"),
    )
    .expect("ui settings should be JSON");
    assert_eq!(ui["language"], "zh");
    assert_eq!(ui["ports"]["backend"], 8125);
    assert_eq!(ui["ports"]["frontend"], 3125);
    assert!(home.join("data/user/workspace/notebook").is_dir());
    assert!(home.join("data/memory").is_dir());
    let _ = fs::remove_dir_all(home);
}

#[test]
fn init_wizard_prompts_for_runtime_settings_when_not_preseeded() {
    let home = unique_temp_dir("init-wizard-interactive");
    let output = socartes_cmd()
        .args(["init", "wizard", "--home"])
        .arg(&home)
        .write_stdin(
            "y\n\
             anthropic\n\
             https://llm.example/v1\n\
             sk-llm-interactive\n\
             claude-interactive\n\
             openai\n\
             https://embedding.example/v1\n\
             sk-embedding-interactive\n\
             text-embedding-interactive\n\
             1024\n\
             brave\n\
             https://search.example\n\
             sk-search-interactive\n\
             8129\n\
             3130\n\
             zh\n",
        )
        .output()
        .expect("socartes init wizard should execute");

    assert!(
        output.status.success(),
        "interactive init wizard failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert_contains_all(
        &stdout,
        &[
            "LLM binding",
            "Embedding model",
            "Search provider",
            "\"initialized\": true",
        ],
    );

    let catalog: Value = serde_json::from_str(
        &fs::read_to_string(home.join("data/user/settings/model_catalog.json"))
            .expect("catalog should be written"),
    )
    .unwrap();
    assert_eq!(
        catalog["services"]["llm"]["profiles"][0]["binding"],
        "anthropic"
    );
    assert_eq!(
        catalog["services"]["llm"]["profiles"][0]["base_url"],
        "https://llm.example/v1"
    );
    assert_eq!(
        catalog["services"]["llm"]["profiles"][0]["api_key"],
        "sk-llm-interactive"
    );
    assert_eq!(
        catalog["services"]["llm"]["profiles"][0]["models"][0]["model"],
        "claude-interactive"
    );
    assert_eq!(
        catalog["services"]["embedding"]["profiles"][0]["models"][0]["model"],
        "text-embedding-interactive"
    );
    assert_eq!(
        catalog["services"]["embedding"]["profiles"][0]["models"][0]["dimension"],
        1024
    );
    assert_eq!(
        catalog["services"]["search"]["profiles"][0]["provider"],
        "brave"
    );

    let ui: Value = serde_json::from_str(
        &fs::read_to_string(home.join("data/user/settings/interface.json"))
            .expect("ui should be written"),
    )
    .unwrap();
    assert_eq!(ui["ports"]["backend"], 8129);
    assert_eq!(ui["ports"]["frontend"], 3130);
    assert_eq!(ui["language"], "zh");

    let _ = fs::remove_dir_all(home);
}

#[test]
fn config_show_reads_local_runtime_settings_without_api_like_python() {
    let home = unique_temp_dir("config");
    let settings_root = home.join("data/user/settings");
    fs::create_dir_all(&settings_root).expect("settings directory should be created");
    fs::write(
        settings_root.join("model_catalog.json"),
        serde_json::to_vec_pretty(&json!({
            "services": {
                "llm": {
                    "active_profile_id": "llm-main",
                    "active_model_id": "model-main",
                    "profiles": [{
                        "id": "llm-main",
                        "binding": "openai",
                        "base_url": "https://llm.example/v1",
                        "api_key": "sk-llm-secret",
                        "api_version": "2026-05-30",
                        "extra_headers": {"X-Trace": "yes"},
                        "models": [{"id": "model-main", "model": "gpt-test"}]
                    }]
                },
                "embedding": {
                    "active_profile_id": "embedding-main",
                    "active_model_id": "embedding-model",
                    "profiles": [{
                        "id": "embedding-main",
                        "binding": "openai",
                        "base_url": "https://embedding.example/v1",
                        "api_key": "sk-embedding-secret",
                        "api_version": "",
                        "models": [{"id": "embedding-model", "model": "text-embedding-test", "dimension": 1536}]
                    }]
                },
                "search": {
                    "active_profile_id": "search-main",
                    "profiles": [{
                        "id": "search-main",
                        "provider": "brave",
                        "base_url": "https://search.example",
                        "proxy": "http://proxy.example",
                        "api_key": "search-secret"
                    }]
                }
            }
        }))
        .unwrap(),
    )
    .expect("catalog should be written");
    fs::write(
        settings_root.join("interface.json"),
        serde_json::to_vec_pretty(&json!({
            "language": "zh",
            "ports": {"backend": 8123, "frontend": 3123},
            "tools": {"rag": {}, "web_search": {}}
        }))
        .unwrap(),
    )
    .expect("ui settings should be written");

    let output = socartes_cmd()
        .args(["config", "show", "--home"])
        .arg(&home)
        .output()
        .expect("socartes config show should execute");

    assert!(
        output.status.success(),
        "config show failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        !stdout.contains("sk-llm-secret") && !stdout.contains("sk-embedding-secret"),
        "config output should not leak secrets:\n{stdout}"
    );
    let value: Value = serde_json::from_str(&stdout).expect("config output should be JSON");
    assert_eq!(value["ports"]["backend"], 8123);
    assert_eq!(value["ports"]["frontend"], 3123);
    assert_eq!(value["llm"]["binding_hint"], "openai");
    assert_eq!(value["llm"]["provider"], "openai");
    assert_eq!(value["llm"]["model"], "gpt-test");
    assert_eq!(value["llm"]["api_key"], "***");
    assert_eq!(value["embedding"]["model"], "text-embedding-test");
    assert_eq!(value["embedding"]["dimension"], 1536);
    assert_eq!(value["search"]["provider"], "brave");
    assert_eq!(value["search"]["api_key"], "***");
    assert_eq!(value["language"], "zh");
    assert_eq!(value["tools"], json!(["rag", "web_search"]));

    let _ = fs::remove_dir_all(home);
}

#[test]
fn config_show_merges_env_model_catalog_and_main_yaml_like_python_cli() {
    let home = unique_temp_dir("config-python-sources");
    let user_settings = home.join("data/user/settings");
    fs::create_dir_all(&user_settings).expect("user settings directory should be created");
    fs::write(
        home.join(".env"),
        "BACKEND_PORT=9001\n\
         FRONTEND_PORT=4001\n\
         LLM_BINDING=openai\n\
         LLM_MODEL=gemini-2.5-pro\n\
         LLM_API_KEY=sk-or-test\n\
         SEARCH_PROVIDER=brave\n\
         SEARCH_API_KEY=\n\
         EMBEDDING_BINDING=openai\n\
         EMBEDDING_MODEL=text-embedding-3-large\n\
         EMBEDDING_API_KEY=sk-embed\n\
         EMBEDDING_DIMENSION=3072\n",
    )
    .expect(".env should be written");
    fs::write(
        user_settings.join("model_catalog.json"),
        serde_json::to_vec_pretty(&json!({
            "version": 1,
            "services": {
                "llm": {"active_profile_id": null, "active_model_id": null, "profiles": []},
                "embedding": {"active_profile_id": null, "active_model_id": null, "profiles": []},
                "search": {"active_profile_id": null, "profiles": []}
            }
        }))
        .unwrap(),
    )
    .expect("model catalog should be written");
    fs::write(
        user_settings.join("main.yaml"),
        "system:\n  language: zh\ntools:\n  rag: {}\n  web_search: {}\n",
    )
    .expect("main.yaml should be written");
    fs::write(
        user_settings.join("agents.yaml"),
        "chat:\n  temperature: 0.99\n  max_tokens: 17\n",
    )
    .expect("agents.yaml should be written");

    let output = socartes_cmd()
        .env_clear()
        .env("HOME", &home)
        .args(["config", "show", "--home"])
        .arg(&home)
        .output()
        .expect("socartes config show should execute");

    let _ = fs::remove_dir_all(home);

    assert!(
        output.status.success(),
        "config show failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        !stdout.contains("sk-or-test") && !stdout.contains("sk-embed"),
        "config output should mask secrets:\n{stdout}"
    );
    assert!(
        !stdout.contains("0.99") && !stdout.contains("max_tokens"),
        "config show should not read agents.yaml:\n{stdout}"
    );
    let value: Value = serde_json::from_str(&stdout).expect("config output should be JSON");
    assert_eq!(value["ports"]["backend"], 9001);
    assert_eq!(value["ports"]["frontend"], 4001);
    assert_eq!(value["language"], "zh");
    assert_eq!(value["tools"], json!(["rag", "web_search"]));
    assert_eq!(value["llm"]["binding_hint"], "openai");
    assert_eq!(value["llm"]["provider"], "openrouter");
    assert_eq!(value["llm"]["provider_mode"], "gateway");
    assert_eq!(value["llm"]["model"], "gemini-2.5-pro");
    assert_eq!(value["llm"]["base_url"], "https://openrouter.ai/api/v1");
    assert_eq!(value["llm"]["api_key"], "***");
    assert_eq!(value["embedding"]["binding_hint"], "openai");
    assert_eq!(value["embedding"]["provider"], "openai");
    assert_eq!(value["embedding"]["model"], "text-embedding-3-large");
    assert_eq!(value["embedding"]["dimension"], 3072);
    assert_eq!(value["embedding"]["api_key"], "***");
    assert_eq!(value["search"]["requested_provider"], "brave");
    assert_eq!(value["search"]["provider"], "duckduckgo");
    assert_eq!(value["search"]["status"], "fallback");
    assert_eq!(
        value["search"]["fallback_reason"],
        "brave requires api_key, falling back to duckduckgo"
    );
}

#[test]
fn config_show_only_reads_system_language_from_main_yaml_like_python_cli() {
    let home = unique_temp_dir("config-system-language");
    let user_settings = home.join("data/user/settings");
    fs::create_dir_all(&user_settings).expect("user settings directory should be created");
    fs::write(
        user_settings.join("main.yaml"),
        "system:\n  language: zh\nassistant:\n  language: ko\ntools:\n  rag: {}\n",
    )
    .expect("main.yaml should be written");

    let output = socartes_cmd()
        .env_clear()
        .env("HOME", &home)
        .args(["config", "show", "--home"])
        .arg(&home)
        .output()
        .expect("socartes config show should execute");

    let _ = fs::remove_dir_all(home);

    assert!(
        output.status.success(),
        "config show failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let value: Value =
        serde_json::from_slice(&output.stdout).expect("config output should be JSON");
    assert_eq!(value["language"], "zh");
    assert_eq!(value["tools"], json!(["rag"]));
}

#[test]
fn init_accepts_non_interactive_runtime_settings_like_python_wizard() {
    let home = unique_temp_dir("init-config");
    let output = socartes_cmd()
        .args(["init", "--yes", "--cli", "--home"])
        .arg(&home)
        .args([
            "--llm-binding",
            "anthropic",
            "--llm-base-url",
            "https://llm.example/v1",
            "--llm-api-key",
            "sk-llm",
            "--llm-model",
            "claude-test",
            "--embedding-binding",
            "openai",
            "--embedding-base-url",
            "https://embedding.example/v1",
            "--embedding-api-key",
            "sk-embedding",
            "--embedding-model",
            "text-embedding-test",
            "--embedding-dimension",
            "1024",
            "--search-provider",
            "brave",
            "--search-base-url",
            "https://search.example",
            "--search-api-key",
            "sk-search",
            "--backend-port",
            "8123",
            "--frontend-port",
            "3123",
            "--language",
            "ko",
        ])
        .output()
        .expect("socartes init should execute");

    assert!(
        output.status.success(),
        "configured init failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let catalog: Value = serde_json::from_str(
        &fs::read_to_string(home.join("data/user/settings/model_catalog.json"))
            .expect("catalog should be written"),
    )
    .unwrap();
    assert_eq!(catalog["services"]["llm"]["active_profile_id"], "llm-main");
    assert_eq!(catalog["services"]["llm"]["active_model_id"], "claude-test");
    assert_eq!(
        catalog["services"]["llm"]["profiles"][0]["binding"],
        "anthropic"
    );
    assert_eq!(
        catalog["services"]["llm"]["profiles"][0]["base_url"],
        "https://llm.example/v1"
    );
    assert_eq!(
        catalog["services"]["llm"]["profiles"][0]["api_key"],
        "sk-llm"
    );
    assert_eq!(
        catalog["services"]["embedding"]["profiles"][0]["models"][0]["dimension"],
        1024
    );
    assert_eq!(
        catalog["services"]["search"]["profiles"][0]["provider"],
        "brave"
    );

    let ui: Value = serde_json::from_str(
        &fs::read_to_string(home.join("data/user/settings/interface.json"))
            .expect("ui should be written"),
    )
    .unwrap();
    assert_eq!(ui["ports"]["backend"], 8123);
    assert_eq!(ui["ports"]["frontend"], 3123);
    assert_eq!(ui["language"], "ko");

    let _ = fs::remove_dir_all(home);
}

#[test]
fn init_writes_env_and_interface_language_like_python_start_tour() {
    let home = unique_temp_dir("init-python-start-tour");
    let output = socartes_cmd()
        .args(["init", "--yes", "--cli", "--home"])
        .arg(&home)
        .args([
            "--llm-binding",
            "anthropic",
            "--llm-base-url",
            "https://llm.example/v1",
            "--llm-api-key",
            "sk-llm",
            "--llm-model",
            "claude-test",
            "--embedding-binding",
            "openai",
            "--embedding-base-url",
            "https://embedding.example/v1",
            "--embedding-api-key",
            "sk-embedding",
            "--embedding-model",
            "text-embedding-test",
            "--embedding-dimension",
            "1024",
            "--search-provider",
            "brave",
            "--search-base-url",
            "https://search.example",
            "--search-api-key",
            "sk-search",
            "--backend-port",
            "8123",
            "--frontend-port",
            "3123",
            "--language",
            "zh",
        ])
        .output()
        .expect("socartes init should execute");

    assert!(
        output.status.success(),
        "configured init failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let env_text = fs::read_to_string(home.join(".env")).expect(".env should be written");
    assert_contains_all(
        &env_text,
        &[
            "BACKEND_PORT=8123",
            "FRONTEND_PORT=3123",
            "LLM_BINDING=anthropic",
            "LLM_MODEL=claude-test",
            "LLM_API_KEY=sk-llm",
            "LLM_HOST=https://llm.example/v1",
            "EMBEDDING_BINDING=openai",
            "EMBEDDING_MODEL=text-embedding-test",
            "EMBEDDING_API_KEY=sk-embedding",
            "EMBEDDING_HOST=https://embedding.example/v1",
            "EMBEDDING_DIMENSION=1024",
            "SEARCH_PROVIDER=brave",
            "SEARCH_API_KEY=sk-search",
            "SEARCH_BASE_URL=https://search.example",
        ],
    );
    assert!(
        !env_text.contains("LLM_BASE_URL=") && !env_text.contains("EMBEDDING_BASE_URL="),
        "Python start_tour writes LLM_HOST/EMBEDDING_HOST keys:\n{env_text}"
    );

    let interface: Value = serde_json::from_str(
        &fs::read_to_string(home.join("data/user/settings/interface.json"))
            .expect("interface.json should be written"),
    )
    .expect("interface.json should be valid JSON");
    assert_eq!(interface["theme"], "light");
    assert_eq!(interface["language"], "zh");

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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_state_mutation_commands_echo_state_and_affect_next_turn_like_python() {
    let captured_turns = Arc::new(Mutex::new(Vec::new()));
    let app = Router::new()
        .route(
            "/api/v1/plugins/capabilities/deep_solve/execute-stream",
            post(capture_new_session_stream),
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
        .write_stdin(
            "/tool on rag\n/cap deep_solve\n/kb course-ai\n/history add session-old\n/notebook add nb1:r1\n/config set temperature=0\nsolve this\n/quit\n",
        )
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
            "\"capability\": \"deep_solve\"",
            "\"rag\"",
            "\"course-ai\"",
            "\"history_references\"",
            "\"session-old\"",
            "\"notebook_id\": \"nb1\"",
            "\"temperature\": 0",
            "first answer",
        ],
    );
    let turns = captured_turns.lock().await;
    assert_eq!(turns.len(), 1);
    assert_eq!(turns[0]["tools"], json!(["rag"]));
    assert_eq!(turns[0]["knowledge_bases"], json!(["course-ai"]));
    assert_eq!(turns[0]["history_references"], json!(["session-old"]));
    assert_eq!(
        turns[0]["notebook_references"],
        json!([{"notebook_id":"nb1","record_ids":["r1"]}])
    );
    assert_eq!(turns[0]["config"]["temperature"], 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_list_passes_limit_to_api_like_python_cli() {
    let captured = Arc::new(Mutex::new(None));
    let app = Router::new()
        .route("/api/v1/sessions", get(capture_session_list_uri))
        .with_state(captured.clone());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let output = socartes_cmd()
        .env("SOCARTES_API_URL", format!("http://{address}"))
        .args(["session", "list", "--limit", "1", "--format", "json"])
        .output()
        .expect("socartes session list should execute");
    server.abort();

    assert!(
        output.status.success(),
        "session list failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        captured.lock().await.as_deref(),
        Some("/api/v1/sessions?limit=1")
    );
    let stdout: Value = serde_json::from_slice(&output.stdout)
        .expect("session list --format json should print JSON");
    assert_eq!(stdout["sessions"].as_array().unwrap().len(), 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_open_enters_chat_repl_with_existing_preferences_like_python() {
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
        .args(["session", "open", "session-existing"])
        .write_stdin("/refs\n/quit\n")
        .output()
        .expect("socartes session open should execute");
    server.abort();

    assert!(
        output.status.success(),
        "session open failed:\nstdout:\n{}\nstderr:\n{}",
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
            "history-1",
            "nb1",
            "r1",
        ],
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn resource_cli_commands_call_python_compatible_api_routes() {
    let captured = Arc::new(Mutex::new(Vec::new()));
    let app = Router::new()
        .route("/api/v1/tutorbot", post(capture_json_request))
        .route(
            "/api/v1/book/books/book-1/health",
            get(capture_json_request),
        )
        .route("/api/v1/memory/clear", post(capture_json_request))
        .with_state(captured.clone());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let commands = [
        vec![
            "bot",
            "create",
            "exam-bot",
            "--name",
            "Exam Bot",
            "--persona",
            "Socratic tutor",
            "--model",
            "gpt-test",
        ],
        vec!["book", "health", "book-1"],
        vec!["memory", "clear", "summary", "--force"],
    ];
    for args in commands {
        let output = socartes_cmd()
            .env("SOCARTES_API_URL", format!("http://{address}"))
            .args(args)
            .output()
            .expect("socartes resource command should execute");
        assert!(
            output.status.success(),
            "resource command failed:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    server.abort();

    let calls = captured.lock().await;
    assert_eq!(calls.len(), 3);
    assert_eq!(calls[0]["method"], "POST");
    assert_eq!(calls[0]["uri"], "/api/v1/tutorbot");
    assert_eq!(calls[0]["body"]["bot_id"], "exam-bot");
    assert_eq!(calls[0]["body"]["name"], "Exam Bot");
    assert_eq!(calls[0]["body"]["persona"], "Socratic tutor");
    assert_eq!(calls[0]["body"]["model"], "gpt-test");
    assert_eq!(calls[1]["method"], "GET");
    assert_eq!(calls[1]["uri"], "/api/v1/book/books/book-1/health");
    assert_eq!(calls[2]["method"], "POST");
    assert_eq!(calls[2]["uri"], "/api/v1/memory/clear");
    assert_eq!(calls[2]["body"], json!({"file": "summary"}));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn kb_search_posts_rag_tool_params_like_python_cli() {
    let captured = Arc::new(Mutex::new(Vec::new()));
    let app = Router::new()
        .route(
            "/api/v1/plugins/tools/rag/execute",
            post(capture_json_request),
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
            "kb",
            "search",
            "course-ai",
            "What does the hidden plot event mean?",
            "--mode",
            "hybrid",
            "--format",
            "json",
        ])
        .output()
        .expect("socartes kb search should execute");
    server.abort();

    assert!(
        output.status.success(),
        "kb search failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let calls = captured.lock().await;
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0]["method"], "POST");
    assert_eq!(calls[0]["uri"], "/api/v1/plugins/tools/rag/execute");
    assert_eq!(
        calls[0]["body"],
        json!({
            "params": {
                "query": "What does the hidden plot event mean?",
                "kb_name": "course-ai",
                "mode": "hybrid"
            }
        })
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn kb_create_docs_dir_uploads_only_python_supported_file_types() {
    let docs_dir = unique_temp_dir("kb-docs-dir");
    fs::create_dir_all(docs_dir.join("nested")).expect("docs directory should be created");
    fs::write(docs_dir.join("lesson.md"), "# Lesson").expect("markdown should be written");
    fs::write(docs_dir.join("paper.PDF"), "%PDF-1.4").expect("pdf should be written");
    fs::write(docs_dir.join("nested").join("notes.txt"), "notes").expect("text should be written");
    fs::write(docs_dir.join("image.png"), [0_u8, 1, 2, 3]).expect("png should be written");
    fs::write(docs_dir.join("archive.bin"), [4_u8, 5, 6]).expect("bin should be written");

    let captured = Arc::new(Mutex::new(Vec::new()));
    let app = Router::new()
        .route("/api/v1/knowledge/create", post(capture_multipart_request))
        .with_state(captured.clone());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let output = socartes_cmd()
        .env("SOCARTES_API_URL", format!("http://{address}"))
        .args(["kb", "create", "course-ai", "--docs-dir"])
        .arg(&docs_dir)
        .output()
        .expect("socartes kb create should execute");
    server.abort();
    let _ = fs::remove_dir_all(docs_dir);

    assert!(
        output.status.success(),
        "kb create failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let calls = captured.lock().await;
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0]["method"], "POST");
    assert_eq!(calls[0]["uri"], "/api/v1/knowledge/create");
    let filenames = calls[0]["filenames"]
        .as_array()
        .expect("filenames should be captured")
        .iter()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>();
    assert_eq!(filenames, vec!["lesson.md", "notes.txt", "paper.PDF"]);
    assert!(
        calls[0]["body"]
            .as_str()
            .unwrap_or("")
            .contains("course-ai")
    );
    assert!(!filenames.contains(&"image.png"));
    assert!(!filenames.contains(&"archive.bin"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn kb_add_docs_dir_posts_upload_endpoint_and_filters_supported_files() {
    let docs_dir = unique_temp_dir("kb-add-docs-dir");
    fs::create_dir_all(docs_dir.join("nested")).expect("docs directory should be created");
    fs::write(docs_dir.join("append.md"), "# Appendix").expect("markdown should be written");
    fs::write(docs_dir.join("nested").join("append.txt"), "append")
        .expect("text should be written");
    fs::write(docs_dir.join("photo.jpg"), [0_u8, 1, 2]).expect("jpg should be written");

    let captured = Arc::new(Mutex::new(Vec::new()));
    let app = Router::new()
        .route(
            "/api/v1/knowledge/course-ai/upload",
            post(capture_multipart_request),
        )
        .with_state(captured.clone());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let output = socartes_cmd()
        .env("SOCARTES_API_URL", format!("http://{address}"))
        .args(["kb", "add", "course-ai", "--docs-dir"])
        .arg(&docs_dir)
        .output()
        .expect("socartes kb add should execute");
    server.abort();
    let _ = fs::remove_dir_all(docs_dir);

    assert!(
        output.status.success(),
        "kb add failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let calls = captured.lock().await;
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0]["method"], "POST");
    assert_eq!(calls[0]["uri"], "/api/v1/knowledge/course-ai/upload");
    let filenames = calls[0]["filenames"]
        .as_array()
        .expect("filenames should be captured")
        .iter()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>();
    assert_eq!(filenames, vec!["append.md", "append.txt"]);
    assert!(!filenames.contains(&"photo.jpg"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn notebook_add_md_posts_markdown_record_payload_like_python_cli() {
    let markdown = unique_temp_dir("notebook-md").with_extension("md");
    fs::write(&markdown, "# Lesson\n\nBody from markdown.").expect("markdown should be written");
    let captured = Arc::new(Mutex::new(Vec::new()));
    let app = Router::new()
        .route("/api/v1/notebook/add_record", post(capture_json_request))
        .with_state(captured.clone());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let output = socartes_cmd()
        .env("SOCARTES_API_URL", format!("http://{address}"))
        .args(["notebook", "add-md", "nb1"])
        .arg(&markdown)
        .args(["--title", "Imported Lesson", "--type", "course_note"])
        .output()
        .expect("socartes notebook add-md should execute");
    server.abort();
    let _ = fs::remove_file(markdown);

    assert!(
        output.status.success(),
        "notebook add-md failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let calls = captured.lock().await;
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0]["uri"], "/api/v1/notebook/add_record");
    assert_eq!(calls[0]["body"]["notebook_ids"], json!(["nb1"]));
    assert_eq!(calls[0]["body"]["record_type"], "course_note");
    assert_eq!(calls[0]["body"]["title"], "Imported Lesson");
    assert_eq!(
        calls[0]["body"]["output"],
        "# Lesson\n\nBody from markdown."
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remaining_resource_cli_commands_call_python_compatible_api_routes() {
    let markdown = unique_temp_dir("notebook-replace").with_extension("md");
    fs::write(&markdown, "Replacement markdown body.").expect("markdown should be written");
    let captured = Arc::new(Mutex::new(Vec::new()));
    let app = Router::new()
        .fallback(capture_json_request)
        .with_state(captured.clone());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let api_url = format!("http://{address}");

    let run = |args: &[&str]| {
        let output = socartes_cmd()
            .env("SOCARTES_API_URL", &api_url)
            .args(args)
            .output()
            .expect("socartes resource command should execute");
        assert!(
            output.status.success(),
            "resource command {args:?} failed:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    };

    for args in [
        vec!["book", "list", "--format", "json"],
        vec!["book", "refresh-fingerprints", "book-1"],
        vec!["bot", "list", "--format", "json"],
        vec!["bot", "start", "exam-bot"],
        vec!["bot", "stop", "exam-bot"],
        vec!["kb", "list", "--format", "json"],
        vec!["kb", "info", "course-ai"],
        vec!["kb", "set-default", "course-ai"],
        vec!["kb", "delete", "course-ai", "--force"],
        vec!["notebook", "list", "--format", "json"],
        vec![
            "notebook",
            "create",
            "Lecture Notes",
            "--description",
            "Imported",
        ],
        vec!["notebook", "show", "nb1", "--format", "json"],
        vec!["notebook", "remove-record", "nb1", "rec1"],
        vec!["memory", "show", "all", "--format", "json"],
        vec!["plugin", "list", "--format", "json"],
        vec!["session", "show", "s1", "--format", "json"],
        vec!["session", "delete", "s1"],
        vec!["session", "rename", "s1", "--title", "New title"],
        vec!["config", "show", "--api"],
    ] {
        run(&args);
    }

    let output = socartes_cmd()
        .env("SOCARTES_API_URL", &api_url)
        .args(["notebook", "replace-md", "nb1", "rec1"])
        .arg(&markdown)
        .output()
        .expect("socartes notebook replace-md should execute");
    server.abort();
    let _ = fs::remove_file(markdown);

    assert!(
        output.status.success(),
        "notebook replace-md failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let calls = captured.lock().await;
    let seen = calls
        .iter()
        .map(|call| {
            format!(
                "{} {}",
                call["method"].as_str().unwrap_or(""),
                call["uri"].as_str().unwrap_or("")
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        seen,
        vec![
            "GET /api/v1/book/books",
            "POST /api/v1/book/books/book-1/refresh-fingerprints",
            "GET /api/v1/tutorbot",
            "POST /api/v1/tutorbot",
            "DELETE /api/v1/tutorbot/exam-bot",
            "GET /api/v1/knowledge/list",
            "GET /api/v1/knowledge/course-ai",
            "PUT /api/v1/knowledge/default/course-ai",
            "DELETE /api/v1/knowledge/course-ai",
            "GET /api/v1/notebook/list",
            "POST /api/v1/notebook/create",
            "GET /api/v1/notebook/nb1",
            "DELETE /api/v1/notebook/nb1/records/rec1",
            "GET /api/v1/memory",
            "GET /api/v1/plugins/list",
            "GET /api/v1/sessions/s1",
            "DELETE /api/v1/sessions/s1",
            "PATCH /api/v1/sessions/s1",
            "GET /api/v1/settings",
            "PUT /api/v1/notebook/nb1/records/rec1",
        ]
    );
    assert_eq!(calls[3]["body"], json!({"bot_id": "exam-bot"}));
    assert_eq!(
        calls[10]["body"],
        json!({"name": "Lecture Notes", "description": "Imported"})
    );
    assert_eq!(calls[17]["body"], json!({"title": "New title"}));
    assert_eq!(
        calls[19]["body"],
        json!({"output": "Replacement markdown body."})
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn plugin_info_reads_tool_and_capability_entries_like_python_cli() {
    let app = Router::new().route("/api/v1/plugins/list", get(plugins_list_response));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let tool_output = socartes_cmd()
        .env("SOCARTES_API_URL", format!("http://{address}"))
        .args(["plugin", "info", "rag"])
        .output()
        .expect("socartes plugin info rag should execute");
    let capability_output = socartes_cmd()
        .env("SOCARTES_API_URL", format!("http://{address}"))
        .args(["plugin", "info", "deep_solve"])
        .output()
        .expect("socartes plugin info deep_solve should execute");
    let plugin_output = socartes_cmd()
        .env("SOCARTES_API_URL", format!("http://{address}"))
        .args(["plugin", "info", "deep_research"])
        .output()
        .expect("socartes plugin info deep_research should execute");
    server.abort();

    assert!(
        tool_output.status.success(),
        "plugin info rag failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&tool_output.stdout),
        String::from_utf8_lossy(&tool_output.stderr)
    );
    assert_contains_all(
        &String::from_utf8(tool_output.stdout).unwrap(),
        &["\"name\": \"rag\"", "Retrieval-Augmented Generation"],
    );
    assert!(
        capability_output.status.success(),
        "plugin info deep_solve failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&capability_output.stdout),
        String::from_utf8_lossy(&capability_output.stderr)
    );
    assert_contains_all(
        &String::from_utf8(capability_output.stdout).unwrap(),
        &[
            "\"name\": \"deep_solve\"",
            "Multi-step reasoning",
            "planner",
        ],
    );
    assert!(
        plugin_output.status.success(),
        "plugin info deep_research failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&plugin_output.stdout),
        String::from_utf8_lossy(&plugin_output.stderr)
    );
    assert_contains_all(
        &String::from_utf8(plugin_output.stdout).unwrap(),
        &[
            "\"name\": \"deep_research\"",
            "Multi-agent research",
            "playground",
        ],
    );
}

#[test]
fn memory_cli_falls_back_to_local_files_when_api_is_unavailable_like_python_cli() {
    let home = unique_temp_dir("memory-local-fallback");
    let data_root = home.join("data");
    let memory_root = data_root.join("memory");
    fs::create_dir_all(&memory_root).expect("memory root should be created");
    fs::write(memory_root.join("SUMMARY.md"), "Local CLI summary.")
        .expect("summary memory should be written");
    fs::write(memory_root.join("PROFILE.md"), "Local CLI profile.")
        .expect("profile memory should be written");
    let unavailable_api = format!("http://127.0.0.1:{}", free_tcp_port());

    let show_output = socartes_cmd()
        .env("SOCARTES_API_URL", &unavailable_api)
        .env("SOCARTES_DATA_DIR", &data_root)
        .args(["memory", "show", "summary"])
        .output()
        .expect("socartes memory show should execute");

    assert!(
        show_output.status.success(),
        "memory show should fall back to local files when API is unavailable:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&show_output.stdout),
        String::from_utf8_lossy(&show_output.stderr)
    );
    assert_contains_all(
        &String::from_utf8(show_output.stdout).unwrap(),
        &["Local CLI summary."],
    );

    let clear_output = socartes_cmd()
        .env("SOCARTES_API_URL", &unavailable_api)
        .env("SOCARTES_DATA_DIR", &data_root)
        .args(["memory", "clear", "profile", "--force"])
        .output()
        .expect("socartes memory clear should execute");

    assert!(
        clear_output.status.success(),
        "memory clear should fall back to local files when API is unavailable:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&clear_output.stdout),
        String::from_utf8_lossy(&clear_output.stderr)
    );
    assert_eq!(
        fs::read_to_string(memory_root.join("PROFILE.md")).unwrap_or_default(),
        ""
    );
    assert_eq!(
        fs::read_to_string(memory_root.join("SUMMARY.md")).unwrap(),
        "Local CLI summary."
    );

    let _ = fs::remove_dir_all(home);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cli_notebook_memory_and_plugin_work_against_real_rust_backend() {
    let real = spawn_real_cli_server("real-notebook-memory-plugin").await;
    let markdown = unique_temp_dir("real-notebook-record").with_extension("md");
    fs::write(
        &markdown,
        "# Imported Lesson\n\nRust CLI end-to-end record.",
    )
    .expect("markdown should be written");
    let memory_root = real.data_root.join("memory");
    fs::create_dir_all(&memory_root).expect("memory root should be created");
    fs::write(
        memory_root.join("SUMMARY.md"),
        "Socartes memory summary from disk.",
    )
    .expect("summary memory should be written");

    let create_output = socartes_cmd()
        .env("SOCARTES_API_URL", &real.api_url)
        .args([
            "notebook",
            "create",
            "Rust CLI E2E",
            "--description",
            "real backend",
        ])
        .output()
        .expect("socartes notebook create should execute");
    assert!(
        create_output.status.success(),
        "notebook create failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&create_output.stdout),
        String::from_utf8_lossy(&create_output.stderr)
    );
    let created: Value =
        serde_json::from_slice(&create_output.stdout).expect("notebook create should print JSON");
    let notebook_id = created["notebook"]["id"]
        .as_str()
        .expect("real backend should return notebook id")
        .to_string();

    let add_output = socartes_cmd()
        .env("SOCARTES_API_URL", &real.api_url)
        .args(["notebook", "add-md", &notebook_id])
        .arg(&markdown)
        .args(["--title", "Imported Lesson", "--type", "chat"])
        .output()
        .expect("socartes notebook add-md should execute");
    assert!(
        add_output.status.success(),
        "notebook add-md failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&add_output.stdout),
        String::from_utf8_lossy(&add_output.stderr)
    );

    let show_output = socartes_cmd()
        .env("SOCARTES_API_URL", &real.api_url)
        .args(["notebook", "show", &notebook_id, "--format", "json"])
        .output()
        .expect("socartes notebook show should execute");
    assert!(
        show_output.status.success(),
        "notebook show failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&show_output.stdout),
        String::from_utf8_lossy(&show_output.stderr)
    );
    let shown: Value =
        serde_json::from_slice(&show_output.stdout).expect("notebook show should print JSON");
    assert_eq!(shown["name"], "Rust CLI E2E");
    assert_eq!(shown["records"][0]["type"], "chat");
    assert_eq!(shown["records"][0]["title"], "Imported Lesson");
    assert_eq!(
        shown["records"][0]["output"],
        "# Imported Lesson\n\nRust CLI end-to-end record."
    );

    let memory_output = socartes_cmd()
        .env("SOCARTES_API_URL", &real.api_url)
        .args(["memory", "show", "summary"])
        .output()
        .expect("socartes memory show should execute");
    assert!(
        memory_output.status.success(),
        "memory show failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&memory_output.stdout),
        String::from_utf8_lossy(&memory_output.stderr)
    );
    assert_contains_all(
        &String::from_utf8(memory_output.stdout).unwrap(),
        &["Socartes memory summary from disk."],
    );

    let clear_output = socartes_cmd()
        .env("SOCARTES_API_URL", &real.api_url)
        .args(["memory", "clear", "summary", "--force"])
        .output()
        .expect("socartes memory clear should execute");
    assert!(
        clear_output.status.success(),
        "memory clear failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&clear_output.stdout),
        String::from_utf8_lossy(&clear_output.stderr)
    );
    assert_eq!(
        fs::read_to_string(memory_root.join("SUMMARY.md")).unwrap_or_default(),
        ""
    );

    let plugin_output = socartes_cmd()
        .env("SOCARTES_API_URL", &real.api_url)
        .args(["plugin", "info", "rag"])
        .output()
        .expect("socartes plugin info should execute");
    assert!(
        plugin_output.status.success(),
        "plugin info failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&plugin_output.stdout),
        String::from_utf8_lossy(&plugin_output.stderr)
    );
    assert_contains_all(
        &String::from_utf8(plugin_output.stdout).unwrap(),
        &["\"name\": \"rag\"", "Retrieval-Augmented Generation"],
    );

    real.server.abort();
    let _ = fs::remove_file(markdown);
    let _ = fs::remove_dir_all(real.data_root);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cli_kb_create_task_stream_and_search_work_against_real_rust_backend() {
    let real = spawn_real_cli_server("real-kb").await;
    let document = unique_temp_dir("real-kb-doc").with_extension("md");
    fs::write(
        &document,
        "# Hidden Lesson\n\nThe Socartes Rust CLI e2e keyword is orrery-quartz.",
    )
    .expect("knowledge document should be written");

    let create_output = socartes_cmd()
        .env("SOCARTES_API_URL", &real.api_url)
        .args(["kb", "create", "course-ai", "--doc"])
        .arg(&document)
        .output()
        .expect("socartes kb create should execute");
    assert!(
        create_output.status.success(),
        "kb create failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&create_output.stdout),
        String::from_utf8_lossy(&create_output.stderr)
    );
    let created: Value =
        serde_json::from_slice(&create_output.stdout).expect("kb create should print JSON");
    let task_id = created["task_id"]
        .as_str()
        .expect("real backend should return a knowledge task id");
    assert!(
        task_id.starts_with("kb_init_"),
        "Python-style create task id expected, got {task_id}"
    );

    let task_stream = wait_for_knowledge_task(&real.api_url, task_id).await;
    assert_contains_all(&task_stream, &["event: process_log", "event: complete"]);

    let list_output = socartes_cmd()
        .env("SOCARTES_API_URL", &real.api_url)
        .args(["kb", "list", "--format", "json"])
        .output()
        .expect("socartes kb list should execute");
    assert!(
        list_output.status.success(),
        "kb list failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&list_output.stdout),
        String::from_utf8_lossy(&list_output.stderr)
    );
    let listed: Value =
        serde_json::from_slice(&list_output.stdout).expect("kb list should print JSON");
    let names = listed
        .as_array()
        .expect("kb list should return the Python-compatible KB array")
        .iter()
        .filter_map(|item| item["name"].as_str())
        .collect::<Vec<_>>();
    assert!(
        names.contains(&"course-ai"),
        "real backend list should include created KB: {listed}"
    );

    let search_output = socartes_cmd()
        .env("SOCARTES_API_URL", &real.api_url)
        .args([
            "kb",
            "search",
            "course-ai",
            "What is the e2e keyword?",
            "--format",
            "json",
        ])
        .output()
        .expect("socartes kb search should execute");
    assert!(
        search_output.status.success(),
        "kb search failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&search_output.stdout),
        String::from_utf8_lossy(&search_output.stderr)
    );
    let searched: Value =
        serde_json::from_slice(&search_output.stdout).expect("kb search should print JSON");
    let search_text = searched.to_string();
    assert!(
        search_text.contains("orrery-quartz"),
        "real backend RAG search should cite indexed document: {searched}"
    );

    real.server.abort();
    let _ = fs::remove_file(document);
    let _ = fs::remove_dir_all(real.data_root);
}
