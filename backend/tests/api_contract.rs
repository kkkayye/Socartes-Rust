use std::{
    fs,
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

use axum::body::Body;
use axum::http;
use futures_util::{SinkExt, StreamExt};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use socartes_backend::{app, app_with_knowledge_root};
use tokio::net::TcpListener;
use tokio_tungstenite::{connect_async, tungstenite::Message as TungsteniteMessage};
use tower::ServiceExt;

static TEST_ROOT_COUNTER: AtomicU64 = AtomicU64::new(0);

async fn json_response(response: axum::response::Response) -> Value {
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("response body")
        .to_bytes();
    serde_json::from_slice(&bytes).expect("json response")
}

async fn text_response(response: axum::response::Response) -> String {
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("response body")
        .to_bytes();
    String::from_utf8(bytes.to_vec()).expect("utf8 response")
}

fn unique_test_knowledge_root() -> std::path::PathBuf {
    let id = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time")
        .as_nanos();
    let counter = TEST_ROOT_COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir()
        .join(format!(
            "socartes-test-{}-{counter}-{id}",
            std::process::id()
        ))
        .join("knowledge")
}

fn test_data_root(knowledge_root: &Path) -> std::path::PathBuf {
    knowledge_root
        .parent()
        .expect("test data root")
        .to_path_buf()
}

fn test_book_root(knowledge_root: &Path) -> std::path::PathBuf {
    test_data_root(knowledge_root)
        .join("user")
        .join("workspace")
        .join("book")
}

fn test_memory_root(knowledge_root: &Path) -> std::path::PathBuf {
    test_data_root(knowledge_root).join("memory")
}

fn test_skills_root(knowledge_root: &Path) -> std::path::PathBuf {
    test_data_root(knowledge_root)
        .join("user")
        .join("workspace")
        .join("skills")
}

fn test_co_writer_docs_root(knowledge_root: &Path) -> std::path::PathBuf {
    test_data_root(knowledge_root)
        .join("user")
        .join("workspace")
        .join("co-writer")
        .join("documents")
}

fn test_user_output_root(knowledge_root: &Path) -> std::path::PathBuf {
    test_data_root(knowledge_root).join("user")
}

#[tokio::test]
async fn health_endpoint_reports_backend_identity() {
    let response = app()
        .oneshot(
            http::Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), http::StatusCode::OK);
    assert_eq!(
        json_response(response).await,
        json!({
            "status": "ok",
            "service": "socartes-backend",
            "version": "0.1.0"
        })
    );
}

#[tokio::test]
async fn learn_endpoint_returns_traceable_agent_output() {
    let request_body = json!({
        "goal": "Explain how planner, executor, and critic agents work together.",
        "learner_context": "Prefer concise explanations."
    });

    let response = app()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/learn")
                .header("content-type", "application/json")
                .body(Body::from(request_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), http::StatusCode::OK);
    let payload = json_response(response).await;
    assert_eq!(payload["plan"]["agent"], "planner");
    assert_eq!(payload["review"]["status"], "approved");
    assert!(!payload["final_answer"].as_str().unwrap().is_empty());
    assert!(!payload["retrieved_context"].as_array().unwrap().is_empty());
    assert!(!payload["tool_results"].as_array().unwrap().is_empty());
    assert!(!payload["reflection_events"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn learn_endpoint_rejects_short_goals_like_the_python_contract() {
    let response = app()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/learn")
                .header("content-type", "application/json")
                .body(Body::from(json!({"goal": "AI"}).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), http::StatusCode::UNPROCESSABLE_ENTITY);
    let payload = json_response(response).await;
    assert_eq!(payload["detail"][0]["type"], "string_too_short");
    assert_eq!(payload["detail"][0]["loc"], json!(["body", "goal"]));
    assert_eq!(payload["detail"][0]["ctx"]["min_length"], 3);
}

#[tokio::test]
async fn agents_endpoint_documents_each_backend_worker() {
    let response = app()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/agents")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), http::StatusCode::OK);
    let payload = json_response(response).await;
    assert!(payload["agents"]["planner"].is_object());
    assert!(payload["agents"]["executor"].is_object());
    assert!(payload["agents"]["critic"].is_object());
}

#[tokio::test]
async fn live_chat_frontend_bootstrap_endpoints_are_available() {
    let knowledge_response = app()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/knowledge/list")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(knowledge_response.status(), http::StatusCode::OK);
    let knowledge_payload = json_response(knowledge_response).await;
    assert!(knowledge_payload["knowledge_bases"].is_array());

    let llm_response = app()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/settings/llm-options")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(llm_response.status(), http::StatusCode::OK);
    let llm_payload = json_response(llm_response).await;
    assert!(llm_payload["options"].is_array());

    let sessions_response = app()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/sessions?limit=50&offset=0")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(sessions_response.status(), http::StatusCode::OK);
    let sessions_payload = json_response(sessions_response).await;
    assert!(sessions_payload["sessions"].is_array());
}

#[tokio::test]
async fn course_knowledge_frontend_bootstrap_endpoints_are_available() {
    let root = unique_test_knowledge_root();
    let app = app_with_knowledge_root(&root);

    let providers_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/knowledge/rag-providers")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(providers_response.status(), http::StatusCode::OK);
    let providers_payload = json_response(providers_response).await;
    assert_eq!(providers_payload["providers"][0]["id"], "llamaindex");

    let policy_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/knowledge/supported-file-types")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(policy_response.status(), http::StatusCode::OK);
    let policy_payload = json_response(policy_response).await;
    assert!(
        policy_payload["extensions"]
            .as_array()
            .unwrap()
            .contains(&json!(".txt"))
    );
    assert_eq!(
        policy_payload["accept"],
        ".txt,.md,.markdown,.pdf,.json,.csv"
    );

    let files_response = app
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/knowledge/socartes-rust-rag/files")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(files_response.status(), http::StatusCode::OK);
    let files_payload = json_response(files_response).await;
    assert!(files_payload["files"].as_array().unwrap().len() >= 2);

    let _ = std::fs::remove_dir_all(root);
}

#[tokio::test]
async fn knowledge_python_config_progress_and_linked_folder_endpoints_match_contract() {
    let root = unique_test_knowledge_root();
    let app = app_with_knowledge_root(&root);

    let health_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/knowledge/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(health_response.status(), http::StatusCode::OK);
    let health = json_response(health_response).await;
    assert_eq!(health["status"], "ok");
    assert!(
        health["config_file"]
            .as_str()
            .unwrap()
            .ends_with("kb_config.json")
    );
    assert_eq!(health["base_dir_exists"], false);

    let default_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/knowledge/default")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(default_response.status(), http::StatusCode::OK);
    assert_eq!(
        json_response(default_response).await["default_kb"],
        "socartes-rust-rag"
    );

    let configs_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/knowledge/configs")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(configs_response.status(), http::StatusCode::OK);
    let configs = json_response(configs_response).await;
    assert_eq!(configs["defaults"]["rag_provider"], "llamaindex");
    assert_eq!(configs["defaults"]["search_mode"], "hybrid");
    assert!(configs["knowledge_bases"].is_object());

    let boundary = "SOCARTESCONFIGBOUNDARY";
    let body = format!(
        "--{boundary}\r\n\
Content-Disposition: form-data; name=\"name\"\r\n\r\n\
python-contract-course\r\n\
--{boundary}\r\n\
Content-Disposition: form-data; name=\"files\"; filename=\"notes.md\"\r\n\
Content-Type: text/markdown\r\n\r\n\
rust replacement notes\r\n\
--{boundary}--\r\n"
    );
    let create_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/knowledge/create")
                .header(
                    http::header::CONTENT_TYPE,
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(create_response.status(), http::StatusCode::OK);

    let config_update_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("PUT")
                .uri("/api/v1/knowledge/python-contract-course/config")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "search_mode": "semantic",
                        "description": "Python config compatibility",
                        "needs_reindex": true
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(config_update_response.status(), http::StatusCode::OK);
    let updated_config = json_response(config_update_response).await;
    assert_eq!(updated_config["status"], "success");
    assert_eq!(updated_config["kb_name"], "python-contract-course");
    assert_eq!(updated_config["config"]["search_mode"], "semantic");
    assert_eq!(updated_config["config"]["needs_reindex"], true);

    let config_get_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/knowledge/python-contract-course/config")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(config_get_response.status(), http::StatusCode::OK);
    let config_get = json_response(config_get_response).await;
    assert_eq!(config_get["config"]["default_kb"], "socartes-rust-rag");
    assert_eq!(config_get["config"]["rag_provider"], "llamaindex");
    assert_eq!(
        config_get["config"]["description"],
        "Python config compatibility"
    );

    let progress_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/knowledge/python-contract-course/progress")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(progress_response.status(), http::StatusCode::OK);
    assert_eq!(
        json_response(progress_response).await,
        json!({"status": "not_started", "message": "Initialization not started"})
    );

    let linked_dir = test_data_root(&root).join("linked-source");
    fs::create_dir_all(&linked_dir).unwrap();
    fs::write(linked_dir.join("linked.md"), "linked source").unwrap();

    let link_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/knowledge/python-contract-course/link-folder")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"folder_path": linked_dir.to_string_lossy()}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(link_response.status(), http::StatusCode::OK);
    let linked = json_response(link_response).await;
    let folder_id = linked["id"].as_str().unwrap();
    assert_eq!(
        linked["path"].as_str().unwrap(),
        linked_dir.to_string_lossy()
    );
    assert_eq!(linked["file_count"], 1);

    let folders_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/knowledge/python-contract-course/linked-folders")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(folders_response.status(), http::StatusCode::OK);
    let folders = json_response(folders_response).await;
    assert_eq!(folders.as_array().unwrap().len(), 1);
    assert_eq!(folders[0]["id"], folder_id);

    let sync_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri(format!(
                    "/api/v1/knowledge/python-contract-course/sync-folder/{folder_id}"
                ))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(sync_response.status(), http::StatusCode::OK);
    let sync = json_response(sync_response).await;
    assert_eq!(sync["file_count"], 1);
    assert!(sync["task_id"].as_str().unwrap().starts_with("kb_upload-"));

    let clear_progress_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/knowledge/python-contract-course/progress/clear")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(clear_progress_response.status(), http::StatusCode::OK);
    assert_eq!(
        json_response(clear_progress_response).await["message"],
        "Progress cleared for python-contract-course"
    );

    let unlink_response = app
        .oneshot(
            http::Request::builder()
                .method("DELETE")
                .uri(format!(
                    "/api/v1/knowledge/python-contract-course/linked-folders/{folder_id}"
                ))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(unlink_response.status(), http::StatusCode::OK);
    assert_eq!(
        json_response(unlink_response).await["message"],
        "Folder unlinked successfully"
    );

    let _ = std::fs::remove_dir_all(root);
}

#[tokio::test]
async fn course_knowledge_mutation_workflow_matches_frontend_contract() {
    let root = unique_test_knowledge_root();
    let app = app_with_knowledge_root(&root);
    let boundary = "SOCARTESBOUNDARY";
    let body = format!(
        "--{boundary}\r\n\
Content-Disposition: form-data; name=\"name\"\r\n\r\n\
test-course\r\n\
--{boundary}\r\n\
Content-Disposition: form-data; name=\"rag_provider\"\r\n\r\n\
llamaindex\r\n\
--{boundary}\r\n\
Content-Disposition: form-data; name=\"files\"; filename=\"notes.txt\"\r\n\
Content-Type: text/plain\r\n\r\n\
planner executor critic notes\r\n\
--{boundary}--\r\n"
    );

    let create_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/knowledge/create")
                .header(
                    http::header::CONTENT_TYPE,
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(create_response.status(), http::StatusCode::OK);
    let create_payload = json_response(create_response).await;
    assert!(
        create_payload["task_id"]
            .as_str()
            .unwrap()
            .starts_with("task-")
    );

    let list_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/knowledge/list")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let list_payload = json_response(list_response).await;
    assert!(
        list_payload["knowledge_bases"]
            .as_array()
            .unwrap()
            .iter()
            .any(|kb| kb["name"] == "test-course")
    );

    let set_default_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("PUT")
                .uri("/api/v1/knowledge/default/test-course")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(set_default_response.status(), http::StatusCode::OK);

    let reindex_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/knowledge/test-course/reindex")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(reindex_response.status(), http::StatusCode::OK);
    let reindex_payload = json_response(reindex_response).await;
    let task_id = reindex_payload["task_id"].as_str().unwrap();

    let stream_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri(format!("/api/v1/knowledge/tasks/{task_id}/stream"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(stream_response.status(), http::StatusCode::OK);
    assert_eq!(
        stream_response
            .headers()
            .get(http::header::CONTENT_TYPE)
            .unwrap(),
        "text/event-stream"
    );
    let stream_body = text_response(stream_response).await;
    assert!(stream_body.contains("event: complete"));

    let delete_response = app
        .oneshot(
            http::Request::builder()
                .method("DELETE")
                .uri("/api/v1/knowledge/test-course")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(delete_response.status(), http::StatusCode::OK);

    let _ = std::fs::remove_dir_all(root);
}

#[tokio::test]
async fn knowledge_reindex_creates_signature_version_and_reports_active_match() {
    let root = unique_test_knowledge_root();
    let app = app_with_knowledge_root(&root);
    let boundary = "SOCARTESINDEXBOUNDARY";
    let body = format!(
        "--{boundary}\r\n\
Content-Disposition: form-data; name=\"name\"\r\n\r\n\
index-course\r\n\
--{boundary}\r\n\
Content-Disposition: form-data; name=\"files\"; filename=\"notes.md\"\r\n\
Content-Type: text/markdown\r\n\r\n\
Index version notes for planner executor critic retrieval.\r\n\
--{boundary}--\r\n"
    );

    let create_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/knowledge/create")
                .header(
                    http::header::CONTENT_TYPE,
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(create_response.status(), http::StatusCode::OK);

    let reindex_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/knowledge/index-course/reindex")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(reindex_response.status(), http::StatusCode::OK);
    let reindex = json_response(reindex_response).await;
    let signature = reindex["signature"].as_str().expect("signature");
    assert_eq!(signature.len(), 16);
    assert_eq!(reindex["noop"], false);
    assert!(
        reindex["task_id"]
            .as_str()
            .unwrap()
            .starts_with("task-reindex-")
    );

    let list_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/knowledge/list")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(list_response.status(), http::StatusCode::OK);
    let list = json_response(list_response).await;
    let course = list["knowledge_bases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|kb| kb["name"] == "index-course")
        .expect("index-course");
    let stats = &course["statistics"];
    assert_eq!(stats["rag_initialized"], true);
    assert_eq!(stats["needs_reindex"], false);
    assert_eq!(stats["active_match"], true);
    assert_eq!(stats["active_signature"], signature);
    let version = &stats["index_versions"][0];
    assert_eq!(version["version"], "version-1");
    assert_eq!(version["signature"], signature);
    assert_eq!(version["layout"], "flat");
    assert_eq!(version["binding"], "rust-local");
    assert_eq!(version["model"], "deterministic-agent-loop");
    assert_eq!(version["dimension"], 0);
    assert_eq!(version["base_url"], "local://socartes-rust");
    assert_eq!(version["api_version"], "");
    assert_eq!(version["ready"], true);

    let noop_response = app
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/knowledge/index-course/reindex")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(noop_response.status(), http::StatusCode::OK);
    let noop = json_response(noop_response).await;
    assert_eq!(noop["signature"], signature);
    assert_eq!(noop["task_id"], Value::Null);
    assert_eq!(noop["noop"], true);

    let _ = std::fs::remove_dir_all(root);
}

#[tokio::test]
async fn knowledge_upload_rejects_kb_that_needs_reindex_like_python() {
    let root = unique_test_knowledge_root();
    let app = app_with_knowledge_root(&root);
    let boundary = "SOCARTESUPLOADREINDEXBOUNDARY";
    let body = format!(
        "--{boundary}\r\n\
Content-Disposition: form-data; name=\"name\"\r\n\r\n\
stale-course\r\n\
--{boundary}\r\n\
Content-Disposition: form-data; name=\"files\"; filename=\"notes.md\"\r\n\
Content-Type: text/markdown\r\n\r\n\
Original stale course material.\r\n\
--{boundary}--\r\n"
    );

    let create_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/knowledge/create")
                .header(
                    http::header::CONTENT_TYPE,
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(create_response.status(), http::StatusCode::OK);

    let config_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("PUT")
                .uri("/api/v1/knowledge/stale-course/config")
                .header("content-type", "application/json")
                .body(Body::from(json!({"needs_reindex": true}).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(config_response.status(), http::StatusCode::OK);

    let upload_boundary = "SOCARTESUPLOADBOUNDARY";
    let upload_body = format!(
        "--{upload_boundary}\r\n\
Content-Disposition: form-data; name=\"files\"; filename=\"extra.md\"\r\n\
Content-Type: text/markdown\r\n\r\n\
New material should wait for reindex.\r\n\
--{upload_boundary}--\r\n"
    );
    let upload_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/knowledge/stale-course/upload")
                .header(
                    http::header::CONTENT_TYPE,
                    format!("multipart/form-data; boundary={upload_boundary}"),
                )
                .body(Body::from(upload_body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(upload_response.status(), http::StatusCode::CONFLICT);
    assert!(
        json_response(upload_response).await["detail"]
            .as_str()
            .unwrap()
            .contains("needs reindex")
    );

    let reindex_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/knowledge/stale-course/reindex")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(reindex_response.status(), http::StatusCode::OK);
    assert_eq!(json_response(reindex_response).await["noop"], false);

    let config_get_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/knowledge/stale-course/config")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(config_get_response.status(), http::StatusCode::OK);
    assert_eq!(
        json_response(config_get_response).await["config"]["needs_reindex"],
        false
    );

    let upload_after_reindex_boundary = "SOCARTESUPLOADAFTERREINDEX";
    let upload_after_reindex_body = format!(
        "--{upload_after_reindex_boundary}\r\n\
Content-Disposition: form-data; name=\"files\"; filename=\"extra.md\"\r\n\
Content-Type: text/markdown\r\n\r\n\
New material can be uploaded after reindex.\r\n\
--{upload_after_reindex_boundary}--\r\n"
    );
    let upload_after_reindex_response = app
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/knowledge/stale-course/upload")
                .header(
                    http::header::CONTENT_TYPE,
                    format!("multipart/form-data; boundary={upload_after_reindex_boundary}"),
                )
                .body(Body::from(upload_after_reindex_body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(upload_after_reindex_response.status(), http::StatusCode::OK);

    let _ = std::fs::remove_dir_all(root);
}

#[tokio::test]
async fn chat_uses_selected_uploaded_course_files_as_rag_sources() {
    let root = unique_test_knowledge_root();
    let app = app_with_knowledge_root(&root);
    let boundary = "SOCARTESCHATBOUNDARY";
    let body = format!(
        "--{boundary}\r\n\
Content-Disposition: form-data; name=\"name\"\r\n\r\n\
plot-course\r\n\
--{boundary}\r\n\
Content-Disposition: form-data; name=\"rag_provider\"\r\n\r\n\
llamaindex\r\n\
--{boundary}\r\n\
Content-Disposition: form-data; name=\"files\"; filename=\"plot-notes.md\"\r\n\
Content-Type: text/markdown\r\n\r\n\
The lantern school rule says students must recite the blue theorem before opening the archive.\r\n\
--{boundary}--\r\n"
    );

    let create_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/knowledge/create")
                .header(
                    http::header::CONTENT_TYPE,
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(create_response.status(), http::StatusCode::OK);

    let chat_payload = json!({
        "type": "start_turn",
        "content": "What must students recite before opening the archive?",
        "language": "en",
        "tools": ["rag"],
        "knowledge_bases": ["plot-course"],
        "llm_selection": {
            "profile_id": "socartes-rust",
            "model_id": "deterministic-agent-loop"
        }
    });

    let chat_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/internal/test-chat-turn")
                .header("content-type", "application/json")
                .body(Body::from(chat_payload.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(chat_response.status(), http::StatusCode::OK);
    let chat_result = json_response(chat_response).await;
    let session_id = chat_result["session_id"].as_str().unwrap();

    let detail_response = app
        .oneshot(
            http::Request::builder()
                .uri(format!("/api/v1/sessions/{session_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(detail_response.status(), http::StatusCode::OK);
    let detail = json_response(detail_response).await;
    let assistant = detail["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|message| message["role"] == "assistant")
        .expect("assistant message");
    let sources_event = assistant["events"]
        .as_array()
        .unwrap()
        .iter()
        .find(|event| event["type"] == "sources")
        .expect("sources event");
    let sources = sources_event["metadata"]["sources"].as_array().unwrap();

    assert!(sources.iter().any(|source| {
        source["source_id"] == "plot-course/plot-notes.md"
            && source["content"]
                .as_str()
                .is_some_and(|content| content.contains("blue theorem"))
    }));
    let assistant_content = assistant["content"].as_str().unwrap();
    assert!(assistant_content.contains("plot-course/plot-notes.md"));
    assert!(assistant_content.contains("blue theorem"));

    let _ = std::fs::remove_dir_all(root);
}

#[tokio::test]
async fn rag_selected_uploaded_kb_returns_no_builtin_fallback_when_no_match() {
    let root = unique_test_knowledge_root();
    let app = app_with_knowledge_root(&root);
    let boundary = "SOCARTESNOFALLBACKBOUNDARY";
    let body = format!(
        "--{boundary}\r\n\
Content-Disposition: form-data; name=\"name\"\r\n\r\n\
plot-course\r\n\
--{boundary}\r\n\
Content-Disposition: form-data; name=\"files\"; filename=\"plot-notes.md\"\r\n\
Content-Type: text/markdown\r\n\r\n\
The lantern school rule says students must recite the blue theorem before opening the archive.\r\n\
--{boundary}--\r\n"
    );

    let create_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/knowledge/create")
                .header(
                    http::header::CONTENT_TYPE,
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(create_response.status(), http::StatusCode::OK);

    let chat_payload = json!({
        "type": "start_turn",
        "content": "How do ocean tides work?",
        "language": "en",
        "tools": ["rag"],
        "knowledge_bases": ["plot-course"]
    });
    let chat_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/internal/test-chat-turn")
                .header("content-type", "application/json")
                .body(Body::from(chat_payload.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(chat_response.status(), http::StatusCode::OK);
    let session_id = json_response(chat_response).await["session_id"]
        .as_str()
        .unwrap()
        .to_string();

    let detail_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri(format!("/api/v1/sessions/{session_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(detail_response.status(), http::StatusCode::OK);
    let detail = json_response(detail_response).await;
    let assistant = detail["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|message| message["role"] == "assistant")
        .expect("assistant message");
    let sources_event = assistant["events"]
        .as_array()
        .unwrap()
        .iter()
        .find(|event| event["type"] == "sources")
        .expect("sources event");
    let sources = sources_event["metadata"]["sources"].as_array().unwrap();
    assert!(sources.is_empty());
    let assistant_content = assistant["content"].as_str().unwrap();
    assert!(!assistant_content.contains("rag-index-18"));
    assert!(!assistant_content.contains("workflow-note-01"));

    let plugin_response = app
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/plugins/tools/rag/execute")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"params": {"query": "ocean tides", "kb_name": "plot-course"}})
                        .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(plugin_response.status(), http::StatusCode::OK);
    let plugin = json_response(plugin_response).await;
    assert_eq!(plugin["sources"], json!([]));
    assert!(
        plugin["content"]
            .as_str()
            .unwrap()
            .contains("No Socartes knowledge base passages matched query")
    );

    let _ = std::fs::remove_dir_all(root);
}

#[tokio::test]
async fn attachment_preview_route_serves_local_chat_files_like_python_contract() {
    let root = unique_test_knowledge_root();
    let app = app_with_knowledge_root(&root);
    let attachment_dir = test_data_root(&root)
        .join("user")
        .join("workspace")
        .join("chat")
        .join("attachments")
        .join("session-a");
    fs::create_dir_all(&attachment_dir).unwrap();
    fs::write(attachment_dir.join("att-1_diagram.png"), b"png-bytes").unwrap();
    fs::write(attachment_dir.join("att-2_notes.txt"), b"notes").unwrap();

    let image_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/attachments/session-a/att-1/diagram.png")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(image_response.status(), http::StatusCode::OK);
    assert_eq!(
        image_response
            .headers()
            .get(http::header::CONTENT_TYPE)
            .unwrap(),
        "image/png"
    );
    assert_eq!(
        image_response
            .headers()
            .get(http::header::CACHE_CONTROL)
            .unwrap(),
        "private, max-age=0, must-revalidate"
    );
    let disposition = image_response
        .headers()
        .get(http::header::CONTENT_DISPOSITION)
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    assert!(disposition.starts_with("inline;"));
    assert!(disposition.contains("filename=\"att-1_diagram.png\""));
    assert_eq!(text_response(image_response).await, "png-bytes");

    let alias_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/attachments/session-a/message-ignored/att-2/notes.txt")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(alias_response.status(), http::StatusCode::OK);
    assert_eq!(text_response(alias_response).await, "notes");

    let missing_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/attachments/session-a/att-404/missing.pdf")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(missing_response.status(), http::StatusCode::NOT_FOUND);
    assert_eq!(
        json_response(missing_response).await["detail"],
        "Attachment not found"
    );

    let traversal_response = app
        .oneshot(
            http::Request::builder()
                .uri("/api/attachments/../att-1/diagram.png")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(traversal_response.status(), http::StatusCode::NOT_FOUND);

    let _ = std::fs::remove_dir_all(test_data_root(&root));
}

#[tokio::test]
async fn memory_file_backed_api_matches_frontend_contract() {
    let root = unique_test_knowledge_root();
    let app = app_with_knowledge_root(&root);

    let empty_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/memory")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(empty_response.status(), http::StatusCode::OK);
    assert_eq!(
        json_response(empty_response).await,
        json!({
            "summary": "",
            "profile": "",
            "summary_updated_at": null,
            "profile_updated_at": null
        })
    );

    let summary_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("PUT")
                .uri("/api/v1/memory")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "file": "summary",
                        "content": "```markdown\n## Current Focus\nLearning Rust RAG.\n```"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(summary_response.status(), http::StatusCode::OK);
    let summary_payload = json_response(summary_response).await;
    assert_eq!(summary_payload["saved"], true);
    assert_eq!(
        summary_payload["summary"],
        "## Current Focus\nLearning Rust RAG."
    );
    assert!(summary_payload["summary_updated_at"].as_str().is_some());
    assert_eq!(
        fs::read_to_string(test_memory_root(&root).join("SUMMARY.md")).unwrap(),
        "## Current Focus\nLearning Rust RAG."
    );

    let profile_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("PUT")
                .uri("/api/v1/memory")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "file": "profile",
                        "content": "## Preferences\nPrefers cited course answers."
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(profile_response.status(), http::StatusCode::OK);
    let profile_payload = json_response(profile_response).await;
    assert_eq!(
        profile_payload["profile"],
        "## Preferences\nPrefers cited course answers."
    );
    assert!(profile_payload["profile_updated_at"].as_str().is_some());

    let invalid_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("PUT")
                .uri("/api/v1/memory")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"file": "other", "content": "bad"}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(invalid_response.status(), http::StatusCode::BAD_REQUEST);
    assert_eq!(
        json_response(invalid_response).await["detail"],
        "Invalid file: other"
    );

    let invalid_content_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("PUT")
                .uri("/api/v1/memory")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"file": "profile", "content": {"bad": true}}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        invalid_content_response.status(),
        http::StatusCode::UNPROCESSABLE_ENTITY
    );
    assert_eq!(
        fs::read_to_string(test_memory_root(&root).join("PROFILE.md")).unwrap(),
        "## Preferences\nPrefers cited course answers."
    );

    let clear_profile_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/memory/clear")
                .header("content-type", "application/json")
                .body(Body::from(json!({"file": "profile"}).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(clear_profile_response.status(), http::StatusCode::OK);
    let clear_profile_payload = json_response(clear_profile_response).await;
    assert_eq!(clear_profile_payload["cleared"], true);
    assert_eq!(
        clear_profile_payload["summary"],
        "## Current Focus\nLearning Rust RAG."
    );
    assert_eq!(clear_profile_payload["profile"], "");
    assert!(!test_memory_root(&root).join("PROFILE.md").exists());

    let missing_refresh_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/memory/refresh")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"session_id": "missing-session", "language": "en"}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        missing_refresh_response.status(),
        http::StatusCode::NOT_FOUND
    );
    assert_eq!(
        json_response(missing_refresh_response).await["detail"],
        "Session not found"
    );

    let empty_refresh_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/memory/refresh")
                .header("content-type", "application/json")
                .body(Body::from(json!({"language": "en"}).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(empty_refresh_response.status(), http::StatusCode::OK);
    assert_eq!(
        json_response(empty_refresh_response).await["changed"],
        false
    );

    let turn_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/internal/test-chat-turn")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "session_id": "memory-session",
                        "content": "Remember that Socartes memory should cite retrieved course files.",
                        "language": "en"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(turn_response.status(), http::StatusCode::OK);

    let refresh_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/memory/refresh")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"session_id": "memory-session", "language": "en"}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(refresh_response.status(), http::StatusCode::OK);
    let refresh_payload = json_response(refresh_response).await;
    assert_eq!(refresh_payload["changed"], true);
    assert!(
        refresh_payload["summary"]
            .as_str()
            .unwrap()
            .contains("## Current Focus")
    );
    assert!(
        refresh_payload["profile"]
            .as_str()
            .unwrap()
            .contains("## Preferences")
    );
    assert!(
        refresh_payload["summary"]
            .as_str()
            .unwrap()
            .contains("Learning Rust RAG")
    );
    assert!(
        refresh_payload["profile"]
            .as_str()
            .unwrap()
            .contains("Recent stable context came from session memory-session")
    );

    let thinking_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("PUT")
                .uri("/api/v1/memory")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "file": "summary",
                        "content": "## Current Focus\nVisible study note\n<think>private reasoning"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(thinking_response.status(), http::StatusCode::OK);
    let thinking_payload = json_response(thinking_response).await;
    assert_eq!(
        thinking_payload["summary"],
        "## Current Focus\nVisible study note"
    );

    let clear_all_response = app
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/memory/clear")
                .header("content-type", "application/json")
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(clear_all_response.status(), http::StatusCode::OK);
    let clear_all_payload = json_response(clear_all_response).await;
    assert_eq!(clear_all_payload["summary"], "");
    assert_eq!(clear_all_payload["profile"], "");
    assert!(!test_memory_root(&root).join("SUMMARY.md").exists());

    let _ = std::fs::remove_dir_all(test_data_root(&root));
}

#[tokio::test]
async fn memory_refresh_reports_corrupt_session_as_server_error() {
    let root = unique_test_knowledge_root();
    let app = app_with_knowledge_root(&root);
    let session_root = test_data_root(&root).join("sessions");
    fs::create_dir_all(&session_root).unwrap();
    fs::write(session_root.join("bad-session.json"), "{not json").unwrap();

    let response = app
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/memory/refresh")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"session_id": "bad-session", "language": "en"}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), http::StatusCode::INTERNAL_SERVER_ERROR);
    assert!(
        json_response(response).await["detail"]
            .as_str()
            .unwrap()
            .contains("Failed to parse session")
    );

    let _ = std::fs::remove_dir_all(test_data_root(&root));
}

#[tokio::test]
async fn skills_file_backed_crud_and_tags_match_frontend_contract() {
    let root = unique_test_knowledge_root();
    let app = app_with_knowledge_root(&root);

    let initial_tags_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/skills/tags/list")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(initial_tags_response.status(), http::StatusCode::OK);
    assert_eq!(
        json_response(initial_tags_response).await,
        json!({"tags": ["style", "tool"]})
    );

    let create_tag_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/skills/tags/create")
                .header("content-type", "application/json")
                .body(Body::from(json!({"name": "Workflow"}).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(create_tag_response.status(), http::StatusCode::OK);
    assert_eq!(json_response(create_tag_response).await["name"], "workflow");

    let create_skill_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/skills/create")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "name": "Course-Coach",
                        "description": "Coach course answers with citations",
                        "content": "Always cite uploaded course material.",
                        "tags": ["Workflow", "tool", "workflow"]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(create_skill_response.status(), http::StatusCode::OK);
    let created = json_response(create_skill_response).await;
    assert_eq!(created["name"], "course-coach");
    assert_eq!(
        created["description"],
        "Coach course answers with citations"
    );
    assert_eq!(created["tags"], json!(["workflow", "tool"]));

    let skill_file = test_skills_root(&root)
        .join("course-coach")
        .join("SKILL.md");
    let saved = fs::read_to_string(skill_file).expect("saved skill");
    assert!(saved.contains("name: course-coach"));
    assert!(saved.contains("description: Coach course answers with citations"));
    assert!(saved.contains("Always cite uploaded course material."));

    let list_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/skills/list")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(list_response.status(), http::StatusCode::OK);
    let list = json_response(list_response).await;
    assert_eq!(list["skills"].as_array().unwrap().len(), 1);
    assert_eq!(list["skills"][0]["name"], "course-coach");
    assert!(list["skills"][0].get("content").is_none());

    let detail_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/skills/course-coach")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(detail_response.status(), http::StatusCode::OK);
    let detail = json_response(detail_response).await;
    assert_eq!(detail["name"], "course-coach");
    assert!(
        detail["content"]
            .as_str()
            .unwrap()
            .contains("Always cite uploaded course material.")
    );

    let rename_tag_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("PUT")
                .uri("/api/v1/skills/tags/workflow")
                .header("content-type", "application/json")
                .body(Body::from(json!({"rename_to": "Study Flow"}).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(rename_tag_response.status(), http::StatusCode::OK);
    assert_eq!(
        json_response(rename_tag_response).await["name"],
        "study flow"
    );

    let update_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("PUT")
                .uri("/api/v1/skills/course-coach")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "rename_to": "citation-coach",
                        "description": "Updated citation coach",
                        "content": "---\nname: old\ntriggers:\n- cite\n---\n\nUse course evidence.",
                        "tags": ["Study Flow"]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(update_response.status(), http::StatusCode::OK);
    let updated = json_response(update_response).await;
    assert_eq!(updated["name"], "citation-coach");
    assert_eq!(updated["description"], "Updated citation coach");
    assert_eq!(updated["tags"], json!(["study flow"]));

    let old_detail_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/skills/course-coach")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(old_detail_response.status(), http::StatusCode::NOT_FOUND);

    let new_detail_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/skills/citation-coach")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(new_detail_response.status(), http::StatusCode::OK);
    let new_detail = json_response(new_detail_response).await;
    assert!(
        new_detail["content"]
            .as_str()
            .unwrap()
            .contains("name: citation-coach")
    );
    assert!(
        new_detail["content"]
            .as_str()
            .unwrap()
            .contains("Use course evidence.")
    );

    let delete_tag_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("DELETE")
                .uri("/api/v1/skills/tags/study%20flow")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(delete_tag_response.status(), http::StatusCode::OK);
    assert_eq!(
        json_response(delete_tag_response).await,
        json!({"status": "deleted", "name": "study flow"})
    );

    let untagged_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/skills/citation-coach")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(untagged_response.status(), http::StatusCode::OK);
    assert_eq!(json_response(untagged_response).await["tags"], json!([]));

    let delete_skill_response = app
        .oneshot(
            http::Request::builder()
                .method("DELETE")
                .uri("/api/v1/skills/citation-coach")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(delete_skill_response.status(), http::StatusCode::OK);
    assert_eq!(
        json_response(delete_skill_response).await,
        json!({"status": "deleted", "name": "citation-coach"})
    );

    let _ = std::fs::remove_dir_all(test_data_root(&root));
}

#[tokio::test]
async fn skills_errors_match_python_status_codes() {
    let root = unique_test_knowledge_root();
    let app = app_with_knowledge_root(&root);

    let invalid_skill_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/skills/create")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "name": "../bad",
                        "description": "",
                        "content": ""
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        invalid_skill_response.status(),
        http::StatusCode::BAD_REQUEST
    );

    let create_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/skills/create")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "name": "dupe-skill",
                        "description": "First",
                        "content": "First body"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(create_response.status(), http::StatusCode::OK);

    let duplicate_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/skills/create")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "name": "dupe-skill",
                        "description": "Second",
                        "content": "Second body"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(duplicate_response.status(), http::StatusCode::CONFLICT);

    let missing_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/skills/missing-skill")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(missing_response.status(), http::StatusCode::NOT_FOUND);

    let invalid_tag_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/skills/tags/create")
                .header("content-type", "application/json")
                .body(Body::from(json!({"name": "$bad"}).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(invalid_tag_response.status(), http::StatusCode::BAD_REQUEST);

    let missing_tag_response = app
        .oneshot(
            http::Request::builder()
                .method("DELETE")
                .uri("/api/v1/skills/tags/not-there")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(missing_tag_response.status(), http::StatusCode::NOT_FOUND);

    let _ = std::fs::remove_dir_all(test_data_root(&root));
}

#[tokio::test]
async fn skills_frontmatter_preserves_yaml_sensitive_scalars() {
    let root = unique_test_knowledge_root();
    let app = app_with_knowledge_root(&root);
    let description = "Coach: cite #1\nsecond line with \"quotes\"";

    let create_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/skills/create")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "name": "yaml-coach",
                        "description": description,
                        "content": "Use the evidence exactly.",
                        "tags": ["style"]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(create_response.status(), http::StatusCode::OK);

    let detail_response = app
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/skills/yaml-coach")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(detail_response.status(), http::StatusCode::OK);
    let detail = json_response(detail_response).await;
    assert_eq!(detail["description"], description);

    let saved = fs::read_to_string(test_skills_root(&root).join("yaml-coach").join("SKILL.md"))
        .expect("saved skill");
    assert!(saved.contains("description: \"Coach: cite #1\\nsecond line with \\\"quotes\\\"\""));

    let _ = std::fs::remove_dir_all(test_data_root(&root));
}

#[cfg(unix)]
#[tokio::test]
async fn skills_api_rejects_symlink_directory_escape() {
    use std::os::unix::fs as unix_fs;

    let root = unique_test_knowledge_root();
    let app = app_with_knowledge_root(&root);
    let skills_root = test_skills_root(&root);
    let outside = root.with_file_name(format!(
        "socartes-skills-outside-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir_all(&skills_root).unwrap();
    fs::create_dir_all(&outside).unwrap();
    fs::write(
        outside.join("SKILL.md"),
        "---\nname: evil-link\ndescription: outside\n---\n\nDo not expose.",
    )
    .unwrap();
    unix_fs::symlink(&outside, skills_root.join("evil-link")).unwrap();

    let list_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/skills/list")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(list_response.status(), http::StatusCode::OK);
    assert_eq!(json_response(list_response).await["skills"], json!([]));

    let get_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/skills/evil-link")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(get_response.status(), http::StatusCode::NOT_FOUND);

    let update_response = app
        .oneshot(
            http::Request::builder()
                .method("PUT")
                .uri("/api/v1/skills/evil-link")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"description": "mutated outside"}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(update_response.status(), http::StatusCode::NOT_FOUND);
    let outside_content = fs::read_to_string(outside.join("SKILL.md")).unwrap();
    assert!(outside_content.contains("description: outside"));

    let _ = std::fs::remove_file(skills_root.join("evil-link"));
    let _ = std::fs::remove_dir_all(test_data_root(&root));
    let _ = std::fs::remove_dir_all(outside);
}

#[cfg(unix)]
#[tokio::test]
async fn skills_tag_vocab_stays_unchanged_when_cascade_rewrite_fails() {
    use std::os::unix::fs::PermissionsExt;

    let root = unique_test_knowledge_root();
    let app = app_with_knowledge_root(&root);

    for name in ["alpha-skill", "zeta-skill"] {
        let response = app
            .clone()
            .oneshot(
                http::Request::builder()
                    .method("POST")
                    .uri("/api/v1/skills/create")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "name": name,
                            "description": "Tag cascade fixture",
                            "content": "Body",
                            "tags": ["workflow"]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), http::StatusCode::OK);
    }

    let locked_skill = test_skills_root(&root).join("alpha-skill").join("SKILL.md");
    let mut permissions = fs::metadata(&locked_skill).unwrap().permissions();
    permissions.set_mode(0o444);
    fs::set_permissions(&locked_skill, permissions).unwrap();

    let rename_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("PUT")
                .uri("/api/v1/skills/tags/workflow")
                .header("content-type", "application/json")
                .body(Body::from(json!({"rename_to": "study flow"}).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        rename_response.status(),
        http::StatusCode::INTERNAL_SERVER_ERROR
    );

    let tags_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/skills/tags/list")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let tags = json_response(tags_response).await["tags"].clone();
    assert!(tags.as_array().unwrap().contains(&json!("workflow")));
    assert!(!tags.as_array().unwrap().contains(&json!("study flow")));
    assert!(
        fs::read_to_string(test_skills_root(&root).join("zeta-skill").join("SKILL.md"))
            .unwrap()
            .contains("- workflow")
    );

    let mut permissions = fs::metadata(&locked_skill).unwrap().permissions();
    permissions.set_mode(0o644);
    fs::set_permissions(&locked_skill, permissions).unwrap();
    let _ = std::fs::remove_dir_all(test_data_root(&root));
}

#[tokio::test]
async fn plugins_list_matches_playground_contract() {
    let response = app()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/plugins/list")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), http::StatusCode::OK);
    let payload = json_response(response).await;
    let tools = payload["tools"].as_array().expect("tools array");
    assert!(tools.iter().any(|tool| {
        tool["name"] == "rag"
            && tool["parameters"]
                .as_array()
                .unwrap()
                .iter()
                .any(|param| param["name"] == "kb_name" && param["required"] == false)
    }));
    assert!(tools.iter().any(|tool| tool["name"] == "code_execution"));

    let capabilities = payload["capabilities"]
        .as_array()
        .expect("capabilities array");
    assert!(capabilities.iter().any(|capability| {
        capability["name"] == "chat"
            && capability["stages"] == json!(["thinking", "acting", "observing", "responding"])
            && capability["tools_used"]
                .as_array()
                .unwrap()
                .contains(&json!("rag"))
            && capability["request_schema"]["title"] == "ChatRequestConfig"
    }));
    assert!(
        capabilities
            .iter()
            .any(|capability| capability["name"] == "deep_research")
    );
    assert_eq!(payload["plugins"], json!([]));
}

#[tokio::test]
async fn plugins_tool_execute_and_stream_match_python_shapes() {
    let direct_response = app()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/plugins/tools/brainstorm/execute")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"params": {"topic": "course planning", "context": "short"}}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(direct_response.status(), http::StatusCode::OK);
    let direct = json_response(direct_response).await;
    assert_eq!(direct["success"], true);
    assert!(
        direct["content"]
            .as_str()
            .unwrap()
            .contains("course planning")
    );
    assert!(direct["sources"].is_array());
    assert!(direct["metadata"].is_object());

    let stream_response = app()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/plugins/tools/rag/execute-stream")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"params": {"query": "blue theorem", "kb_name": "socartes-rust-rag"}})
                        .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(stream_response.status(), http::StatusCode::OK);
    assert_eq!(
        stream_response
            .headers()
            .get(http::header::CONTENT_TYPE)
            .unwrap(),
        "text/event-stream"
    );
    let stream = text_response(stream_response).await;
    assert!(stream.contains("event: process_log"));
    assert!(stream.contains("event: result"));
    assert!(stream.contains("\"success\":true"));
    assert!(stream.contains("\"elapsed_ms\""));
}

#[tokio::test]
async fn plugins_capability_stream_matches_playground_contract() {
    let response = app()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/plugins/capabilities/chat/execute-stream")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "content": "Explain RAG briefly",
                        "tools": ["rag"],
                        "knowledge_bases": ["socartes-rust-rag"],
                        "language": "en",
                        "config": {},
                        "attachments": []
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), http::StatusCode::OK);
    assert_eq!(
        response.headers().get(http::header::CONTENT_TYPE).unwrap(),
        "text/event-stream"
    );
    let stream = text_response(response).await;
    assert!(stream.contains("event: process_log"));
    assert!(stream.contains("event: stream"));
    assert!(stream.contains("\"type\":\"content\""));
    assert!(stream.contains("event: result"));
    assert!(stream.contains("\"success\":true"));
    assert!(stream.contains("\"data\""));
    assert!(stream.contains("\"elapsed_ms\""));
}

#[tokio::test]
async fn page_agent_chat_completion_returns_agent_output_tool_call() {
    let response = app()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/page-agent/openai/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "gpt-5.5",
                        "messages": [
                            {"role": "system", "content": "You are the Socartes page agent."},
                            {"role": "user", "content": "What can you do on this page?"}
                        ],
                        "tools": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "AgentOutput",
                                    "parameters": {
                                        "type": "object",
                                        "required": ["type"],
                                        "properties": {
                                            "type": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        ],
                        "tool_choice": {"type": "function", "function": {"name": "AgentOutput"}},
                        "temperature": 0.7,
                        "extra_frontend_field": "allowed"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), http::StatusCode::OK);
    let payload = json_response(response).await;
    assert!(
        payload["id"]
            .as_str()
            .unwrap()
            .starts_with("chatcmpl-page-agent-")
    );
    assert_eq!(payload["object"], "chat.completion");
    assert_eq!(payload["model"], "gpt-5.5");
    assert!(payload["created"].is_number());
    assert_eq!(payload["choices"][0]["index"], 0);
    assert_eq!(payload["choices"][0]["finish_reason"], "tool_calls");

    let message = &payload["choices"][0]["message"];
    assert_eq!(message["role"], "assistant");
    assert_eq!(message["content"], "");

    let tool_call = &message["tool_calls"][0];
    assert!(tool_call["id"].as_str().unwrap().starts_with("call_"));
    assert_eq!(tool_call["type"], "function");
    assert_eq!(tool_call["function"]["name"], "AgentOutput");

    let arguments = tool_call["function"]["arguments"]
        .as_str()
        .expect("AgentOutput arguments must be a JSON string");
    let parsed_arguments: Value =
        serde_json::from_str(arguments).expect("AgentOutput arguments JSON");
    assert_eq!(parsed_arguments["type"], "done");
    assert!(
        parsed_arguments["message"]
            .as_str()
            .unwrap()
            .contains("Page agent")
    );
    assert_eq!(payload["usage"]["total_tokens"], 0);
}

#[tokio::test]
async fn page_agent_chat_completion_requires_messages_array() {
    let response = app()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/page-agent/openai/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(json!({"model": "gpt-5.5"}).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), http::StatusCode::UNPROCESSABLE_ENTITY);
    let payload = json_response(response).await;
    assert!(payload["detail"].as_str().unwrap().contains("messages"));
}

#[tokio::test]
async fn co_writer_documents_crud_matches_frontend_contract() {
    let root = unique_test_knowledge_root();
    let app = app_with_knowledge_root(&root);

    let empty_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/co_writer/documents")
                .header("content-type", "application/json")
                .body(Body::from(json!({}).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(empty_response.status(), http::StatusCode::OK);
    let empty_doc = json_response(empty_response).await;
    let empty_id = empty_doc["id"].as_str().unwrap();
    assert_eq!(empty_id.len(), 12);
    assert!(empty_id.chars().all(|ch| ch.is_ascii_hexdigit()));
    assert_eq!(empty_doc["title"], "Untitled draft");
    assert_eq!(empty_doc["content"], "");
    assert!(empty_doc["created_at"].is_number());
    assert!(empty_doc["updated_at"].is_number());

    let derived_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/co_writer/documents")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "content": "# Derived Title\n\nFirst paragraph.\nSecond paragraph."
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(derived_response.status(), http::StatusCode::OK);
    let derived_doc = json_response(derived_response).await;
    let derived_id = derived_doc["id"].as_str().unwrap();
    assert_eq!(derived_doc["title"], "Derived Title");

    let get_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri(format!("/api/v1/co_writer/documents/{derived_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(get_response.status(), http::StatusCode::OK);
    assert_eq!(json_response(get_response).await["id"], derived_id);

    let update_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("PUT")
                .uri(format!("/api/v1/co_writer/documents/{empty_id}"))
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "title": null,
                        "content": "# Long Title\n\nAlpha\nBeta"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(update_response.status(), http::StatusCode::OK);
    let updated = json_response(update_response).await;
    assert_eq!(updated["title"], "Long Title");

    let list_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/co_writer/documents")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(list_response.status(), http::StatusCode::OK);
    let list = json_response(list_response).await;
    let documents = list["documents"].as_array().unwrap();
    assert_eq!(documents.len(), 2);
    assert_eq!(documents[0]["id"], empty_id);
    assert!(
        documents[0]["preview"]
            .as_str()
            .unwrap()
            .contains("# Long Title  Alpha  Beta")
    );

    let manifest = test_co_writer_docs_root(&root)
        .join(format!("doc_{empty_id}"))
        .join("manifest.json");
    assert!(manifest.exists());

    let missing_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/co_writer/documents/missingdoc999")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(missing_response.status(), http::StatusCode::NOT_FOUND);
    assert_eq!(
        json_response(missing_response).await["detail"],
        "Document not found"
    );

    let delete_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("DELETE")
                .uri(format!("/api/v1/co_writer/documents/{derived_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(delete_response.status(), http::StatusCode::OK);
    assert_eq!(json_response(delete_response).await["deleted"], true);

    let deleted_response = app
        .oneshot(
            http::Request::builder()
                .uri(format!("/api/v1/co_writer/documents/{derived_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(deleted_response.status(), http::StatusCode::NOT_FOUND);

    let _ = std::fs::remove_dir_all(test_data_root(&root));
}

#[tokio::test]
async fn co_writer_edit_automark_and_stream_match_frontend_contract() {
    let root = unique_test_knowledge_root();
    let app = app_with_knowledge_root(&root);

    let edit_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/co_writer/edit")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "text": "Original paragraph with extra detail.",
                        "instruction": "make it concise",
                        "action": "shorten",
                        "source": null,
                        "kb_name": null
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(edit_response.status(), http::StatusCode::OK);
    let edit_payload = json_response(edit_response).await;
    assert!(edit_payload["operation_id"].as_str().unwrap().len() > 8);
    assert!(
        edit_payload["edited_text"]
            .as_str()
            .unwrap()
            .contains("Original paragraph")
    );

    let automark_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/co_writer/automark")
                .header("content-type", "application/json")
                .body(Body::from(json!({"text": "Key concept"}).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(automark_response.status(), http::StatusCode::OK);
    let automark_payload = json_response(automark_response).await;
    assert!(automark_payload["operation_id"].as_str().unwrap().len() > 8);
    assert_eq!(automark_payload["marked_text"], "==Key concept==");

    let stream_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/co_writer/edit_react/stream")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "selected_text": "Original sentence",
                        "instruction": "make it clearer",
                        "mode": "rewrite",
                        "tools": ["brainstorm", "not-a-tool"],
                        "kb_name": null
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(stream_response.status(), http::StatusCode::OK);
    assert_eq!(
        stream_response
            .headers()
            .get(http::header::CONTENT_TYPE)
            .unwrap(),
        "text/event-stream"
    );
    let stream_body = text_response(stream_response).await;
    assert!(stream_body.contains("event: stream"));
    assert!(stream_body.contains("\"type\":\"thinking\""));
    assert!(stream_body.contains("\"type\":\"content\""));
    assert!(stream_body.contains("\"stage\":\"responding\""));
    assert!(stream_body.contains("event: result"));
    assert!(stream_body.contains("\"edited_text\""));
    assert!(stream_body.contains("\"tool_traces\""));

    let empty_selection_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/co_writer/edit_react/stream")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "selected_text": "   ",
                        "instruction": "rewrite",
                        "mode": "rewrite",
                        "tools": []
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        empty_selection_response.status(),
        http::StatusCode::BAD_REQUEST
    );
    assert_eq!(
        json_response(empty_selection_response).await["detail"],
        "Please select a text passage first."
    );

    let missing_instruction_response = app
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/co_writer/edit_react/stream")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "selected_text": "Original sentence",
                        "instruction": "",
                        "mode": "none",
                        "tools": []
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        missing_instruction_response.status(),
        http::StatusCode::BAD_REQUEST
    );
    assert!(
        json_response(missing_instruction_response).await["detail"]
            .as_str()
            .unwrap()
            .contains("Provide an edit instruction")
    );

    let _ = std::fs::remove_dir_all(test_data_root(&root));
}

#[tokio::test]
async fn co_writer_history_tool_calls_export_and_non_stream_react_match_python_contract() {
    let root = unique_test_knowledge_root();
    let app = app_with_knowledge_root(&root);

    let empty_history_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/co_writer/history")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(empty_history_response.status(), http::StatusCode::OK);
    assert_eq!(
        json_response(empty_history_response).await,
        json!({"history": [], "total": 0})
    );

    let react_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/co_writer/edit_react")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "selected_text": "Draft sentence",
                        "instruction": "make it stronger",
                        "mode": "rewrite",
                        "tools": ["brainstorm", "not-a-tool"],
                        "kb_name": "course-notes"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(react_response.status(), http::StatusCode::OK);
    let react_payload = json_response(react_response).await;
    let operation_id = react_payload["operation_id"].as_str().unwrap();
    assert!(operation_id.len() > 8);
    assert!(
        react_payload["edited_text"]
            .as_str()
            .unwrap()
            .contains("Draft sentence")
    );
    assert!(
        react_payload["thinking"]
            .as_str()
            .unwrap()
            .contains("rewrite")
    );
    assert_eq!(react_payload["tool_traces"].as_array().unwrap().len(), 1);
    assert_eq!(react_payload["tool_traces"][0]["name"], "brainstorm");

    let history_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/co_writer/history")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(history_response.status(), http::StatusCode::OK);
    let history = json_response(history_response).await;
    assert_eq!(history["total"], 1);
    assert_eq!(history["history"][0]["id"], operation_id);
    assert_eq!(history["history"][0]["action"], "react_edit");
    assert_eq!(history["history"][0]["mode"], "rewrite");

    let operation_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri(format!("/api/v1/co_writer/history/{operation_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(operation_response.status(), http::StatusCode::OK);
    let operation = json_response(operation_response).await;
    assert_eq!(operation["id"], operation_id);
    assert_eq!(operation["input"]["selected_text"], "Draft sentence");
    assert_eq!(
        operation["output"]["edited_text"],
        react_payload["edited_text"]
    );

    let tool_call_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri(format!("/api/v1/co_writer/tool_calls/{operation_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(tool_call_response.status(), http::StatusCode::OK);
    let tool_call = json_response(tool_call_response).await;
    assert_eq!(tool_call["type"], "react_tools");
    assert_eq!(tool_call["operation_id"], operation_id);
    assert_eq!(tool_call["tools"], json!(["brainstorm"]));
    assert_eq!(tool_call["tool_traces"][0]["name"], "brainstorm");

    let missing_operation_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/co_writer/history/missing-operation")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        missing_operation_response.status(),
        http::StatusCode::NOT_FOUND
    );
    assert_eq!(
        json_response(missing_operation_response).await["detail"],
        "Operation not found"
    );

    let missing_tool_call_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/co_writer/tool_calls/missing-operation")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        missing_tool_call_response.status(),
        http::StatusCode::NOT_FOUND
    );
    assert_eq!(
        json_response(missing_tool_call_response).await["detail"],
        "Tool call not found"
    );

    let export_response = app
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/co_writer/export/markdown")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "content": "# Exported\nBody",
                        "filename": "draft.md"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(export_response.status(), http::StatusCode::OK);
    assert_eq!(
        export_response
            .headers()
            .get(http::header::CONTENT_TYPE)
            .unwrap(),
        "text/markdown"
    );
    assert_eq!(
        export_response
            .headers()
            .get(http::header::CONTENT_DISPOSITION)
            .unwrap(),
        "attachment; filename=draft.md"
    );
    assert_eq!(text_response(export_response).await, "# Exported\nBody");

    let _ = std::fs::remove_dir_all(test_data_root(&root));
}

#[tokio::test]
async fn tutorbot_management_profiles_and_souls_match_frontend_contract() {
    let root = unique_test_knowledge_root();
    let app = app_with_knowledge_root(&root);

    let empty_list = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/tutorbot")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(empty_list.status(), http::StatusCode::OK);
    assert_eq!(json_response(empty_list).await, json!([]));

    let souls_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/tutorbot/souls")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(souls_response.status(), http::StatusCode::OK);
    let souls = json_response(souls_response).await;
    assert!(souls.as_array().unwrap().len() >= 3);
    assert!(souls.as_array().unwrap().iter().any(|soul| {
        soul["id"] == "default-tutorbot" && soul["content"].as_str().unwrap().contains("# Soul")
    }));

    let create_soul_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/tutorbot/souls")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "id": "exam-coach",
                        "name": "Exam Coach",
                        "content": "# Soul\n\nCoach for exams."
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(create_soul_response.status(), http::StatusCode::OK);
    assert_eq!(
        json_response(create_soul_response).await["id"],
        "exam-coach"
    );

    let duplicate_soul_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/tutorbot/souls")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"id": "exam-coach", "name": "Exam Coach", "content": "again"})
                        .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(duplicate_soul_response.status(), http::StatusCode::CONFLICT);

    let update_soul_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("PUT")
                .uri("/api/v1/tutorbot/souls/exam-coach")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"name": "Exam Coach Plus", "content": "# Soul\n\nUpdated."}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(update_soul_response.status(), http::StatusCode::OK);
    assert_eq!(
        json_response(update_soul_response).await["name"],
        "Exam Coach Plus"
    );

    let schema_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/tutorbot/channels/schema")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(schema_response.status(), http::StatusCode::OK);
    let schema = json_response(schema_response).await;
    assert!(schema["channels"].is_object());
    assert!(schema["global"]["json_schema"].is_object());

    let create_bot_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/tutorbot")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "bot_id": "exam-bot",
                        "name": "Exam Bot",
                        "description": "Exam support",
                        "persona": "# Soul\n\nFocus on evidence.",
                        "channels": {
                            "send_progress": true,
                            "telegram": {"enabled": true, "token": "secret-token"}
                        },
                        "llm_selection": {"profile_id": "socartes-rust", "model_id": "deterministic-agent-loop"}
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(create_bot_response.status(), http::StatusCode::OK);
    let bot = json_response(create_bot_response).await;
    assert_eq!(bot["bot_id"], "exam-bot");
    assert_eq!(bot["running"], true);
    assert_eq!(bot["channels"]["telegram"]["token"], "***");

    let list_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/tutorbot")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(list_response.status(), http::StatusCode::OK);
    let list = json_response(list_response).await;
    assert_eq!(list[0]["bot_id"], "exam-bot");
    assert!(
        list[0]["channels"]
            .as_array()
            .unwrap()
            .contains(&json!("telegram"))
    );

    let detail_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/tutorbot/exam-bot")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(detail_response.status(), http::StatusCode::OK);
    let detail = json_response(detail_response).await;
    assert_eq!(detail["channels"]["telegram"]["token"], "***");

    let files_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/tutorbot/exam-bot/files")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(files_response.status(), http::StatusCode::OK);
    let files = json_response(files_response).await;
    assert!(
        files["SOUL.md"]
            .as_str()
            .unwrap()
            .contains("Focus on evidence")
    );
    assert!(files["USER.md"].is_string());
    assert!(files["TOOLS.md"].is_string());
    assert!(files["AGENTS.md"].is_string());
    assert!(files["HEARTBEAT.md"].is_string());

    let save_file_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("PUT")
                .uri("/api/v1/tutorbot/exam-bot/files/USER.md")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"content": "Learner likes short answers."}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(save_file_response.status(), http::StatusCode::OK);
    assert_eq!(json_response(save_file_response).await["saved"], true);

    let read_file_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/tutorbot/exam-bot/files/USER.md")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(read_file_response.status(), http::StatusCode::OK);
    assert_eq!(
        json_response(read_file_response).await["content"],
        "Learner likes short answers."
    );

    let patch_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("PATCH")
                .uri("/api/v1/tutorbot/exam-bot")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "persona": "# Soul\n\nUpdated persona.",
                        "channels": {
                            "send_progress": true,
                            "send_tool_hints": true,
                            "telegram": {"enabled": true, "token": "***"}
                        },
                        "model": "local-test-model"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(patch_response.status(), http::StatusCode::OK);
    let patched = json_response(patch_response).await;
    assert_eq!(patched["persona"], "# Soul\n\nUpdated persona.");
    assert_eq!(patched["channels"]["telegram"]["token"], "***");
    assert_eq!(patched["model"], "local-test-model");

    let history_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/tutorbot/exam-bot/history")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(history_response.status(), http::StatusCode::OK);
    assert_eq!(json_response(history_response).await, json!([]));

    let stop_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("DELETE")
                .uri("/api/v1/tutorbot/exam-bot")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(stop_response.status(), http::StatusCode::OK);
    assert_eq!(json_response(stop_response).await["stopped"], true);

    let restart_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/tutorbot")
                .header("content-type", "application/json")
                .body(Body::from(json!({"bot_id": "exam-bot"}).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(restart_response.status(), http::StatusCode::OK);
    assert_eq!(json_response(restart_response).await["running"], true);

    let recent_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/tutorbot/recent?limit=3")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(recent_response.status(), http::StatusCode::OK);
    assert_eq!(json_response(recent_response).await, json!([]));

    let destroy_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("DELETE")
                .uri("/api/v1/tutorbot/exam-bot/destroy")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(destroy_response.status(), http::StatusCode::OK);
    assert_eq!(json_response(destroy_response).await["destroyed"], true);

    let deleted_soul_response = app
        .oneshot(
            http::Request::builder()
                .method("DELETE")
                .uri("/api/v1/tutorbot/souls/exam-coach")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(deleted_soul_response.status(), http::StatusCode::OK);
    assert_eq!(json_response(deleted_soul_response).await["deleted"], true);

    let _ = std::fs::remove_dir_all(test_data_root(&root));
}

#[tokio::test]
async fn notebook_crud_and_streamed_save_match_frontend_contract() {
    let root = unique_test_knowledge_root();
    let app = app_with_knowledge_root(&root);

    let create_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/notebook/create")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "name": "RAG Notes",
                        "description": "Saved chat outputs",
                        "color": "#22C55E",
                        "icon": "notebook"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(create_response.status(), http::StatusCode::OK);
    let create_payload = json_response(create_response).await;
    assert_eq!(create_payload["success"], true);
    let notebook_id = create_payload["notebook"]["id"].as_str().unwrap();
    assert_eq!(create_payload["notebook"]["name"], "RAG Notes");

    let list_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/notebook/list")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(list_response.status(), http::StatusCode::OK);
    let list_payload = json_response(list_response).await;
    assert_eq!(list_payload["total"], 1);
    assert_eq!(list_payload["notebooks"][0]["id"], notebook_id);
    assert_eq!(list_payload["notebooks"][0]["record_count"], 0);

    let save_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/notebook/add_record_with_summary")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "notebook_ids": [notebook_id],
                        "record_type": "chat",
                        "title": "Course answer",
                        "summary": "A concise saved summary.",
                        "user_query": "What did the course say?",
                        "output": "The course evidence came from the uploaded notes.",
                        "metadata": { "session_id": "chat-session-1", "ui_language": "en" },
                        "kb_name": "course-live-check"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(save_response.status(), http::StatusCode::OK);
    assert_eq!(
        save_response
            .headers()
            .get(http::header::CONTENT_TYPE)
            .unwrap(),
        "text/event-stream"
    );
    let save_body = text_response(save_response).await;
    assert!(save_body.contains("\"type\":\"summary_chunk\""));
    assert!(save_body.contains("\"type\":\"result\""));
    assert!(save_body.contains("A concise saved summary."));

    let detail_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri(format!("/api/v1/notebook/{notebook_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(detail_response.status(), http::StatusCode::OK);
    let detail = json_response(detail_response).await;
    assert_eq!(detail["records"].as_array().unwrap().len(), 1);
    let record_id = detail["records"][0]["id"].as_str().unwrap();
    assert_eq!(detail["records"][0]["type"], "chat");
    assert_eq!(detail["records"][0]["summary"], "A concise saved summary.");
    assert_eq!(
        detail["records"][0]["metadata"]["session_id"],
        "chat-session-1"
    );
    assert_eq!(detail["records"][0]["kb_name"], "course-live-check");

    let update_record_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("PUT")
                .uri(format!(
                    "/api/v1/notebook/{notebook_id}/records/{record_id}"
                ))
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "title": "Updated course answer",
                        "summary": "Updated summary",
                        "metadata": { "edited": true },
                        "kb_name": null
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(update_record_response.status(), http::StatusCode::OK);
    let update_record_payload = json_response(update_record_response).await;
    assert_eq!(update_record_payload["success"], true);
    assert_eq!(
        update_record_payload["record"]["title"],
        "Updated course answer"
    );
    assert_eq!(
        update_record_payload["record"]["metadata"]["session_id"],
        "chat-session-1"
    );
    assert_eq!(update_record_payload["record"]["metadata"]["edited"], true);
    assert_eq!(update_record_payload["record"]["kb_name"], Value::Null);

    let statistics_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/notebook/statistics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(statistics_response.status(), http::StatusCode::OK);
    let statistics = json_response(statistics_response).await;
    assert_eq!(statistics["total_notebooks"], 1);
    assert_eq!(statistics["total_records"], 1);
    assert_eq!(statistics["records_by_type"]["chat"], 1);

    let remove_record_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("DELETE")
                .uri(format!(
                    "/api/v1/notebook/{notebook_id}/records/{record_id}"
                ))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(remove_record_response.status(), http::StatusCode::OK);
    assert_eq!(json_response(remove_record_response).await["success"], true);

    let update_notebook_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("PUT")
                .uri(format!("/api/v1/notebook/{notebook_id}"))
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"name": "Updated RAG Notes", "color": "#6366F1"}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(update_notebook_response.status(), http::StatusCode::OK);
    assert_eq!(
        json_response(update_notebook_response).await["notebook"]["name"],
        "Updated RAG Notes"
    );

    let delete_notebook_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("DELETE")
                .uri(format!("/api/v1/notebook/{notebook_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(delete_notebook_response.status(), http::StatusCode::OK);
    assert_eq!(
        json_response(delete_notebook_response).await["success"],
        true
    );

    let missing_response = app
        .oneshot(
            http::Request::builder()
                .uri(format!("/api/v1/notebook/{notebook_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(missing_response.status(), http::StatusCode::NOT_FOUND);

    let _ = std::fs::remove_dir_all(test_data_root(&root));
}

#[tokio::test]
async fn question_notebook_entries_and_categories_match_frontend_contract() {
    let root = unique_test_knowledge_root();
    let app = app_with_knowledge_root(&root);

    let chat_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/internal/test-chat-turn")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "session_id": "quiz-session-1",
                        "content": "Create a quiz source session."
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(chat_response.status(), http::StatusCode::OK);

    let missing_upsert_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/question-notebook/entries/upsert")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "session_id": "missing-session",
                        "question_id": "q1",
                        "question": "Missing?"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        missing_upsert_response.status(),
        http::StatusCode::NOT_FOUND
    );

    let upsert_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/question-notebook/entries/upsert")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "session_id": "quiz-session-1",
                        "question_id": "q-42",
                        "question": "Which agent checks citations?",
                        "question_type": "multiple_choice",
                        "options": { "A": "Planner", "B": "Critic" },
                        "correct_answer": "B",
                        "explanation": "The critic reviews citation coverage.",
                        "difficulty": "medium",
                        "user_answer": "A",
                        "is_correct": false
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(upsert_response.status(), http::StatusCode::OK);
    let entry = json_response(upsert_response).await;
    let entry_id = entry["id"].as_i64().unwrap();
    assert_eq!(entry["session_id"], "quiz-session-1");
    assert_eq!(entry["session_title"], "Create a quiz source session.");
    assert_eq!(entry["question_id"], "q-42");
    assert_eq!(entry["options"]["B"], "Critic");
    assert_eq!(entry["bookmarked"], false);

    let lookup_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/question-notebook/entries/lookup/by-question?session_id=quiz-session-1&question_id=q-42")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(lookup_response.status(), http::StatusCode::OK);
    assert_eq!(json_response(lookup_response).await["id"], entry_id);

    let category_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/question-notebook/categories")
                .header("content-type", "application/json")
                .body(Body::from(json!({"name": "Wrong answers"}).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(category_response.status(), http::StatusCode::CREATED);
    let category = json_response(category_response).await;
    let category_id = category["id"].as_i64().unwrap();
    assert_eq!(category["name"], "Wrong answers");
    assert_eq!(category["entry_count"], 0);

    let add_category_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri(format!(
                    "/api/v1/question-notebook/entries/{entry_id}/categories"
                ))
                .header("content-type", "application/json")
                .body(Body::from(json!({"category_id": category_id}).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(add_category_response.status(), http::StatusCode::OK);
    assert_eq!(json_response(add_category_response).await["added"], true);

    let list_filtered_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri(format!(
                    "/api/v1/question-notebook/entries?category_id={category_id}&is_correct=false&limit=200"
                ))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(list_filtered_response.status(), http::StatusCode::OK);
    let filtered = json_response(list_filtered_response).await;
    assert_eq!(filtered["total"], 1);
    assert_eq!(filtered["items"][0]["id"], entry_id);

    let get_entry_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri(format!("/api/v1/question-notebook/entries/{entry_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(get_entry_response.status(), http::StatusCode::OK);
    let entry_detail = json_response(get_entry_response).await;
    assert_eq!(entry_detail["categories"][0]["id"], category_id);
    assert_eq!(entry_detail["categories"][0]["name"], "Wrong answers");

    let update_entry_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("PATCH")
                .uri(format!("/api/v1/question-notebook/entries/{entry_id}"))
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "bookmarked": true,
                        "followup_session_id": "followup-1"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(update_entry_response.status(), http::StatusCode::OK);
    assert_eq!(json_response(update_entry_response).await["updated"], true);

    let bookmarked_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/question-notebook/entries?bookmarked=true")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(bookmarked_response.status(), http::StatusCode::OK);
    assert_eq!(json_response(bookmarked_response).await["total"], 1);

    let rename_category_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("PATCH")
                .uri(format!(
                    "/api/v1/question-notebook/categories/{category_id}"
                ))
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"name": "Reviewed wrong answers"}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(rename_category_response.status(), http::StatusCode::OK);
    assert_eq!(
        json_response(rename_category_response).await["name"],
        "Reviewed wrong answers"
    );

    let categories_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/question-notebook/categories")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(categories_response.status(), http::StatusCode::OK);
    let categories = json_response(categories_response).await;
    assert_eq!(categories[0]["entry_count"], 1);

    let remove_category_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("DELETE")
                .uri(format!(
                    "/api/v1/question-notebook/entries/{entry_id}/categories/{category_id}"
                ))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(remove_category_response.status(), http::StatusCode::OK);
    assert_eq!(
        json_response(remove_category_response).await["removed"],
        true
    );

    let delete_entry_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("DELETE")
                .uri(format!("/api/v1/question-notebook/entries/{entry_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(delete_entry_response.status(), http::StatusCode::OK);

    let delete_category_response = app
        .oneshot(
            http::Request::builder()
                .method("DELETE")
                .uri(format!(
                    "/api/v1/question-notebook/categories/{category_id}"
                ))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(delete_category_response.status(), http::StatusCode::OK);

    let _ = std::fs::remove_dir_all(test_data_root(&root));
}

#[tokio::test]
async fn chat_sessions_are_persisted_and_manageable() {
    let root = unique_test_knowledge_root();
    let app = app_with_knowledge_root(&root);

    let ws_payload = json!({
        "type": "start_turn",
        "content": "Explain persistent Socartes chat history.",
        "language": "en",
        "knowledge_bases": ["socartes-rust-rag"],
        "llm_selection": {
            "profile_id": "socartes-rust",
            "model_id": "deterministic-agent-loop"
        }
    });

    let learn_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/internal/test-chat-turn")
                .header("content-type", "application/json")
                .body(Body::from(ws_payload.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(learn_response.status(), http::StatusCode::OK);
    let learn_payload = json_response(learn_response).await;
    let session_id = learn_payload["session_id"].as_str().unwrap();

    let list_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/sessions?limit=50&offset=0")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(list_response.status(), http::StatusCode::OK);
    let list_payload = json_response(list_response).await;
    assert_eq!(list_payload["sessions"][0]["session_id"], session_id);
    assert_eq!(list_payload["sessions"][0]["message_count"], 2);

    let detail_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri(format!("/api/v1/sessions/{session_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(detail_response.status(), http::StatusCode::OK);
    let detail_payload = json_response(detail_response).await;
    assert_eq!(detail_payload["messages"][0]["role"], "user");
    assert_eq!(detail_payload["messages"][1]["role"], "assistant");
    assert_eq!(
        detail_payload["preferences"]["knowledge_bases"],
        json!(["socartes-rust-rag"])
    );

    let rename_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("PATCH")
                .uri(format!("/api/v1/sessions/{session_id}"))
                .header("content-type", "application/json")
                .body(Body::from(json!({"title": "Renamed session"}).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(rename_response.status(), http::StatusCode::OK);
    let rename_payload = json_response(rename_response).await;
    assert_eq!(rename_payload["session"]["title"], "Renamed session");

    let quiz_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri(format!("/api/v1/sessions/{session_id}/quiz-results"))
                .header("content-type", "application/json")
                .body(Body::from(json!({"answers": []}).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(quiz_response.status(), http::StatusCode::OK);
    assert_eq!(json_response(quiz_response).await["recorded"], true);

    let delete_response = app
        .oneshot(
            http::Request::builder()
                .method("DELETE")
                .uri(format!("/api/v1/sessions/{session_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(delete_response.status(), http::StatusCode::OK);
    assert_eq!(json_response(delete_response).await["deleted"], true);

    let _ = std::fs::remove_dir_all(root);
}

#[tokio::test]
async fn chat_ws_subscribe_turn_replays_persisted_events_after_seq() {
    let root = unique_test_knowledge_root();
    let server_root = root.clone();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, app_with_knowledge_root(server_root))
            .await
            .unwrap();
    });

    let (mut socket, _) = connect_async(format!("ws://{addr}/api/v1/ws"))
        .await
        .unwrap();
    socket
        .send(TungsteniteMessage::Text(
            json!({
                "type": "start_turn",
                "content": "Explain replayable Socartes turn events.",
                "language": "en",
                "knowledge_bases": ["socartes-rust-rag"]
            })
            .to_string()
            .into(),
        ))
        .await
        .unwrap();

    let mut first_events = Vec::new();
    while let Some(message) = socket.next().await {
        match message.unwrap() {
            TungsteniteMessage::Text(text) => {
                let event: Value = serde_json::from_str(&text).unwrap();
                let done = event["type"] == "done";
                first_events.push(event);
                if done {
                    break;
                }
            }
            TungsteniteMessage::Close(_) => break,
            _ => {}
        }
    }
    assert_eq!(
        first_events
            .iter()
            .map(|event| event["seq"].as_u64().unwrap())
            .collect::<Vec<_>>(),
        vec![1, 2, 3, 4, 5, 6, 7]
    );
    let session_id = first_events[0]["session_id"].as_str().unwrap().to_string();
    let turn_id = first_events[0]["turn_id"].as_str().unwrap().to_string();
    socket.close(None).await.unwrap();

    let (mut replay_socket, _) = connect_async(format!("ws://{addr}/api/v1/ws"))
        .await
        .unwrap();
    replay_socket
        .send(TungsteniteMessage::Text(
            json!({
                "type": "subscribe_turn",
                "turn_id": turn_id,
                "after_seq": 3
            })
            .to_string()
            .into(),
        ))
        .await
        .unwrap();

    let mut replay_events = Vec::new();
    while let Some(message) = replay_socket.next().await {
        match message.unwrap() {
            TungsteniteMessage::Text(text) => {
                let event: Value = serde_json::from_str(&text).unwrap();
                let done = event["type"] == "done";
                replay_events.push(event);
                if done {
                    break;
                }
            }
            TungsteniteMessage::Close(_) => break,
            _ => {}
        }
    }

    server.abort();
    assert_eq!(
        replay_events
            .iter()
            .map(|event| event["seq"].as_u64().unwrap())
            .collect::<Vec<_>>(),
        vec![4, 5, 6, 7]
    );
    assert!(
        replay_events
            .iter()
            .all(|event| event["session_id"] == session_id)
    );
    assert!(
        replay_events
            .iter()
            .all(|event| event["turn_id"] == turn_id)
    );
    assert_eq!(replay_events.last().unwrap()["type"], "done");
    assert_eq!(
        replay_events.last().unwrap()["metadata"]["status"],
        "completed"
    );

    let _ = std::fs::remove_dir_all(test_data_root(&root));
}

#[tokio::test]
async fn chat_ws_resume_from_replays_tail_events_for_existing_turn() {
    let root = unique_test_knowledge_root();
    let app = app_with_knowledge_root(&root);
    let request_body = json!({
        "type": "start_turn",
        "content": "Explain resumable Socartes turn events.",
        "language": "en",
        "knowledge_bases": ["socartes-rust-rag"]
    });

    let turn_response = app
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/internal/test-chat-turn")
                .header("content-type", "application/json")
                .body(Body::from(request_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(turn_response.status(), http::StatusCode::OK);
    let turn = json_response(turn_response).await;
    let turn_id = turn["turn_id"].as_str().unwrap().to_string();

    let server_root = root.clone();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, app_with_knowledge_root(server_root))
            .await
            .unwrap();
    });
    let (mut socket, _) = connect_async(format!("ws://{addr}/api/v1/ws"))
        .await
        .unwrap();
    socket
        .send(TungsteniteMessage::Text(
            json!({
                "type": "resume_from",
                "turn_id": turn_id,
                "seq": 5
            })
            .to_string()
            .into(),
        ))
        .await
        .unwrap();

    let mut replay_events = Vec::new();
    while let Some(message) = socket.next().await {
        match message.unwrap() {
            TungsteniteMessage::Text(text) => {
                let event: Value = serde_json::from_str(&text).unwrap();
                let done = event["type"] == "done";
                replay_events.push(event);
                if done {
                    break;
                }
            }
            TungsteniteMessage::Close(_) => break,
            _ => {}
        }
    }

    server.abort();
    assert_eq!(
        replay_events
            .iter()
            .map(|event| event["seq"].as_u64().unwrap())
            .collect::<Vec<_>>(),
        vec![6, 7]
    );
    assert_eq!(replay_events[0]["type"], "stage_end");
    assert_eq!(replay_events[1]["type"], "done");

    let _ = std::fs::remove_dir_all(test_data_root(&root));
}

#[tokio::test]
async fn book_frontend_bootstrap_reads_file_backed_books_and_outputs() {
    let root = unique_test_knowledge_root();
    let app = app_with_knowledge_root(&root);
    let book_root = test_book_root(&root);
    let book_dir = book_root.join("book_existing-book");
    fs::create_dir_all(book_dir.join("pages")).unwrap();

    fs::write(
        book_dir.join("manifest.json"),
        serde_json::to_vec_pretty(&json!({
            "id": "existing-book",
            "title": "Existing Socartes Book",
            "description": "Loaded from the Python-compatible file-backed store.",
            "status": "ready",
            "proposal": null,
            "knowledge_bases": ["socartes-rust-rag"],
            "language": "en",
            "page_count": 1,
            "chapter_count": 1,
            "created_at": 1.0,
            "updated_at": 2.0,
            "metadata": { "page_chat_sessions": {} },
            "kb_fingerprints": {},
            "stale_page_ids": []
        }))
        .unwrap(),
    )
    .unwrap();
    fs::write(
        book_dir.join("spine.json"),
        serde_json::to_vec_pretty(&json!({
            "book_id": "existing-book",
            "chapters": [{
                "id": "chapter-1",
                "title": "Agent foundations",
                "learning_objectives": ["Trace Planner, Executor, and Critic roles"],
                "content_type": "overview",
                "source_anchors": [],
                "prerequisites": [],
                "page_ids": ["page-1"],
                "summary": "A compact Socartes chapter.",
                "order": 1
            }],
            "version": 1,
            "updated_at": 2.0,
            "concept_graph": null,
            "exploration_summary": "File-backed smoke test"
        }))
        .unwrap(),
    )
    .unwrap();
    fs::write(
        book_dir.join("progress.json"),
        serde_json::to_vec_pretty(&json!({
            "book_id": "existing-book",
            "current_page_id": "page-1",
            "visited_page_ids": [],
            "bookmarked_page_ids": [],
            "quiz_attempts": [],
            "weak_chapters": [],
            "score": 0.0,
            "updated_at": 2.0
        }))
        .unwrap(),
    )
    .unwrap();
    fs::write(
        book_dir.join("pages").join("page-1.json"),
        serde_json::to_vec_pretty(&json!({
            "id": "page-1",
            "book_id": "existing-book",
            "chapter_id": "chapter-1",
            "title": "Planner to Critic",
            "learning_objectives": ["Explain the loop"],
            "content_type": "overview",
            "status": "ready",
            "order": 1,
            "blocks": [{
                "id": "block-1",
                "type": "text",
                "status": "ready",
                "title": "Traceable answer loop",
                "params": {},
                "payload": { "body": "Planner proposes, Executor drafts, Critic checks evidence." },
                "source_anchors": [],
                "metadata": {},
                "error": "",
                "created_at": 2.0,
                "updated_at": 2.0
            }],
            "links": [],
            "parent_page_id": "",
            "error": "",
            "created_at": 2.0,
            "updated_at": 2.0
        }))
        .unwrap(),
    )
    .unwrap();

    let output_path = test_user_output_root(&root)
        .join("workspace")
        .join("chat")
        .join("math_animator")
        .join("run-1")
        .join("artifacts");
    fs::create_dir_all(&output_path).unwrap();
    fs::write(output_path.join("demo.txt"), "artifact body").unwrap();

    let list_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/book/books")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(list_response.status(), http::StatusCode::OK);
    let list_payload = json_response(list_response).await;
    assert_eq!(list_payload["books"][0]["id"], "existing-book");

    let detail_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/book/books/existing-book")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(detail_response.status(), http::StatusCode::OK);
    let detail_payload = json_response(detail_response).await;
    assert_eq!(detail_payload["book"]["title"], "Existing Socartes Book");
    assert_eq!(detail_payload["pages"][0]["blocks"][0]["type"], "text");

    let spine_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/book/books/existing-book/spine")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(spine_response.status(), http::StatusCode::OK);
    assert_eq!(
        json_response(spine_response).await["spine"]["chapters"][0]["id"],
        "chapter-1"
    );

    let page_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/book/books/existing-book/pages/page-1")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(page_response.status(), http::StatusCode::OK);
    assert_eq!(json_response(page_response).await["page"]["id"], "page-1");

    let health_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/book/books/existing-book/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(health_response.status(), http::StatusCode::OK);
    assert_eq!(
        json_response(health_response).await["kb_drift"]["has_drift"],
        false
    );

    let refresh_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/book/books/existing-book/refresh-fingerprints")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(refresh_response.status(), http::StatusCode::OK);
    assert_eq!(
        json_response(refresh_response).await["book_id"],
        "existing-book"
    );

    let output_response = app
        .oneshot(
            http::Request::builder()
                .uri("/api/outputs/workspace/chat/math_animator/run-1/artifacts/demo.txt")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(output_response.status(), http::StatusCode::OK);
    assert_eq!(text_response(output_response).await, "artifact body");

    let _ = std::fs::remove_dir_all(test_data_root(&root));
}

#[tokio::test]
async fn book_creation_and_editing_workflow_matches_frontend_contract() {
    let root = unique_test_knowledge_root();
    let app = app_with_knowledge_root(&root);

    let create_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/book/books")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "user_intent": "Build a short book about Rust multi-agent learning.",
                        "knowledge_bases": ["socartes-rust-rag"],
                        "language": "en"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(create_response.status(), http::StatusCode::OK);
    let create_payload = json_response(create_response).await;
    let book_id = create_payload["book"]["id"].as_str().unwrap().to_string();
    assert_eq!(create_payload["book"]["status"], "draft");
    assert_eq!(create_payload["proposal"]["estimated_chapters"], 1);

    let confirm_proposal_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/book/books/confirm-proposal")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "book_id": book_id,
                        "proposal": {
                            "title": "Edited Rust Agents",
                            "description": "Edited proposal",
                            "scope": "short",
                            "target_level": "intermediate",
                            "estimated_chapters": 1,
                            "rationale": "contract test"
                        }
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(confirm_proposal_response.status(), http::StatusCode::OK);
    let proposal_payload = json_response(confirm_proposal_response).await;
    assert_eq!(proposal_payload["book"]["status"], "spine_ready");
    assert_eq!(
        proposal_payload["spine"]["chapters"][0]["page_ids"][0],
        "page-1"
    );

    let confirm_spine_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/book/books/confirm-spine")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "book_id": book_id,
                        "spine": null,
                        "auto_compile": true
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(confirm_spine_response.status(), http::StatusCode::OK);
    let pages_payload = json_response(confirm_spine_response).await;
    let page_id = pages_payload["pages"][0]["id"]
        .as_str()
        .unwrap()
        .to_string();
    let first_block_id = pages_payload["pages"][0]["blocks"][0]["id"]
        .as_str()
        .unwrap()
        .to_string();
    assert_eq!(pages_payload["pages"][0]["status"], "ready");

    let compile_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/book/books/compile-page")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"book_id": book_id, "page_id": page_id, "force": true}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(compile_response.status(), http::StatusCode::OK);
    assert_eq!(json_response(compile_response).await["page"]["id"], page_id);

    let regenerate_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/book/books/regenerate-block")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "book_id": book_id,
                        "page_id": page_id,
                        "block_id": first_block_id,
                        "params_override": {"focus": "RAG"}
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(regenerate_response.status(), http::StatusCode::OK);
    assert_eq!(
        json_response(regenerate_response).await["block"]["metadata"]["regenerated"],
        true
    );

    let insert_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/book/books/insert-block")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "book_id": book_id,
                        "page_id": page_id,
                        "block_type": "callout",
                        "params": {"topic": "critic"},
                        "position": 1,
                        "compile_now": true
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(insert_response.status(), http::StatusCode::OK);
    let inserted_block = json_response(insert_response).await["block"].clone();
    let inserted_block_id = inserted_block["id"].as_str().unwrap().to_string();

    let move_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/book/books/move-block")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "book_id": book_id,
                        "page_id": page_id,
                        "block_id": inserted_block_id,
                        "new_position": 0
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(move_response.status(), http::StatusCode::OK);
    assert_eq!(json_response(move_response).await["ok"], true);

    let change_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/book/books/change-block-type")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "book_id": book_id,
                        "page_id": page_id,
                        "block_id": inserted_block_id,
                        "new_type": "text",
                        "params_override": {"body": "changed"}
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(change_response.status(), http::StatusCode::OK);
    assert_eq!(
        json_response(change_response).await["block"]["type"],
        "text"
    );

    let supplement_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/book/books/supplement")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"book_id": book_id, "page_id": page_id, "topic": "reflection"})
                        .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(supplement_response.status(), http::StatusCode::OK);
    assert_eq!(
        json_response(supplement_response).await["block"]["type"],
        "callout"
    );

    let deep_dive_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/book/books/deep-dive")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "book_id": book_id,
                        "parent_page_id": page_id,
                        "topic": "critic loop",
                        "content_type": "concept"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(deep_dive_response.status(), http::StatusCode::OK);
    assert_eq!(
        json_response(deep_dive_response).await["page"]["parent_page_id"],
        page_id
    );

    let quiz_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/book/books/quiz-attempt")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "book_id": book_id,
                        "page_id": page_id,
                        "block_id": first_block_id,
                        "question_id": "q1",
                        "user_answer": "planner",
                        "is_correct": true
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(quiz_response.status(), http::StatusCode::OK);
    assert_eq!(
        json_response(quiz_response).await["progress"]["quiz_attempts"][0]["is_correct"],
        true
    );

    let chat_link_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/book/books/page-chat-session")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "book_id": book_id,
                        "page_id": page_id,
                        "session_id": "session-123"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(chat_link_response.status(), http::StatusCode::OK);
    assert_eq!(
        json_response(chat_link_response).await["book"]["metadata"]["page_chat_sessions"][&page_id],
        "session-123"
    );

    let rebuild_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/book/books/rebuild")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"book_id": book_id, "auto_compile": true}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(rebuild_response.status(), http::StatusCode::OK);
    assert!(
        json_response(rebuild_response).await["pages"]
            .as_array()
            .unwrap()
            .iter()
            .any(|page| page["id"] == page_id)
    );

    let delete_block_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/book/books/delete-block")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"book_id": book_id, "page_id": page_id, "block_id": inserted_block_id})
                        .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(delete_block_response.status(), http::StatusCode::OK);
    assert_eq!(json_response(delete_block_response).await["ok"], true);

    let delete_book_response = app
        .oneshot(
            http::Request::builder()
                .method("DELETE")
                .uri(format!("/api/v1/book/books/{book_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(delete_book_response.status(), http::StatusCode::OK);
    assert_eq!(json_response(delete_book_response).await["deleted"], true);

    let _ = std::fs::remove_dir_all(test_data_root(&root));
}

#[tokio::test]
async fn settings_and_system_status_match_frontend_contract() {
    let root = unique_test_knowledge_root();
    let app = app_with_knowledge_root(&root);

    let settings_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/settings")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(settings_response.status(), http::StatusCode::OK);
    let settings_payload = json_response(settings_response).await;
    assert_eq!(settings_payload["ui"]["theme"], "light");
    assert!(settings_payload["catalog"]["services"]["llm"]["profiles"].is_array());
    assert_eq!(
        settings_payload["providers"]["embedding"][0]["default_dim"],
        "3072"
    );

    let ui_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("PUT")
                .uri("/api/v1/settings/ui")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"theme": "dark", "language": "ko"}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(ui_response.status(), http::StatusCode::OK);
    let ui_payload = json_response(ui_response).await;
    assert_eq!(ui_payload["theme"], "dark");
    assert_eq!(ui_payload["language"], "ko");

    let mut catalog = settings_payload["catalog"].clone();
    catalog["services"]["llm"]["profiles"][0]["models"][0]["model"] = json!("socartes-test-model");
    let catalog_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("PUT")
                .uri("/api/v1/settings/catalog")
                .header("content-type", "application/json")
                .body(Body::from(json!({"catalog": catalog}).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(catalog_response.status(), http::StatusCode::OK);
    let catalog_payload = json_response(catalog_response).await;
    assert_eq!(
        catalog_payload["catalog"]["services"]["llm"]["profiles"][0]["models"][0]["model"],
        "socartes-test-model"
    );

    let apply_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/settings/apply")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"catalog": catalog_payload["catalog"].clone()}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(apply_response.status(), http::StatusCode::OK);
    let apply_payload = json_response(apply_response).await;
    assert!(
        apply_payload["message"]
            .as_str()
            .unwrap()
            .contains("Catalog applied")
    );
    assert_eq!(
        apply_payload["env"]["SOCARTES_LLM_MODEL"],
        "socartes-test-model"
    );

    let status_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/system/status")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(status_response.status(), http::StatusCode::OK);
    let status_payload = json_response(status_response).await;
    assert_eq!(status_payload["backend"]["status"], "online");
    assert_eq!(status_payload["llm"]["model"], "socartes-test-model");
    assert_eq!(status_payload["embeddings"]["status"], "configured");

    let start_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/settings/tests/embedding/start")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"catalog": catalog_payload["catalog"].clone()}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(start_response.status(), http::StatusCode::OK);
    let run_id = json_response(start_response).await["run_id"]
        .as_str()
        .unwrap()
        .to_string();

    let events_response = app
        .oneshot(
            http::Request::builder()
                .uri(format!("/api/v1/settings/tests/embedding/{run_id}/events"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(events_response.status(), http::StatusCode::OK);
    assert_eq!(
        events_response
            .headers()
            .get(http::header::CONTENT_TYPE)
            .unwrap(),
        "text/event-stream"
    );
    let events_body = text_response(events_response).await;
    assert!(events_body.contains("\"type\":\"capabilities\""));
    assert!(events_body.contains("\"type\":\"completed\""));

    let _ = std::fs::remove_dir_all(test_data_root(&root));
}

#[tokio::test]
async fn legacy_settings_dashboard_agent_config_and_solve_routes_match_python_contracts() {
    let root = unique_test_knowledge_root();
    let session_root = test_data_root(&root).join("sessions");
    fs::create_dir_all(&session_root).unwrap();
    fs::write(
        session_root.join("legacy-session.json"),
        json!({
            "id": "legacy-session",
            "session_id": "legacy-session",
            "title": "Legacy Solve",
            "created_at": 10.0,
            "updated_at": 20.0,
            "status": "idle",
            "capability": "deep_solve",
            "preferences": {"knowledge_bases": ["course-a"]},
            "messages": [
                {"role": "user", "content": "Solve this", "created_at": 11.0},
                {"role": "assistant", "content": "Solved answer", "created_at": 12.0}
            ],
            "active_turns": [],
            "compressed_summary": "short summary"
        })
        .to_string(),
    )
    .unwrap();
    let app = app_with_knowledge_root(&root);

    let description_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("PUT")
                .uri("/api/v1/settings/sidebar/description")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"description": "Socartes Lab"}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(description_response.status(), http::StatusCode::OK);
    assert_eq!(
        json_response(description_response).await,
        json!({"description": "Socartes Lab"})
    );

    let nav_order = json!({
        "start": ["/", "/knowledge"],
        "learnResearch": ["/question", "/co_writer"]
    });
    let nav_response = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("PUT")
                .uri("/api/v1/settings/sidebar/nav-order")
                .header("content-type", "application/json")
                .body(Body::from(json!({"nav_order": nav_order}).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(nav_response.status(), http::StatusCode::OK);
    assert_eq!(json_response(nav_response).await["nav_order"], nav_order);

    let sidebar = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/settings/sidebar")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let sidebar_payload = json_response(sidebar).await;
    assert_eq!(sidebar_payload["description"], "Socartes Lab");
    assert_eq!(sidebar_payload["nav_order"], nav_order);

    let tour_status = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/settings/tour/status")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(tour_status.status(), http::StatusCode::OK);
    assert_eq!(
        json_response(tour_status).await,
        json!({"active": false, "status": "none", "launch_at": null, "redirect_at": null})
    );

    let complete_tour = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/settings/tour/complete")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"test_results": {"llm": "ok"}}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(complete_tour.status(), http::StatusCode::OK);
    let complete_payload = json_response(complete_tour).await;
    assert_eq!(complete_payload["status"], "completed");
    assert!(complete_payload["launch_at"].as_i64().unwrap() > 0);
    assert!(complete_payload["redirect_at"].as_i64().unwrap() > 0);
    assert!(complete_payload["env"].is_object());

    let reopen_tour = app
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/settings/tour/reopen")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(reopen_tour.status(), http::StatusCode::OK);
    assert_eq!(
        json_response(reopen_tour).await["command"],
        "python scripts/start_tour.py"
    );

    let agents = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/agent-config/agents")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(agents.status(), http::StatusCode::OK);
    let agents_payload = json_response(agents).await;
    assert_eq!(agents_payload["solve"]["icon"], "HelpCircle");
    assert_eq!(agents_payload["co_writer"]["color"], "amber");

    let missing_agent = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/agent-config/agents/unknown")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(missing_agent.status(), http::StatusCode::OK);
    assert_eq!(
        json_response(missing_agent).await,
        json!({"error": "Agent type 'unknown' not found"})
    );

    let dashboard_recent = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/dashboard/recent?type=solve&limit=10")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(dashboard_recent.status(), http::StatusCode::OK);
    let activities = json_response(dashboard_recent).await;
    assert_eq!(activities.as_array().unwrap().len(), 1);
    assert_eq!(activities[0]["id"], "legacy-session");
    assert_eq!(activities[0]["type"], "solve");
    assert_eq!(activities[0]["summary"], "Solved answer");

    let dashboard_entry = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/dashboard/legacy-session")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(dashboard_entry.status(), http::StatusCode::OK);
    let entry = json_response(dashboard_entry).await;
    assert_eq!(entry["id"], "legacy-session");
    assert_eq!(entry["type"], "solve");
    assert_eq!(entry["content"]["messages"].as_array().unwrap().len(), 2);

    let solve_sessions = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/solve/sessions?limit=5")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(solve_sessions.status(), http::StatusCode::OK);
    let solve_payload = json_response(solve_sessions).await;
    assert_eq!(solve_payload.as_array().unwrap().len(), 1);
    assert_eq!(solve_payload[0]["session_id"], "legacy-session");
    assert_eq!(solve_payload[0]["kb_name"], "course-a");
    assert_eq!(solve_payload[0]["last_message"], "Solved answer");

    let solve_detail = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/solve/sessions/legacy-session")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(solve_detail.status(), http::StatusCode::OK);
    let solve_detail_payload = json_response(solve_detail).await;
    assert_eq!(solve_detail_payload["session_id"], "legacy-session");
    assert!(solve_detail_payload["token_stats"].is_object());

    let missing_solve = app
        .clone()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/solve/sessions/missing-session")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(missing_solve.status(), http::StatusCode::NOT_FOUND);
    assert_eq!(
        json_response(missing_solve).await["detail"],
        "Session not found"
    );

    let missing_dashboard = app
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/dashboard/missing-session")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(missing_dashboard.status(), http::StatusCode::NOT_FOUND);
    assert_eq!(
        json_response(missing_dashboard).await["detail"],
        "Entry not found"
    );

    let _ = std::fs::remove_dir_all(test_data_root(&root));
}

#[tokio::test]
async fn vision_analyze_rest_route_matches_legacy_validation_contract() {
    let no_image_response = app()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/vision/analyze")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "question": "Describe the geometry",
                        "session_id": "vision-session"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(no_image_response.status(), http::StatusCode::OK);
    let no_image = json_response(no_image_response).await;
    assert_eq!(no_image["session_id"], "vision-session");
    assert_eq!(no_image["has_image"], false);
    assert_eq!(no_image["final_ggb_commands"], json!([]));
    assert_eq!(no_image["ggb_script"], Value::Null);
    assert_eq!(no_image["analysis_summary"], json!({}));

    let invalid_base64_response = app()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/vision/analyze")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "question": "Analyze this",
                        "image_base64": "not-a-data-uri"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        invalid_base64_response.status(),
        http::StatusCode::BAD_REQUEST
    );
    assert!(
        json_response(invalid_base64_response).await["detail"]
            .as_str()
            .unwrap()
            .contains("Invalid base64 image format")
    );

    let invalid_url_response = app()
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/vision/analyze")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "question": "Analyze this",
                        "image_url": "ftp://example.com/image.png"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(invalid_url_response.status(), http::StatusCode::BAD_REQUEST);
    assert_eq!(
        json_response(invalid_url_response).await["detail"],
        "Invalid image URL: ftp://example.com/image.png"
    );

    let base64_priority_response = app()
        .clone()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/vision/analyze")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "question": "Analyze this",
                        "image_base64": "not-a-data-uri",
                        "image_url": "https://example.com/image.png"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        base64_priority_response.status(),
        http::StatusCode::BAD_REQUEST
    );
    assert!(
        json_response(base64_priority_response).await["detail"]
            .as_str()
            .unwrap()
            .contains("Invalid base64 image format")
    );

    let image_response = app()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/vision/analyze")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "question": "Analyze this",
                        "image_base64": "data:image/png;base64,iVBORw0KGgo="
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(image_response.status(), http::StatusCode::OK);
    let image_payload = json_response(image_response).await;
    assert_eq!(image_payload["has_image"], true);
    assert_eq!(image_payload["final_ggb_commands"], json!([]));
    assert_eq!(image_payload["ggb_script"], Value::Null);
    assert_eq!(image_payload["analysis_summary"]["mime_type"], "image/png");
    assert_eq!(
        image_payload["analysis_summary"]["analysis_mode"],
        "metadata_only"
    );
    assert_eq!(image_payload["analysis_summary"]["commands_count"], 0);

    let default_session_response = app()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/vision/analyze")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "question": "Describe this"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(default_session_response.status(), http::StatusCode::OK);
    assert!(
        json_response(default_session_response).await["session_id"]
            .as_str()
            .unwrap()
            .starts_with("vision_")
    );
}

#[tokio::test]
async fn vision_solve_websocket_route_is_registered_for_legacy_clients() {
    let response = app()
        .oneshot(
            http::Request::builder()
                .uri("/api/v1/vision/solve")
                .header("connection", "upgrade")
                .header("upgrade", "websocket")
                .header("sec-websocket-version", "13")
                .header("sec-websocket-key", "dGhlIHNhbXBsZSBub25jZQ==")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_ne!(response.status(), http::StatusCode::NOT_FOUND);
    assert_ne!(response.status(), http::StatusCode::METHOD_NOT_ALLOWED);
}

#[tokio::test]
async fn vision_solve_websocket_streams_legacy_no_image_events() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, app()).await.unwrap();
    });

    let (mut socket, _) = connect_async(format!("ws://{addr}/api/v1/vision/solve"))
        .await
        .unwrap();
    socket
        .send(TungsteniteMessage::Text(
            json!({
                "question": "Explain the diagram",
                "session_id": "vision-test-session"
            })
            .to_string()
            .into(),
        ))
        .await
        .unwrap();

    let mut events = Vec::new();
    while let Some(message) = socket.next().await {
        match message.unwrap() {
            TungsteniteMessage::Text(text) => {
                let event: Value = serde_json::from_str(&text).unwrap();
                let event_type = event["type"].as_str().unwrap().to_string();
                events.push(event);
                if event_type == "done" {
                    break;
                }
            }
            TungsteniteMessage::Close(_) => break,
            _ => {}
        }
    }

    server.abort();
    assert_eq!(
        events[0],
        json!({"type": "session", "session_id": "vision-test-session"})
    );
    assert_eq!(events[1], json!({"type": "no_image", "data": {}}));
    assert_eq!(events[2], json!({"type": "done"}));
}

#[tokio::test]
async fn vision_solve_websocket_streams_metadata_only_image_events() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, app()).await.unwrap();
    });

    let (mut socket, _) = connect_async(format!("ws://{addr}/api/v1/vision/solve"))
        .await
        .unwrap();
    socket
        .send(TungsteniteMessage::Text(
            json!({
                "question": "Explain the diagram",
                "session_id": "vision-test-image",
                "image_base64": "data:image/png;base64,iVBORw0KGgo="
            })
            .to_string()
            .into(),
        ))
        .await
        .unwrap();

    let mut types = Vec::new();
    let mut events = Vec::new();
    while let Some(message) = socket.next().await {
        match message.unwrap() {
            TungsteniteMessage::Text(text) => {
                let event: Value = serde_json::from_str(&text).unwrap();
                let event_type = event["type"].as_str().unwrap().to_string();
                types.push(event_type.clone());
                events.push(event);
                if event_type == "done" {
                    break;
                }
            }
            TungsteniteMessage::Close(_) => break,
            _ => {}
        }
    }

    server.abort();
    assert_eq!(
        types,
        [
            "session",
            "analysis_start",
            "bbox_complete",
            "analysis_complete",
            "ggbscript_complete",
            "reflection_complete",
            "analysis_message_complete",
            "answer_start",
            "done"
        ]
    );
    let summary = &events[6]["data"]["analysis_summary"];
    assert_eq!(summary["analysis_mode"], "metadata_only");
    assert_eq!(summary["input_source"], "image_base64");
    assert_eq!(summary["mime_type"], "image/png");
}

#[tokio::test]
async fn story_rag_endpoint_returns_grounded_source_id() {
    let response = app()
        .oneshot(
            http::Request::builder()
                .method("POST")
                .uri("/api/v1/story-rag/ask")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"question": "What did Jenkins say was in the pajama leg?"}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), http::StatusCode::OK);
    let payload = json_response(response).await;
    assert_eq!(payload["grounded"], true);
    assert_eq!(
        payload["source_ids"],
        json!(["haunted-pajamas-ch02-tarantula"])
    );
    assert!(
        payload["answer"]
            .as_str()
            .unwrap()
            .to_lowercase()
            .contains("tarantula")
    );
}

#[tokio::test]
async fn openapi_json_matches_the_original_documentation_route() {
    let response = app()
        .oneshot(
            http::Request::builder()
                .uri("/openapi.json")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), http::StatusCode::OK);
    let payload = json_response(response).await;
    assert_eq!(payload["info"]["title"], "Socartes Backend");
    assert_eq!(payload["info"]["version"], "0.1.0");
    assert!(payload["paths"]["/health"].is_object());
    assert!(payload["paths"]["/api/v1/agents"].is_object());
    assert!(payload["paths"]["/api/v1/learn"].is_object());
    assert!(payload["paths"]["/api/v1/story-rag/ask"].is_object());
}

#[tokio::test]
async fn fastapi_documentation_routes_are_available() {
    for (path, expected_text) in [
        ("/docs", "Swagger UI"),
        ("/docs/oauth2-redirect", "Swagger UI: OAuth2 Redirect"),
        ("/redoc", "ReDoc"),
    ] {
        let response = app()
            .oneshot(
                http::Request::builder()
                    .uri(path)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), http::StatusCode::OK);
        assert_eq!(
            response.headers().get(http::header::CONTENT_TYPE).unwrap(),
            "text/html; charset=utf-8"
        );

        let body = text_response(response).await;
        assert!(body.contains(expected_text));
        if path != "/docs/oauth2-redirect" {
            assert!(body.contains("/openapi.json"));
        }
    }
}
