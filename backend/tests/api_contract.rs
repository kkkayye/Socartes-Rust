use std::{
    fs,
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};

use axum::body::Body;
use axum::http;
use http_body_util::BodyExt;
use serde_json::{Value, json};
use socartes_backend::{app, app_with_knowledge_root};
use tower::ServiceExt;

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
    std::env::temp_dir()
        .join(format!("socartes-test-{id}"))
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
