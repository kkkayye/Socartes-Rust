use axum::body::Body;
use axum::http;
use http_body_util::BodyExt;
use serde_json::{Value, json};
use socartes_backend::app;
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
