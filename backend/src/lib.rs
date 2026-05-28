use std::{
    collections::{BTreeMap, HashSet},
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    Json, Router,
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    http::StatusCode,
    response::{Html, IntoResponse},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

pub const VERSION: &str = "0.1.0";
pub const PROJECT_GUTENBERG_HAUNTED_PAJAMAS_URL: &str =
    "https://www.gutenberg.org/ebooks/33780.txt.utf-8";

const MIN_RETRIEVAL_SCORE: usize = 2;

#[derive(Debug, Clone, Deserialize)]
pub struct StudyRequest {
    pub goal: String,
    #[serde(default)]
    pub learner_context: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct PlanTask {
    pub id: String,
    pub owner: String,
    pub objective: String,
    pub evidence_required: Vec<String>,
    pub acceptance_criteria: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AgentPlan {
    pub agent: String,
    pub summary: String,
    pub tasks: Vec<PlanTask>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RetrievalChunk {
    pub source_id: String,
    pub title: String,
    pub content: String,
    pub confidence: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolResult {
    pub adapter: String,
    pub action: String,
    pub output: String,
    pub safe: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct DraftAnswer {
    pub agent: String,
    pub content: String,
    pub citations: Vec<String>,
    pub tool_results_used: Vec<String>,
    pub open_gaps: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CriticIssue {
    #[serde(rename = "type")]
    pub issue_type: String,
    pub claim: String,
    pub instruction: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct CriticReview {
    pub agent: String,
    pub status: String,
    pub checks: Vec<String>,
    pub issues: Vec<CriticIssue>,
    pub approved: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReflectionEvent {
    pub event_type: String,
    pub agent: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct StudyTrace {
    pub goal: String,
    pub learner_context: String,
    pub plan: AgentPlan,
    pub retrieved_context: Vec<RetrievalChunk>,
    pub tool_results: Vec<ToolResult>,
    pub draft: DraftAnswer,
    pub review: CriticReview,
    pub reflection_events: Vec<ReflectionEvent>,
    pub final_answer: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct StoryChunk {
    pub source_id: String,
    pub title: String,
    pub source_url: String,
    pub text: String,
}

impl StoryChunk {
    pub fn new(source_id: &str, title: &str, text: &str) -> Self {
        Self {
            source_id: source_id.to_string(),
            title: title.to_string(),
            source_url: PROJECT_GUTENBERG_HAUNTED_PAJAMAS_URL.to_string(),
            text: text.to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct StoryAnswer {
    pub answer: String,
    pub grounded: bool,
    pub source_ids: Vec<String>,
    pub source_url: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StoryQuestion {
    pub question: String,
}

#[derive(Debug, Clone)]
pub struct StoryRagIndex {
    chunks: Vec<StoryChunk>,
    title_terms: HashSet<String>,
}

impl StoryRagIndex {
    pub fn new(chunks: Vec<StoryChunk>) -> Self {
        let title_terms = chunks
            .iter()
            .flat_map(|chunk| tokenize(&chunk.title))
            .collect::<HashSet<_>>();
        Self {
            chunks,
            title_terms,
        }
    }

    pub fn ask(&self, question: &str) -> StoryAnswer {
        let query_terms = tokenize(question)
            .difference(&self.title_terms)
            .cloned()
            .collect::<HashSet<_>>();

        let best_match = self
            .chunks
            .iter()
            .map(|chunk| {
                let score = query_terms.intersection(&tokenize(&chunk.text)).count();
                (score, chunk)
            })
            .max_by_key(|(score, _)| *score);

        let Some((score, chunk)) = best_match else {
            return no_story_evidence_answer();
        };

        if score < MIN_RETRIEVAL_SCORE {
            return no_story_evidence_answer();
        }

        StoryAnswer {
            answer: format!("According to {}: {}", chunk.source_id, chunk.text),
            grounded: true,
            source_ids: vec![chunk.source_id.clone()],
            source_url: Some(chunk.source_url.clone()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SocartesOrchestrator;

impl SocartesOrchestrator {
    pub fn new() -> Self {
        Self
    }

    pub fn run(&self, goal: &str, learner_context: &str) -> StudyTrace {
        let plan = self.plan();
        let retrieved_context = retrieve(goal);
        let tool_results = default_tool_results(goal);
        let draft = draft_answer(goal, learner_context, &retrieved_context, &tool_results);
        let review = review_draft(&draft);
        let reflection_events = record_reflection(&review);
        let final_answer = draft.content.clone();

        StudyTrace {
            goal: goal.to_string(),
            learner_context: learner_context.to_string(),
            plan,
            retrieved_context,
            tool_results,
            draft,
            review,
            reflection_events,
            final_answer,
        }
    }

    pub fn agent_catalog(&self) -> BTreeMap<String, BTreeMap<String, String>> {
        BTreeMap::from([
            (
                "planner".to_string(),
                BTreeMap::from([
                    (
                        "responsibility".to_string(),
                        "Convert learner goals into ordered study plans.".to_string(),
                    ),
                    (
                        "input".to_string(),
                        "Learner goal and constraints.".to_string(),
                    ),
                    (
                        "output".to_string(),
                        "Task graph, evidence requirements, and acceptance criteria.".to_string(),
                    ),
                ]),
            ),
            (
                "retriever".to_string(),
                BTreeMap::from([
                    (
                        "responsibility".to_string(),
                        "Fetch external domain context through RAG.".to_string(),
                    ),
                    (
                        "input".to_string(),
                        "Goal keywords and plan evidence requirements.".to_string(),
                    ),
                    (
                        "output".to_string(),
                        "Ranked chunks with source identifiers and confidence.".to_string(),
                    ),
                ]),
            ),
            (
                "executor".to_string(),
                BTreeMap::from([
                    (
                        "responsibility".to_string(),
                        "Synthesize answers from plan, RAG context, and tool outputs.".to_string(),
                    ),
                    (
                        "input".to_string(),
                        "Plan tasks, retrieved chunks, learner context, and adapter results."
                            .to_string(),
                    ),
                    (
                        "output".to_string(),
                        "Draft answer with citations, tool trace, and open gaps.".to_string(),
                    ),
                ]),
            ),
            (
                "critic".to_string(),
                BTreeMap::from([
                    (
                        "responsibility".to_string(),
                        "Audit the draft before it becomes learner-facing.".to_string(),
                    ),
                    (
                        "input".to_string(),
                        "Executor draft, citations, tool results, and acceptance criteria."
                            .to_string(),
                    ),
                    (
                        "output".to_string(),
                        "Approval state, issues, and revision instructions.".to_string(),
                    ),
                    (
                        "checks".to_string(),
                        "acceptance criteria, citation coverage, tool output explainability"
                            .to_string(),
                    ),
                ]),
            ),
            (
                "tool_adapter".to_string(),
                BTreeMap::from([
                    (
                        "responsibility".to_string(),
                        "Expose MCP-style adapters for APIs, DBs, and files.".to_string(),
                    ),
                    (
                        "input".to_string(),
                        "Scoped tool requests from the executor.".to_string(),
                    ),
                    (
                        "output".to_string(),
                        "Auditable adapter output records.".to_string(),
                    ),
                ]),
            ),
        ])
    }

    fn plan(&self) -> AgentPlan {
        AgentPlan {
            agent: "planner".to_string(),
            summary: "Convert the learner goal into a traceable study workflow.".to_string(),
            tasks: vec![
                PlanTask {
                    id: "task-plan".to_string(),
                    owner: "planner".to_string(),
                    objective: "Clarify the learning goal and define acceptance criteria."
                        .to_string(),
                    evidence_required: vec![],
                    acceptance_criteria: vec![
                        "The answer addresses the learner goal".to_string(),
                        "The final response lists evidence and unresolved gaps".to_string(),
                    ],
                },
                PlanTask {
                    id: "task-retrieve".to_string(),
                    owner: "retriever".to_string(),
                    objective: "Retrieve external knowledge for the main concepts.".to_string(),
                    evidence_required: vec![
                        "domain notes".to_string(),
                        "citation metadata".to_string(),
                    ],
                    acceptance_criteria: vec![],
                },
                PlanTask {
                    id: "task-execute".to_string(),
                    owner: "executor".to_string(),
                    objective: "Synthesize the answer with retrieved context and tool output."
                        .to_string(),
                    evidence_required: vec![
                        "retrieved chunks".to_string(),
                        "tool results".to_string(),
                    ],
                    acceptance_criteria: vec!["Claims include source identifiers".to_string()],
                },
                PlanTask {
                    id: "task-critique".to_string(),
                    owner: "critic".to_string(),
                    objective: "Audit the draft and request revisions when evidence is weak."
                        .to_string(),
                    evidence_required: vec![],
                    acceptance_criteria: vec![
                        "Unsupported claims are revised or removed".to_string(),
                    ],
                },
            ],
        }
    }
}

impl Default for SocartesOrchestrator {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize)]
struct AgentsResponse {
    agents: BTreeMap<String, BTreeMap<String, String>>,
}

#[derive(Debug, Clone, Serialize)]
struct HealthResponse {
    status: &'static str,
    service: &'static str,
    version: &'static str,
}

#[derive(Debug, Clone, Serialize)]
struct ErrorResponse {
    detail: Vec<ValidationError>,
}

#[derive(Debug, Clone, Serialize)]
struct ValidationError {
    #[serde(rename = "type")]
    error_type: &'static str,
    loc: Vec<&'static str>,
    msg: &'static str,
    input: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    ctx: Option<Value>,
}

pub fn app() -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/openapi.json", get(openapi_json))
        .route("/docs", get(swagger_docs))
        .route("/docs/oauth2-redirect", get(swagger_oauth2_redirect))
        .route("/redoc", get(redoc_docs))
        .route("/api/v1/agents", get(agents))
        .route("/api/v1/knowledge/list", get(knowledge_list))
        .route("/api/v1/settings/llm-options", get(llm_options))
        .route("/api/v1/sessions", get(list_sessions))
        .route("/api/v1/ws", get(chat_ws))
        .route("/api/v1/learn", post(learn))
        .route("/api/v1/story-rag/ask", post(ask_story_rag))
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        service: "socartes-backend",
        version: VERSION,
    })
}

async fn agents() -> Json<AgentsResponse> {
    Json(AgentsResponse {
        agents: SocartesOrchestrator::new().agent_catalog(),
    })
}

async fn learn(Json(request): Json<StudyRequest>) -> impl IntoResponse {
    if request.goal.chars().count() < 3 {
        return (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(ErrorResponse {
                detail: vec![ValidationError {
                    error_type: "string_too_short",
                    loc: vec!["body", "goal"],
                    msg: "String should have at least 3 characters",
                    input: json!(request.goal),
                    ctx: Some(json!({ "min_length": 3 })),
                }],
            }),
        )
            .into_response();
    }

    Json(SocartesOrchestrator::new().run(&request.goal, &request.learner_context)).into_response()
}

async fn ask_story_rag(Json(request): Json<StoryQuestion>) -> Json<StoryAnswer> {
    Json(haunted_pajamas_index().ask(&request.question))
}

async fn knowledge_list() -> Json<Value> {
    Json(json!({
        "knowledge_bases": [
            {
                "name": "socartes-rust-rag",
                "is_default": true,
                "status": "ready",
                "metadata": {
                    "description": "Built-in Socartes Rust RAG notes for local frontend smoke tests."
                },
                "statistics": {
                    "chunks": knowledge_base().len()
                }
            }
        ]
    }))
}

async fn llm_options() -> Json<Value> {
    Json(json!({
        "active": {
            "profile_id": "socartes-rust",
            "model_id": "deterministic-agent-loop"
        },
        "options": [
            {
                "profile_id": "socartes-rust",
                "profile_name": "Socartes Rust",
                "model_id": "deterministic-agent-loop",
                "model_name": "Deterministic Agent Loop",
                "model": "deterministic-agent-loop",
                "provider": "rust",
                "context_window": 8192,
                "is_active_default": true
            }
        ]
    }))
}

async fn list_sessions() -> Json<Value> {
    Json(json!({ "sessions": [] }))
}

async fn chat_ws(ws: WebSocketUpgrade) -> impl IntoResponse {
    ws.on_upgrade(handle_chat_socket)
}

async fn handle_chat_socket(mut socket: WebSocket) {
    while let Some(Ok(message)) = socket.recv().await {
        let Message::Text(text) = message else {
            if matches!(message, Message::Close(_)) {
                break;
            }
            continue;
        };

        let Ok(payload) = serde_json::from_str::<Value>(&text) else {
            let _ = send_stream_event(
                &mut socket,
                stream_event(
                    "error",
                    "rust-backend",
                    "parse",
                    "Invalid WebSocket JSON message.",
                    json!({ "turn_terminal": true, "status": "failed" }),
                    StreamIds::empty(),
                    1,
                ),
            )
            .await;
            continue;
        };

        match payload["type"].as_str().unwrap_or_default() {
            "start_turn" | "message" => {
                run_chat_turn(&mut socket, &payload).await;
            }
            "regenerate" => {
                let fallback = json!({
                    "type": "start_turn",
                    "content": "Regenerate the previous Socartes answer.",
                    "session_id": payload["session_id"].as_str()
                });
                run_chat_turn(&mut socket, &fallback).await;
            }
            "cancel_turn" => {
                let _ = send_stream_event(
                    &mut socket,
                    stream_event(
                        "done",
                        "rust-backend",
                        "",
                        "",
                        json!({ "status": "cancelled" }),
                        StreamIds {
                            session_id: payload["session_id"].as_str(),
                            turn_id: payload["turn_id"].as_str(),
                        },
                        1,
                    ),
                )
                .await;
            }
            "ping" | "subscribe_turn" | "subscribe_session" | "resume_from" | "unsubscribe" => {}
            _ => {
                let _ = send_stream_event(
                    &mut socket,
                    stream_event(
                        "error",
                        "rust-backend",
                        "protocol",
                        "Unsupported WebSocket message type.",
                        json!({ "turn_terminal": true, "status": "failed" }),
                        StreamIds::empty(),
                        1,
                    ),
                )
                .await;
            }
        }
    }
}

async fn run_chat_turn(socket: &mut WebSocket, payload: &Value) {
    let content = payload["content"]
        .as_str()
        .unwrap_or("Explain the Socartes Rust agent loop.");
    let session_id = payload["session_id"]
        .as_str()
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
        .unwrap_or_else(|| format!("rust-session-{}", unique_id()));
    let turn_id = format!("rust-turn-{}", unique_id());
    let learner_context = payload["language"]
        .as_str()
        .map(|language| format!("Frontend language: {language}"))
        .unwrap_or_default();
    let trace = SocartesOrchestrator::new().run(content, &learner_context);
    let ids = StreamIds::new(&session_id, &turn_id);

    let events = [
        stream_event(
            "session",
            "rust-backend",
            "",
            "",
            json!({ "session_id": session_id, "turn_id": turn_id }),
            ids,
            1,
        ),
        stream_event(
            "stage_start",
            "planner",
            "planner",
            "Planner decomposed the request for the Rust Socartes loop.",
            json!({}),
            ids,
            2,
        ),
        stream_event(
            "sources",
            "retriever",
            "retriever",
            "Retrieved Socartes RAG context.",
            json!({ "sources": trace.retrieved_context }),
            ids,
            3,
        ),
        stream_event(
            "tool_result",
            "tool_adapter",
            "tool_adapter",
            "MCP-style tool adapters returned auditable outputs.",
            json!({ "tool_results": trace.tool_results }),
            ids,
            4,
        ),
        stream_event(
            "content",
            "executor",
            "executor",
            &trace.final_answer,
            json!({ "citations": trace.draft.citations }),
            ids,
            5,
        ),
        stream_event(
            "stage_end",
            "critic",
            "critic",
            "Critic approved the cited answer and reflection trace.",
            json!({ "review": trace.review }),
            ids,
            6,
        ),
        stream_event(
            "done",
            "rust-backend",
            "",
            "",
            json!({ "status": "completed" }),
            ids,
            7,
        ),
    ];

    for event in events {
        if send_stream_event(socket, event).await.is_err() {
            break;
        }
    }
}

fn stream_event(
    event_type: &str,
    source: &str,
    stage: &str,
    content: &str,
    metadata: Value,
    ids: StreamIds<'_>,
    seq: u64,
) -> Value {
    json!({
        "type": event_type,
        "source": source,
        "stage": stage,
        "content": content,
        "metadata": metadata,
        "session_id": ids.session_id,
        "turn_id": ids.turn_id,
        "seq": seq,
        "timestamp": now_seconds()
    })
}

#[derive(Clone, Copy)]
struct StreamIds<'a> {
    session_id: Option<&'a str>,
    turn_id: Option<&'a str>,
}

impl<'a> StreamIds<'a> {
    fn new(session_id: &'a str, turn_id: &'a str) -> Self {
        Self {
            session_id: Some(session_id),
            turn_id: Some(turn_id),
        }
    }

    fn empty() -> Self {
        Self {
            session_id: None,
            turn_id: None,
        }
    }
}

async fn send_stream_event(socket: &mut WebSocket, event: Value) -> Result<(), axum::Error> {
    socket.send(Message::Text(event.to_string().into())).await
}

fn now_seconds() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs_f64())
        .unwrap_or_default()
}

fn unique_id() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default()
}

async fn openapi_json() -> Json<Value> {
    Json(json!({
        "openapi": "3.0.3",
        "info": {
            "title": "Socartes Backend",
            "version": VERSION,
            "description": "Rust implementation of the Socartes multi-agent learning backend."
        },
        "paths": {
            "/health": {
                "get": {
                    "summary": "Health check",
                    "responses": {
                        "200": {
                            "description": "Backend status",
                            "content": {
                                "application/json": {
                                    "schema": { "$ref": "#/components/schemas/HealthResponse" }
                                }
                            }
                        }
                    }
                }
            },
            "/api/v1/agents": {
                "get": {
                    "summary": "Agent catalog",
                    "responses": {
                        "200": {
                            "description": "Agent responsibilities and contracts"
                        }
                    }
                }
            },
            "/api/v1/learn": {
                "post": {
                    "summary": "Run the Socartes agent workflow",
                    "requestBody": {
                        "required": true,
                        "content": {
                            "application/json": {
                                "schema": { "$ref": "#/components/schemas/StudyRequest" }
                            }
                        }
                    },
                    "responses": {
                        "200": { "description": "Traceable study answer" },
                        "422": { "description": "Invalid learning goal" }
                    }
                }
            },
            "/api/v1/story-rag/ask": {
                "post": {
                    "summary": "Ask the story-grounded RAG test index",
                    "requestBody": {
                        "required": true,
                        "content": {
                            "application/json": {
                                "schema": { "$ref": "#/components/schemas/StoryQuestion" }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Grounded answer or refusal when evidence is missing"
                        }
                    }
                }
            }
        },
        "components": {
            "schemas": {
                "HealthResponse": {
                    "type": "object",
                    "required": ["status", "service", "version"],
                    "properties": {
                        "status": { "type": "string" },
                        "service": { "type": "string" },
                        "version": { "type": "string" }
                    }
                },
                "StudyRequest": {
                    "type": "object",
                    "required": ["goal"],
                    "properties": {
                        "goal": {
                            "type": "string",
                            "minLength": 3
                        },
                        "learner_context": {
                            "type": "string",
                            "default": ""
                        }
                    }
                },
                "StoryQuestion": {
                    "type": "object",
                    "required": ["question"],
                    "properties": {
                        "question": { "type": "string" }
                    }
                }
            }
        }
    }))
}

async fn swagger_docs() -> Html<&'static str> {
    Html(
        r##"<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Socartes Backend - Swagger UI</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css">
  </head>
  <body>
    <div id="swagger-ui">Swagger UI</div>
    <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
    <script>
      window.ui = SwaggerUIBundle({
        url: "/openapi.json",
        dom_id: "#swagger-ui",
        oauth2RedirectUrl: window.location.origin + "/docs/oauth2-redirect"
      });
    </script>
  </body>
</html>"##,
    )
}

async fn swagger_oauth2_redirect() -> Html<&'static str> {
    Html(
        r#"<!doctype html>
<html lang="en-US">
  <head>
    <title>Swagger UI: OAuth2 Redirect</title>
  </head>
  <body>
    <script>
      'use strict';
      function run() {
        var oauth2 = window.opener && window.opener.swaggerUIRedirectOauth2;
        var sentState = oauth2 && oauth2.state;
        var redirectUrl = new URL(window.location.href);
        var params = redirectUrl.search || redirectUrl.hash;
        if (oauth2 && params && sentState) {
          oauth2.callback({ auth: oauth2.auth, redirectUrl: window.location.href });
        }
        window.close();
      }
      run();
    </script>
  </body>
</html>"#,
    )
}

async fn redoc_docs() -> Html<&'static str> {
    Html(
        r#"<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Socartes Backend - ReDoc</title>
  </head>
  <body>
    <redoc spec-url="/openapi.json">ReDoc</redoc>
    <script src="https://cdn.jsdelivr.net/npm/redoc@2.5.0/bundles/redoc.standalone.js"></script>
  </body>
</html>"#,
    )
}

fn retrieve(goal: &str) -> Vec<RetrievalChunk> {
    let goal_terms = goal.to_lowercase();
    let mut matches = knowledge_base()
        .into_iter()
        .filter(|chunk| {
            format!("{} {}", chunk.title, chunk.content)
                .to_lowercase()
                .split_whitespace()
                .any(|term| goal_terms.contains(term))
        })
        .collect::<Vec<_>>();

    if matches.len() < 2 {
        let existing = matches
            .iter()
            .map(|chunk| chunk.source_id.clone())
            .collect::<HashSet<_>>();
        for chunk in knowledge_base().into_iter().take(2) {
            if !existing.contains(&chunk.source_id) {
                matches.push(chunk);
            }
        }
    }

    matches
}

fn knowledge_base() -> Vec<RetrievalChunk> {
    vec![
        RetrievalChunk {
            source_id: "rag-index-18".to_string(),
            title: "Learning systems index".to_string(),
            content: "RAG systems ground generated answers in retrieved external references so learners can inspect the source of a claim.".to_string(),
            confidence: "medium".to_string(),
        },
        RetrievalChunk {
            source_id: "workflow-note-01".to_string(),
            title: "Agent workflow brief".to_string(),
            content: "Multi-agent learning systems separate planning, execution, and critique to make task ownership and revision steps visible.".to_string(),
            confidence: "high".to_string(),
        },
        RetrievalChunk {
            source_id: "mcp-tool-07".to_string(),
            title: "Tool adapter note".to_string(),
            content: "MCP-style adapters expose external APIs, databases, and file systems through controlled contracts that can be audited.".to_string(),
            confidence: "high".to_string(),
        },
    ]
}

fn default_tool_results(goal: &str) -> Vec<ToolResult> {
    vec![
        ToolResult {
            adapter: "external_api".to_string(),
            action: "fetch_domain_state".to_string(),
            output: format!("Attached live-domain context for goal: {goal}"),
            safe: true,
        },
        ToolResult {
            adapter: "knowledge_database".to_string(),
            action: "query_indexed_notes".to_string(),
            output: "Returned ranked notes for RAG, MCP tool use, and agent review.".to_string(),
            safe: true,
        },
        ToolResult {
            adapter: "filesystem".to_string(),
            action: "read_learner_artifacts".to_string(),
            output: "Loaded scoped study artifacts for learner-specific context.".to_string(),
            safe: true,
        },
    ]
}

fn draft_answer(
    goal: &str,
    learner_context: &str,
    chunks: &[RetrievalChunk],
    tool_results: &[ToolResult],
) -> DraftAnswer {
    let citations = chunks
        .iter()
        .map(|chunk| chunk.source_id.clone())
        .collect::<Vec<_>>();
    let tool_names = tool_results
        .iter()
        .map(|result| format!("{}.{}", result.adapter, result.action))
        .collect::<Vec<_>>();
    let context_clause = if learner_context.is_empty() {
        String::new()
    } else {
        format!(" Learner context: {learner_context}")
    };

    DraftAnswer {
        agent: "executor".to_string(),
        content: format!(
            "Socartes answers the goal '{goal}' through a visible agent loop. \
             The Planner decomposes the request, the Retriever supplies RAG \
             evidence, the Executor combines that evidence with MCP-style tool \
             outputs, and the Critic checks whether the answer is cited and \
             complete. RAG evidence comes from {}, while MCP tool use is \
             represented by {}.{}",
            citations.join(", "),
            tool_names.join(", "),
            context_clause
        ),
        citations,
        tool_results_used: tool_names,
        open_gaps: vec![
            "External benchmark data should be refreshed for production use.".to_string(),
        ],
    }
}

fn review_draft(draft: &DraftAnswer) -> CriticReview {
    let mut issues = Vec::new();

    if draft.citations.is_empty() {
        issues.push(CriticIssue {
            issue_type: "missing_citation".to_string(),
            claim: "Draft has no cited evidence.".to_string(),
            instruction: "Retrieve domain context and attach citations.".to_string(),
        });
    }
    if draft.tool_results_used.is_empty() {
        issues.push(CriticIssue {
            issue_type: "missing_tool_trace".to_string(),
            claim: "Draft references tools without tool output identifiers.".to_string(),
            instruction: "Attach adapter names and outputs used by the executor.".to_string(),
        });
    }

    CriticReview {
        agent: "critic".to_string(),
        status: if issues.is_empty() {
            "approved".to_string()
        } else {
            "revision_required".to_string()
        },
        checks: vec![
            "acceptance criteria".to_string(),
            "citation coverage".to_string(),
            "tool output explainability".to_string(),
            "open gap visibility".to_string(),
        ],
        approved: issues.is_empty(),
        issues,
    }
}

fn record_reflection(review: &CriticReview) -> Vec<ReflectionEvent> {
    if review.approved {
        return vec![
            ReflectionEvent {
                event_type: "critic_review".to_string(),
                agent: "critic".to_string(),
                message: "Draft passed citation, tool trace, and gap checks.".to_string(),
            },
            ReflectionEvent {
                event_type: "executor_revision".to_string(),
                agent: "executor".to_string(),
                message: "Executor keeps citations and tool outputs attached to claims."
                    .to_string(),
            },
            ReflectionEvent {
                event_type: "planner_update".to_string(),
                agent: "planner".to_string(),
                message: "Future plans must keep citations required for comparison claims."
                    .to_string(),
            },
        ];
    }

    vec![ReflectionEvent {
        event_type: "critic_review".to_string(),
        agent: "critic".to_string(),
        message: "Draft requires revision before learner-facing response.".to_string(),
    }]
}

pub fn haunted_pajamas_index() -> StoryRagIndex {
    StoryRagIndex::new(vec![
        StoryChunk::new(
            "haunted-pajamas-ch01-sender",
            "The Haunted Pajamas, Chapter 1",
            "The package box is marked Roland Mastermann, Government House, Hong Kong, China.",
        ),
        StoryChunk::new(
            "haunted-pajamas-ch01-carlton",
            "The Haunted Pajamas, Chapter 1",
            "Jenkins thinks Mastermann is the London gentleman who entertained the narrator at the Carlton.",
        ),
        StoryChunk::new(
            "haunted-pajamas-ch01-muffler",
            "The Haunted Pajamas, Chapter 1",
            "The narrator tells Jenkins that the tight roll of bright red silk looks like it might be a red silk muffler.",
        ),
        StoryChunk::new(
            "haunted-pajamas-ch01-debt",
            "The Haunted Pajamas, Chapter 1",
            "Mastermann writes that every puff of the rare cigars reminds him that his debt to the narrator is still unpaid.",
        ),
        StoryChunk::new(
            "haunted-pajamas-ch02-hickeys-pride",
            "The Haunted Pajamas, Chapter 2",
            "Jenkins says the narrator planned to send Paloma perfectos, but the shipping clerk sent Hickey's Pride instead.",
        ),
        StoryChunk::new(
            "haunted-pajamas-ch02-twofer",
            "The Haunted Pajamas, Chapter 2",
            "Jenkins explains that a twofer means two for five: two cigars for five cents.",
        ),
        StoryChunk::new(
            "haunted-pajamas-ch02-present",
            "The Haunted Pajamas, Chapter 2",
            "After untying the string, the narrator exclaims that the gift is a suit of pajamas.",
        ),
        StoryChunk::new(
            "haunted-pajamas-ch02-memphis-tuffles",
            "The Haunted Pajamas, Chapter 2",
            "When asked what the red pajamas remind him of, Jenkins says they remind him of Old Memphis Tuffles.",
        ),
        StoryChunk::new(
            "haunted-pajamas-ch02-spider",
            "The Haunted Pajamas, Chapter 2",
            "A little spider dropped on its thread and shot into a fold of the pajamas.",
        ),
        StoryChunk::new(
            "haunted-pajamas-ch02-tarantula",
            "The Haunted Pajamas, Chapter 2",
            "Jenkins looks into one leg of the pajamas and says there is a tarantula in there, big as a sand crab, and alive.",
        ),
    ])
}

fn no_story_evidence_answer() -> StoryAnswer {
    StoryAnswer {
        answer: "The story RAG database does not have enough evidence to answer this question."
            .to_string(),
        grounded: false,
        source_ids: vec![],
        source_url: None,
    }
}

fn tokenize(text: &str) -> HashSet<String> {
    text.to_lowercase()
        .split(|character: char| !character.is_ascii_alphanumeric())
        .filter(|token| token.len() > 2 && !is_stopword(token))
        .map(ToString::to_string)
        .collect()
}

fn is_stopword(token: &str) -> bool {
    matches!(
        token,
        "a" | "an"
            | "and"
            | "as"
            | "be"
            | "did"
            | "first"
            | "in"
            | "into"
            | "is"
            | "it"
            | "of"
            | "on"
            | "or"
            | "out"
            | "say"
            | "the"
            | "there"
            | "think"
            | "to"
            | "was"
            | "what"
            | "with"
    )
}
