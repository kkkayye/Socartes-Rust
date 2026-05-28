use std::{
    collections::{BTreeMap, HashSet},
    env, fs,
    io::Write,
    path::{Path as FsPath, PathBuf},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    Json, Router,
    extract::{
        Multipart, Path, Query, State,
        ws::{Message, WebSocket, WebSocketUpgrade},
    },
    http::{StatusCode, header},
    response::{Html, IntoResponse},
    routing::{delete, get, patch, post, put},
};
use chrono::{SecondsFormat, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

pub const VERSION: &str = "0.1.0";
pub const PROJECT_GUTENBERG_HAUNTED_PAJAMAS_URL: &str =
    "https://www.gutenberg.org/ebooks/33780.txt.utf-8";

const MIN_RETRIEVAL_SCORE: usize = 2;
const BUILTIN_KNOWLEDGE_BASE: &str = "socartes-rust-rag";
const DEFAULT_RAG_PROVIDER: &str = "llamaindex";
const SUPPORTED_KNOWLEDGE_EXTENSIONS: &[&str] =
    &[".txt", ".md", ".markdown", ".pdf", ".json", ".csv"];
const TUTORBOT_EDITABLE_FILES: &[&str] = &[
    "SOUL.md",
    "USER.md",
    "TOOLS.md",
    "AGENTS.md",
    "HEARTBEAT.md",
];
const SECRET_MASK: &str = "***";

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

#[derive(Debug, Clone, Deserialize)]
struct TutorBotRecentQuery {
    limit: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
struct TutorBotHistoryQuery {
    limit: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
struct TutorBotDetailQuery {
    include_secrets: Option<bool>,
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
        self.run_with_retrieved_context(goal, learner_context, retrieve(goal))
    }

    pub fn run_with_retrieved_context(
        &self,
        goal: &str,
        learner_context: &str,
        retrieved_context: Vec<RetrievalChunk>,
    ) -> StudyTrace {
        let plan = self.plan();
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

#[derive(Debug, Clone)]
struct AppState {
    knowledge_root: Arc<PathBuf>,
    session_root: Arc<PathBuf>,
    book_root: Arc<PathBuf>,
    output_root: Arc<PathBuf>,
    settings_root: Arc<PathBuf>,
    tutorbot_root: Arc<PathBuf>,
}

impl AppState {
    fn new(knowledge_root: PathBuf) -> Self {
        let data_root = knowledge_root
            .parent()
            .unwrap_or_else(|| FsPath::new("."))
            .to_path_buf();
        let user_data_root = data_root.join("user");
        let session_root = env::var_os("SOCARTES_SESSION_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|| data_root.join("sessions"));
        let book_root = env::var_os("SOCARTES_BOOK_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|| user_data_root.join("workspace").join("book"));
        let output_root = env::var_os("SOCARTES_OUTPUT_ROOT")
            .map(PathBuf::from)
            .unwrap_or(user_data_root);
        let settings_root = env::var_os("SOCARTES_SETTINGS_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|| data_root.join("settings"));
        let tutorbot_root = env::var_os("SOCARTES_TUTORBOT_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|| data_root.join("tutorbot"));
        Self {
            knowledge_root: Arc::new(knowledge_root),
            session_root: Arc::new(session_root),
            book_root: Arc::new(book_root),
            output_root: Arc::new(output_root),
            settings_root: Arc::new(settings_root),
            tutorbot_root: Arc::new(tutorbot_root),
        }
    }
}

pub fn app() -> Router {
    app_with_knowledge_root(default_knowledge_root())
}

pub fn app_with_knowledge_root(path: impl Into<PathBuf>) -> Router {
    let state = AppState::new(path.into());
    Router::new()
        .route("/health", get(health))
        .route("/openapi.json", get(openapi_json))
        .route("/docs", get(swagger_docs))
        .route("/docs/oauth2-redirect", get(swagger_oauth2_redirect))
        .route("/redoc", get(redoc_docs))
        .route("/api/v1/agents", get(agents))
        .route("/api/v1/knowledge/list", get(knowledge_list))
        .route("/api/v1/knowledge/rag-providers", get(rag_providers))
        .route(
            "/api/v1/knowledge/supported-file-types",
            get(supported_file_types),
        )
        .route("/api/v1/knowledge/create", post(create_knowledge_base))
        .route(
            "/api/v1/knowledge/default/{name}",
            put(set_default_knowledge_base),
        )
        .route(
            "/api/v1/knowledge/tasks/{task_id}/stream",
            get(knowledge_task_stream),
        )
        .route("/api/v1/knowledge/{name}/files", get(list_knowledge_files))
        .route(
            "/api/v1/knowledge/{name}/files/{*file_path}",
            get(read_knowledge_file),
        )
        .route(
            "/api/v1/knowledge/{name}/upload",
            post(upload_knowledge_files),
        )
        .route(
            "/api/v1/knowledge/{name}/reindex",
            post(reindex_knowledge_base),
        )
        .route("/api/v1/knowledge/{name}", delete(delete_knowledge_base))
        .route("/api/v1/settings", get(get_settings))
        .route(
            "/api/v1/settings/catalog",
            get(get_settings_catalog).put(update_settings_catalog),
        )
        .route("/api/v1/settings/apply", post(apply_settings_catalog))
        .route("/api/v1/settings/ui", put(update_ui_settings_endpoint))
        .route("/api/v1/settings/theme", put(update_theme_endpoint))
        .route("/api/v1/settings/language", put(update_language_endpoint))
        .route("/api/v1/settings/reset", post(reset_settings_endpoint))
        .route("/api/v1/settings/themes", get(settings_themes))
        .route("/api/v1/settings/sidebar", get(settings_sidebar))
        .route(
            "/api/v1/settings/tests/{service}/start",
            post(start_settings_test),
        )
        .route(
            "/api/v1/settings/tests/{service}/{run_id}/events",
            get(settings_test_events),
        )
        .route(
            "/api/v1/settings/tests/{service}/{run_id}/cancel",
            post(cancel_settings_test),
        )
        .route("/api/v1/settings/llm-options", get(llm_options))
        .route("/api/v1/system/status", get(system_status))
        .route(
            "/api/v1/system/runtime-topology",
            get(system_runtime_topology),
        )
        .route("/api/v1/system/test/llm", post(system_test_llm))
        .route(
            "/api/v1/system/test/embeddings",
            post(system_test_embeddings),
        )
        .route("/api/v1/system/test/search", post(system_test_search))
        .route("/api/v1/tutorbot", get(list_tutorbots).post(start_tutorbot))
        .route("/api/v1/tutorbot/recent", get(recent_tutorbots))
        .route(
            "/api/v1/tutorbot/channels/schema",
            get(tutorbot_channel_schema),
        )
        .route(
            "/api/v1/tutorbot/souls",
            get(list_tutorbot_souls).post(create_tutorbot_soul),
        )
        .route(
            "/api/v1/tutorbot/souls/{soul_id}",
            get(get_tutorbot_soul)
                .put(update_tutorbot_soul)
                .delete(delete_tutorbot_soul),
        )
        .route(
            "/api/v1/tutorbot/{bot_id}",
            get(get_tutorbot)
                .patch(update_tutorbot)
                .delete(stop_tutorbot),
        )
        .route(
            "/api/v1/tutorbot/{bot_id}/destroy",
            delete(destroy_tutorbot),
        )
        .route("/api/v1/tutorbot/{bot_id}/files", get(list_tutorbot_files))
        .route(
            "/api/v1/tutorbot/{bot_id}/files/{filename}",
            get(read_tutorbot_file).put(write_tutorbot_file),
        )
        .route("/api/v1/tutorbot/{bot_id}/history", get(tutorbot_history))
        .route("/api/v1/tutorbot/{bot_id}/ws", get(tutorbot_ws))
        .route("/api/v1/sessions", get(list_sessions))
        .route("/api/v1/sessions/{session_id}", get(get_session))
        .route("/api/v1/sessions/{session_id}", patch(update_session_title))
        .route("/api/v1/sessions/{session_id}", delete(delete_session))
        .route(
            "/api/v1/sessions/{session_id}/quiz-results",
            post(record_quiz_results),
        )
        .route("/api/v1/chat/sessions", get(list_sessions))
        .route("/api/v1/chat/sessions/{session_id}", get(get_session))
        .route("/api/v1/internal/test-chat-turn", post(run_test_chat_turn))
        .route("/api/v1/ws", get(chat_ws))
        .route("/api/v1/book/health", get(book_service_health))
        .route("/api/v1/book/books", get(list_books).post(create_book))
        .route(
            "/api/v1/book/books/confirm-proposal",
            post(confirm_book_proposal),
        )
        .route("/api/v1/book/books/confirm-spine", post(confirm_book_spine))
        .route("/api/v1/book/books/compile-page", post(compile_book_page))
        .route(
            "/api/v1/book/books/regenerate-block",
            post(regenerate_book_block),
        )
        .route("/api/v1/book/books/insert-block", post(insert_book_block))
        .route("/api/v1/book/books/delete-block", post(delete_book_block))
        .route("/api/v1/book/books/move-block", post(move_book_block))
        .route(
            "/api/v1/book/books/change-block-type",
            post(change_book_block_type),
        )
        .route("/api/v1/book/books/deep-dive", post(book_deep_dive))
        .route(
            "/api/v1/book/books/quiz-attempt",
            post(record_book_quiz_attempt),
        )
        .route("/api/v1/book/books/supplement", post(supplement_book))
        .route(
            "/api/v1/book/books/page-chat-session",
            post(set_book_page_chat_session),
        )
        .route("/api/v1/book/books/rebuild", post(rebuild_book))
        .route(
            "/api/v1/book/books/{book_id}",
            get(get_book).delete(delete_book),
        )
        .route("/api/v1/book/books/{book_id}/spine", get(get_book_spine))
        .route(
            "/api/v1/book/books/{book_id}/pages/{page_id}",
            get(get_book_page),
        )
        .route("/api/v1/book/books/{book_id}/health", get(book_health))
        .route(
            "/api/v1/book/books/{book_id}/refresh-fingerprints",
            post(refresh_book_fingerprints),
        )
        .route("/api/v1/book/ws", get(book_ws))
        .route("/api/outputs/{*file_path}", get(read_output_file))
        .route("/api/v1/learn", post(learn))
        .route("/api/v1/story-rag/ask", post(ask_story_rag))
        .with_state(state)
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

async fn knowledge_list(State(state): State<AppState>) -> Json<Value> {
    Json(json!({
        "knowledge_bases": knowledge_base_summaries(&state)
    }))
}

async fn rag_providers() -> Json<Value> {
    Json(json!({
        "providers": [
            {
                "id": DEFAULT_RAG_PROVIDER,
                "name": "LlamaIndex",
                "description": "Local Rust file-backed retrieval over uploaded course documents."
            }
        ]
    }))
}

async fn supported_file_types() -> Json<Value> {
    Json(json!({
        "extensions": SUPPORTED_KNOWLEDGE_EXTENSIONS,
        "accept": SUPPORTED_KNOWLEDGE_EXTENSIONS.join(","),
        "max_file_size_bytes": 100 * 1024 * 1024,
        "max_pdf_size_bytes": 50 * 1024 * 1024
    }))
}

async fn list_knowledge_files(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    match knowledge_files(&state, &name) {
        Ok(files) => Json(json!({ "files": files })).into_response(),
        Err(error) => error.into_response(),
    }
}

async fn read_knowledge_file(
    State(state): State<AppState>,
    Path((name, file_path)): Path<(String, String)>,
) -> impl IntoResponse {
    if name == BUILTIN_KNOWLEDGE_BASE {
        if let Some(content) = builtin_knowledge_file(&file_path) {
            return (
                [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
                content,
            )
                .into_response();
        }
        return api_error(StatusCode::NOT_FOUND, "File not found").into_response();
    }

    let Some(filename) = safe_filename(&file_path) else {
        return api_error(StatusCode::BAD_REQUEST, "Invalid filename").into_response();
    };
    let path = knowledge_files_dir(&state, &name).join(filename);
    match fs::read(path) {
        Ok(bytes) => ([(header::CONTENT_TYPE, "application/octet-stream")], bytes).into_response(),
        Err(_) => api_error(StatusCode::NOT_FOUND, "File not found").into_response(),
    }
}

async fn create_knowledge_base(
    State(state): State<AppState>,
    multipart: Multipart,
) -> impl IntoResponse {
    match save_knowledge_base_from_multipart(&state, multipart, None).await {
        Ok((name, task_id)) => Json(json!({
            "task_id": task_id,
            "message": format!("Knowledge base {name} created")
        }))
        .into_response(),
        Err(error) => error.into_response(),
    }
}

async fn upload_knowledge_files(
    State(state): State<AppState>,
    Path(name): Path<String>,
    multipart: Multipart,
) -> impl IntoResponse {
    match save_knowledge_base_from_multipart(&state, multipart, Some(name)).await {
        Ok((name, task_id)) => Json(json!({
            "task_id": task_id,
            "message": format!("Files uploaded to {name}")
        }))
        .into_response(),
        Err(error) => error.into_response(),
    }
}

async fn set_default_knowledge_base(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    if !knowledge_base_exists(&state, &name) {
        return api_error(StatusCode::NOT_FOUND, "Knowledge base not found").into_response();
    }
    if let Err(error) = fs::create_dir_all(&*state.knowledge_root) {
        return api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("Failed to create knowledge root: {error}"),
        )
        .into_response();
    }
    match fs::write(default_knowledge_path(&state), &name) {
        Ok(()) => Json(json!({ "status": "ok", "default": name })).into_response(),
        Err(error) => api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("Failed to update default knowledge base: {error}"),
        )
        .into_response(),
    }
}

async fn reindex_knowledge_base(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    if !knowledge_base_exists(&state, &name) {
        return api_error(StatusCode::NOT_FOUND, "Knowledge base not found").into_response();
    }
    let task_id = format!("task-reindex-{}", unique_id());
    let metadata = json!({
        "last_indexed_at": now_seconds(),
        "last_indexed_action": "reindex"
    });
    let _ = write_knowledge_metadata(&state, &name, DEFAULT_RAG_PROVIDER, Some(metadata));
    Json(json!({
        "task_id": task_id,
        "message": format!("Re-index started for {name}")
    }))
    .into_response()
}

async fn knowledge_task_stream(Path(task_id): Path<String>) -> impl IntoResponse {
    let body = format!(
        "event: process_log\ndata: {{\"message\":\"Socartes Rust task {task_id} started\"}}\n\n\
event: progress\ndata: {{\"task_id\":\"{task_id}\",\"stage\":\"completed\",\"message\":\"Task completed\",\"current\":1,\"total\":1,\"progress_percent\":100}}\n\n\
event: complete\ndata: {{\"task_id\":\"{task_id}\",\"status\":\"completed\"}}\n\n"
    );
    ([(header::CONTENT_TYPE, "text/event-stream")], body)
}

async fn delete_knowledge_base(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    if name == BUILTIN_KNOWLEDGE_BASE {
        return api_error(
            StatusCode::BAD_REQUEST,
            "The built-in Socartes course cannot be deleted",
        )
        .into_response();
    }

    let path = knowledge_base_dir(&state, &name);
    if !path.exists() {
        return api_error(StatusCode::NOT_FOUND, "Knowledge base not found").into_response();
    }

    match fs::remove_dir_all(path) {
        Ok(()) => Json(json!({ "status": "ok" })).into_response(),
        Err(error) => api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("Failed to delete knowledge base: {error}"),
        )
        .into_response(),
    }
}

async fn book_service_health() -> Json<Value> {
    Json(json!({ "status": "healthy", "service": "book" }))
}

async fn list_books(State(state): State<AppState>) -> impl IntoResponse {
    match load_book_list(&state) {
        Ok(books) => Json(json!({ "books": books })).into_response(),
        Err(error) => error.into_response(),
    }
}

async fn get_book(State(state): State<AppState>, Path(book_id): Path<String>) -> impl IntoResponse {
    match load_book_detail(&state, &book_id) {
        Ok(detail) => Json(detail).into_response(),
        Err(error) => error.into_response(),
    }
}

async fn get_book_spine(
    State(state): State<AppState>,
    Path(book_id): Path<String>,
) -> impl IntoResponse {
    match load_book_json(&state, &book_id, "spine.json") {
        Ok(spine) => Json(json!({ "spine": spine })).into_response(),
        Err(error) => error.into_response(),
    }
}

async fn get_book_page(
    State(state): State<AppState>,
    Path((book_id, page_id)): Path<(String, String)>,
) -> impl IntoResponse {
    match load_book_page(&state, &book_id, &page_id) {
        Ok(page) => Json(json!({ "page": page })).into_response(),
        Err(error) => error.into_response(),
    }
}

async fn delete_book(
    State(state): State<AppState>,
    Path(book_id): Path<String>,
) -> impl IntoResponse {
    if !book_exists(&state, &book_id) {
        return api_error(StatusCode::NOT_FOUND, "Book not found").into_response();
    }
    match fs::remove_dir_all(book_dir(&state, &book_id)) {
        Ok(()) => Json(json!({ "deleted": true, "book_id": book_id })).into_response(),
        Err(error) => api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("Failed to delete book: {error}"),
        )
        .into_response(),
    }
}

async fn create_book(
    State(state): State<AppState>,
    Json(request): Json<Value>,
) -> impl IntoResponse {
    match create_book_record(&state, &request) {
        Ok((book, proposal)) => Json(json!({ "book": book, "proposal": proposal })).into_response(),
        Err(error) => error.into_response(),
    }
}

async fn confirm_book_proposal(
    State(state): State<AppState>,
    Json(request): Json<Value>,
) -> impl IntoResponse {
    let Some(book_id) = request["book_id"].as_str() else {
        return api_error(StatusCode::BAD_REQUEST, "book_id is required").into_response();
    };
    match confirm_book_proposal_record(&state, book_id, request.get("proposal")) {
        Ok((book, spine)) => Json(json!({ "book": book, "spine": spine })).into_response(),
        Err(error) => error.into_response(),
    }
}

async fn confirm_book_spine(
    State(state): State<AppState>,
    Json(request): Json<Value>,
) -> impl IntoResponse {
    let Some(book_id) = request["book_id"].as_str() else {
        return api_error(StatusCode::BAD_REQUEST, "book_id is required").into_response();
    };
    let auto_compile = request["auto_compile"].as_bool().unwrap_or(true);
    match confirm_book_spine_record(&state, book_id, request.get("spine"), auto_compile) {
        Ok(pages) => Json(json!({ "pages": pages })).into_response(),
        Err(error) => error.into_response(),
    }
}

async fn compile_book_page(
    State(state): State<AppState>,
    Json(request): Json<Value>,
) -> impl IntoResponse {
    match with_book_page_mut(&state, &request, |page| {
        page["status"] = json!("ready");
        page["updated_at"] = json!(now_seconds());
        Ok(page.clone())
    }) {
        Ok(page) => Json(json!({ "page": page })).into_response(),
        Err(error) => error.into_response(),
    }
}

async fn regenerate_book_block(
    State(state): State<AppState>,
    Json(request): Json<Value>,
) -> impl IntoResponse {
    match with_book_block_mut(&state, &request, |block| {
        block["status"] = json!("ready");
        block["updated_at"] = json!(now_seconds());
        block["metadata"]["regenerated"] = json!(true);
        if let Some(params) = request
            .get("params_override")
            .filter(|value| value.is_object())
        {
            merge_object_value(&mut block["params"], params);
        }
        Ok(block.clone())
    }) {
        Ok(block) => Json(json!({ "block": block })).into_response(),
        Err(error) => error.into_response(),
    }
}

async fn insert_book_block(
    State(state): State<AppState>,
    Json(request): Json<Value>,
) -> impl IntoResponse {
    let Some(book_id) = request["book_id"].as_str() else {
        return api_error(StatusCode::BAD_REQUEST, "book_id is required").into_response();
    };
    let Some(page_id) = request["page_id"].as_str() else {
        return api_error(StatusCode::BAD_REQUEST, "page_id is required").into_response();
    };
    let block_type = request["block_type"].as_str().unwrap_or("text");
    let params = request.get("params").cloned().unwrap_or_else(|| json!({}));
    let mut page = match load_book_page(&state, book_id, page_id) {
        Ok(page) => page,
        Err(error) => return error.into_response(),
    };
    let block = new_book_block(
        block_type,
        &format!("Inserted {}", block_type.replace('_', " ")),
        &params,
        request["compile_now"].as_bool().unwrap_or(true),
    );
    let position = request["position"].as_u64().map(|value| value as usize);
    if let Some(blocks) = page["blocks"].as_array_mut() {
        let index = position.unwrap_or(blocks.len()).min(blocks.len());
        blocks.insert(index, block.clone());
    } else {
        page["blocks"] = json!([block.clone()]);
    }
    page["updated_at"] = json!(now_seconds());
    match write_book_page(&state, book_id, &page) {
        Ok(()) => Json(json!({ "block": block })).into_response(),
        Err(error) => error.into_response(),
    }
}

async fn delete_book_block(
    State(state): State<AppState>,
    Json(request): Json<Value>,
) -> impl IntoResponse {
    match remove_or_move_book_block(&state, &request, None) {
        Ok(()) => Json(json!({ "ok": true })).into_response(),
        Err(error) => error.into_response(),
    }
}

async fn move_book_block(
    State(state): State<AppState>,
    Json(request): Json<Value>,
) -> impl IntoResponse {
    let new_position = request["new_position"].as_u64().map(|value| value as usize);
    match remove_or_move_book_block(&state, &request, new_position) {
        Ok(()) => Json(json!({ "ok": true })).into_response(),
        Err(error) => error.into_response(),
    }
}

async fn change_book_block_type(
    State(state): State<AppState>,
    Json(request): Json<Value>,
) -> impl IntoResponse {
    let new_type = request["new_type"].as_str().unwrap_or("text").to_string();
    match with_book_block_mut(&state, &request, |block| {
        block["type"] = json!(new_type);
        block["updated_at"] = json!(now_seconds());
        if let Some(params) = request
            .get("params_override")
            .filter(|value| value.is_object())
        {
            merge_object_value(&mut block["params"], params);
            merge_object_value(&mut block["payload"], params);
        }
        Ok(block.clone())
    }) {
        Ok(block) => Json(json!({ "block": block })).into_response(),
        Err(error) => error.into_response(),
    }
}

async fn book_deep_dive(
    State(state): State<AppState>,
    Json(request): Json<Value>,
) -> impl IntoResponse {
    let Some(book_id) = request["book_id"].as_str() else {
        return api_error(StatusCode::BAD_REQUEST, "book_id is required").into_response();
    };
    let parent_page_id = request["parent_page_id"].as_str().unwrap_or_default();
    let topic = request["topic"].as_str().unwrap_or("Deep dive");
    if !book_exists(&state, book_id) {
        return api_error(StatusCode::NOT_FOUND, "Book not found").into_response();
    }
    let page = new_book_page(
        book_id,
        &format!("deep-dive-{}", unique_id()),
        "deep-dive",
        topic,
        request["content_type"].as_str().unwrap_or("concept"),
        10_000,
        parent_page_id,
    );
    match write_book_page(&state, book_id, &page)
        .and_then(|()| refresh_book_counts(&state, book_id))
    {
        Ok(()) => Json(json!({ "page": page })).into_response(),
        Err(error) => error.into_response(),
    }
}

async fn record_book_quiz_attempt(
    State(state): State<AppState>,
    Json(request): Json<Value>,
) -> impl IntoResponse {
    let Some(book_id) = request["book_id"].as_str() else {
        return api_error(StatusCode::BAD_REQUEST, "book_id is required").into_response();
    };
    let mut progress =
        load_book_progress(&state, book_id).unwrap_or_else(|_| default_progress(book_id));
    let attempt = json!({
        "block_id": request["block_id"].as_str().unwrap_or_default(),
        "page_id": request["page_id"].as_str().unwrap_or_default(),
        "question_id": request["question_id"].as_str().unwrap_or_default(),
        "user_answer": request["user_answer"].as_str().unwrap_or_default(),
        "is_correct": request["is_correct"].as_bool().unwrap_or(false),
        "timestamp": now_seconds()
    });
    if let Some(attempts) = progress["quiz_attempts"].as_array_mut() {
        attempts.push(attempt);
    } else {
        progress["quiz_attempts"] = json!([attempt]);
    }
    let attempts = progress["quiz_attempts"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let correct = attempts
        .iter()
        .filter(|attempt| attempt["is_correct"].as_bool().unwrap_or(false))
        .count();
    progress["score"] = if attempts.is_empty() {
        json!(0.0)
    } else {
        json!(correct as f64 / attempts.len() as f64)
    };
    progress["updated_at"] = json!(now_seconds());
    match write_book_json(&state, book_id, "progress.json", &progress) {
        Ok(()) => Json(json!({ "progress": progress })).into_response(),
        Err(error) => error.into_response(),
    }
}

async fn supplement_book(
    State(state): State<AppState>,
    Json(request): Json<Value>,
) -> impl IntoResponse {
    let topic = request["topic"].as_str().unwrap_or("Supplement");
    let params = json!({ "topic": topic, "role": "supplement" });
    let block = new_book_block("callout", &format!("Supplement: {topic}"), &params, true);
    match append_book_block(&state, &request, block.clone()) {
        Ok(()) => Json(json!({ "block": block })).into_response(),
        Err(error) => error.into_response(),
    }
}

async fn set_book_page_chat_session(
    State(state): State<AppState>,
    Json(request): Json<Value>,
) -> impl IntoResponse {
    let Some(book_id) = request["book_id"].as_str() else {
        return api_error(StatusCode::BAD_REQUEST, "book_id is required").into_response();
    };
    let page_id = request["page_id"].as_str().unwrap_or_default();
    let session_id = request["session_id"].as_str().unwrap_or_default();
    let mut book = match load_book_manifest(&state, book_id) {
        Ok(book) => book,
        Err(error) => return error.into_response(),
    };
    ensure_object(&mut book["metadata"]);
    ensure_object(&mut book["metadata"]["page_chat_sessions"]);
    book["metadata"]["page_chat_sessions"][page_id] = json!(session_id);
    book["updated_at"] = json!(now_seconds());
    match write_book_manifest(&state, book_id, &book) {
        Ok(()) => Json(json!({ "book": book })).into_response(),
        Err(error) => error.into_response(),
    }
}

async fn rebuild_book(
    State(state): State<AppState>,
    Json(request): Json<Value>,
) -> impl IntoResponse {
    let Some(book_id) = request["book_id"].as_str() else {
        return api_error(StatusCode::BAD_REQUEST, "book_id is required").into_response();
    };
    match compile_pages_from_spine(&state, book_id, true) {
        Ok(pages) => Json(json!({ "pages": pages })).into_response(),
        Err(error) => error.into_response(),
    }
}

async fn book_health(
    State(state): State<AppState>,
    Path(book_id): Path<String>,
) -> impl IntoResponse {
    if !book_exists(&state, &book_id) {
        return api_error(StatusCode::NOT_FOUND, "Book not found").into_response();
    }
    let pages = load_book_pages(&state, &book_id).unwrap_or_default();
    Json(json!({
        "kb_drift": {
            "book_id": book_id,
            "has_drift": false,
            "new_kbs": [],
            "removed_kbs": [],
            "changed_kbs": [],
            "stale_page_ids": []
        },
        "log_health": {
            "book_id": book_id,
            "total_entries": pages.len(),
            "error_entries": 0,
            "block_failures": 0,
            "last_compile_at": null,
            "last_error_at": null,
            "repeated_failures": []
        }
    }))
    .into_response()
}

async fn refresh_book_fingerprints(
    State(state): State<AppState>,
    Path(book_id): Path<String>,
) -> impl IntoResponse {
    let mut book = match load_book_manifest(&state, &book_id) {
        Ok(book) => book,
        Err(error) => return error.into_response(),
    };
    let fingerprints = book["knowledge_bases"]
        .as_array()
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.as_str())
                .map(|name| (name.to_string(), json!(format!("rust-file-{name}"))))
                .collect::<serde_json::Map<_, _>>()
        })
        .unwrap_or_default();
    book["kb_fingerprints"] = Value::Object(fingerprints.clone());
    book["stale_page_ids"] = json!([]);
    book["updated_at"] = json!(now_seconds());
    match write_book_manifest(&state, &book_id, &book) {
        Ok(()) => Json(json!({
            "book_id": book_id,
            "kb_fingerprints": fingerprints,
            "stale_page_ids": []
        }))
        .into_response(),
        Err(error) => error.into_response(),
    }
}

async fn book_ws(ws: WebSocketUpgrade) -> impl IntoResponse {
    ws.on_upgrade(handle_book_ws)
}

async fn handle_book_ws(mut socket: WebSocket) {
    let connected = json!({
        "type": "connected",
        "source": "book",
        "stage": "ready",
        "content": "Socartes Rust book stream connected",
        "metadata": {}
    });
    if socket
        .send(Message::Text(connected.to_string().into()))
        .await
        .is_err()
    {
        return;
    }

    while let Some(message) = socket.recv().await {
        match message {
            Ok(Message::Text(text)) => {
                let message_type = serde_json::from_str::<Value>(&text)
                    .ok()
                    .and_then(|value| value["type"].as_str().map(ToString::to_string))
                    .unwrap_or_else(|| "unknown".to_string());
                let response = json!({
                    "type": "error",
                    "source": "book",
                    "stage": "unsupported",
                    "content": format!("Book WebSocket action {message_type} should use the REST compatibility endpoints in this Rust build."),
                    "metadata": { "request_type": message_type }
                });
                if socket
                    .send(Message::Text(response.to_string().into()))
                    .await
                    .is_err()
                {
                    return;
                }
            }
            Ok(Message::Close(_)) | Err(_) => return,
            _ => {}
        }
    }
}

async fn read_output_file(
    State(state): State<AppState>,
    Path(file_path): Path<String>,
) -> impl IntoResponse {
    let Some(relative_path) = safe_relative_output_path(&file_path) else {
        return api_error(StatusCode::BAD_REQUEST, "Invalid output path").into_response();
    };
    if !is_allowed_output_path(&relative_path) {
        return api_error(StatusCode::FORBIDDEN, "Output path is not public").into_response();
    }
    let path = state.output_root.join(&relative_path);
    let Ok(root) = fs::canonicalize(&*state.output_root) else {
        return api_error(StatusCode::NOT_FOUND, "Output root not found").into_response();
    };
    let Ok(canonical) = fs::canonicalize(&path) else {
        return api_error(StatusCode::NOT_FOUND, "Output file not found").into_response();
    };
    if !canonical.starts_with(root) || !canonical.is_file() {
        return api_error(StatusCode::FORBIDDEN, "Output path is not public").into_response();
    }
    match fs::read(&canonical) {
        Ok(bytes) => (
            [(header::CONTENT_TYPE, output_mime_type(&relative_path))],
            bytes,
        )
            .into_response(),
        Err(_) => api_error(StatusCode::NOT_FOUND, "Output file not found").into_response(),
    }
}

type ApiError = (StatusCode, Json<Value>);

fn api_error(status: StatusCode, detail: &str) -> ApiError {
    (status, Json(json!({ "detail": detail })))
}

fn book_dir(state: &AppState, book_id: &str) -> PathBuf {
    let component = safe_storage_component(book_id).unwrap_or_else(|| "invalid".to_string());
    state.book_root.join(format!("book_{component}"))
}

fn book_pages_dir(state: &AppState, book_id: &str) -> PathBuf {
    book_dir(state, book_id).join("pages")
}

fn book_exists(state: &AppState, book_id: &str) -> bool {
    book_dir(state, book_id).join("manifest.json").is_file()
}

fn load_book_list(state: &AppState) -> Result<Vec<Value>, ApiError> {
    let mut books = Vec::new();
    let Ok(entries) = fs::read_dir(&*state.book_root) else {
        return Ok(books);
    };

    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(dirname) = path.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        let Some(book_id) = dirname.strip_prefix("book_") else {
            continue;
        };
        if let Ok(book) = load_book_manifest(state, book_id) {
            books.push(book);
        }
    }

    books.sort_by(|left, right| {
        right["updated_at"]
            .as_f64()
            .partial_cmp(&left["updated_at"].as_f64())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(books)
}

fn load_book_manifest(state: &AppState, book_id: &str) -> Result<Value, ApiError> {
    load_book_json(state, book_id, "manifest.json").map(normalize_book_manifest)
}

fn write_book_manifest(state: &AppState, book_id: &str, book: &Value) -> Result<(), ApiError> {
    write_book_json(state, book_id, "manifest.json", book)
}

fn load_book_detail(state: &AppState, book_id: &str) -> Result<Value, ApiError> {
    let book = load_book_manifest(state, book_id)?;
    let spine = load_book_json(state, book_id, "spine.json").unwrap_or(Value::Null);
    let pages = load_book_pages(state, book_id).unwrap_or_default();
    let progress = load_book_progress(state, book_id).unwrap_or_else(|_| default_progress(book_id));
    Ok(json!({
        "book": book,
        "spine": spine,
        "pages": pages,
        "progress": progress
    }))
}

fn load_book_json(state: &AppState, book_id: &str, filename: &str) -> Result<Value, ApiError> {
    if safe_storage_component(book_id).is_none() {
        return Err(api_error(StatusCode::BAD_REQUEST, "Invalid book id"));
    }
    let path = book_dir(state, book_id).join(filename);
    let text =
        fs::read_to_string(path).map_err(|_| api_error(StatusCode::NOT_FOUND, "Book not found"))?;
    serde_json::from_str(&text).map_err(|error| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("Invalid book JSON: {error}"),
        )
    })
}

fn write_book_json(
    state: &AppState,
    book_id: &str,
    filename: &str,
    value: &Value,
) -> Result<(), ApiError> {
    let dir = book_dir(state, book_id);
    fs::create_dir_all(&dir).map_err(|error| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("Failed to create book directory: {error}"),
        )
    })?;
    let bytes = serde_json::to_vec_pretty(value).map_err(|error| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("Failed to serialize book JSON: {error}"),
        )
    })?;
    fs::write(dir.join(filename), bytes).map_err(|error| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("Failed to write book JSON: {error}"),
        )
    })
}

fn load_book_pages(state: &AppState, book_id: &str) -> Result<Vec<Value>, ApiError> {
    if !book_exists(state, book_id) {
        return Err(api_error(StatusCode::NOT_FOUND, "Book not found"));
    }
    let mut pages = Vec::new();
    if let Ok(entries) = fs::read_dir(book_pages_dir(state, book_id)) {
        for entry in entries.filter_map(Result::ok) {
            let path = entry.path();
            if path.extension().and_then(|value| value.to_str()) != Some("json") {
                continue;
            }
            if let Ok(text) = fs::read_to_string(path)
                && let Ok(page) = serde_json::from_str::<Value>(&text)
            {
                pages.push(page);
            }
        }
    }
    pages.sort_by(|left, right| {
        let left_key = (
            left["order"].as_i64().unwrap_or_default(),
            left["created_at"].as_f64().unwrap_or_default() as i64,
        );
        let right_key = (
            right["order"].as_i64().unwrap_or_default(),
            right["created_at"].as_f64().unwrap_or_default() as i64,
        );
        left_key.cmp(&right_key)
    });
    Ok(pages)
}

fn load_book_page(state: &AppState, book_id: &str, page_id: &str) -> Result<Value, ApiError> {
    let Some(filename) = safe_filename(&format!("{page_id}.json")) else {
        return Err(api_error(StatusCode::BAD_REQUEST, "Invalid page id"));
    };
    let path = book_pages_dir(state, book_id).join(filename);
    let text =
        fs::read_to_string(path).map_err(|_| api_error(StatusCode::NOT_FOUND, "Page not found"))?;
    serde_json::from_str(&text).map_err(|error| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("Invalid page JSON: {error}"),
        )
    })
}

fn write_book_page(state: &AppState, book_id: &str, page: &Value) -> Result<(), ApiError> {
    let Some(page_id) = page["id"].as_str().and_then(safe_storage_component) else {
        return Err(api_error(StatusCode::BAD_REQUEST, "Page id is required"));
    };
    let dir = book_pages_dir(state, book_id);
    fs::create_dir_all(&dir).map_err(|error| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("Failed to create pages directory: {error}"),
        )
    })?;
    let bytes = serde_json::to_vec_pretty(page).map_err(|error| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("Failed to serialize page JSON: {error}"),
        )
    })?;
    fs::write(dir.join(format!("{page_id}.json")), bytes).map_err(|error| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("Failed to write page JSON: {error}"),
        )
    })
}

fn load_book_progress(state: &AppState, book_id: &str) -> Result<Value, ApiError> {
    load_book_json(state, book_id, "progress.json")
}

fn create_book_record(state: &AppState, request: &Value) -> Result<(Value, Value), ApiError> {
    let user_intent = request["user_intent"]
        .as_str()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| api_error(StatusCode::BAD_REQUEST, "user_intent is required"))?;
    let book_id = format!("bk_{}", unique_id());
    let now = now_seconds();
    let proposal = request
        .get("proposal")
        .filter(|value| value.is_object())
        .cloned()
        .unwrap_or_else(|| default_book_proposal(user_intent));
    let title = proposal["title"]
        .as_str()
        .map(ToString::to_string)
        .unwrap_or_else(|| title_from_content(user_intent));
    let description = proposal["description"]
        .as_str()
        .unwrap_or("Socartes Rust generated book")
        .to_string();
    let knowledge_bases = as_string_array(&request["knowledge_bases"]);
    let language = request["language"].as_str().unwrap_or("en");
    let book = json!({
        "id": book_id,
        "title": title,
        "description": description,
        "status": "draft",
        "proposal": proposal,
        "knowledge_bases": knowledge_bases,
        "language": language,
        "page_count": 0,
        "chapter_count": 0,
        "created_at": now,
        "updated_at": now,
        "metadata": { "page_chat_sessions": {} },
        "kb_fingerprints": {},
        "stale_page_ids": []
    });
    fs::create_dir_all(book_pages_dir(state, book["id"].as_str().unwrap())).map_err(|error| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("Failed to create book directory: {error}"),
        )
    })?;
    write_book_manifest(state, book["id"].as_str().unwrap(), &book)?;
    write_book_json(state, book["id"].as_str().unwrap(), "inputs.json", request)?;
    Ok((book.clone(), book["proposal"].clone()))
}

fn confirm_book_proposal_record(
    state: &AppState,
    book_id: &str,
    proposal: Option<&Value>,
) -> Result<(Value, Value), ApiError> {
    let mut book = load_book_manifest(state, book_id)?;
    let proposal = proposal
        .filter(|value| value.is_object())
        .cloned()
        .or_else(|| {
            book.get("proposal")
                .filter(|value| value.is_object())
                .cloned()
        })
        .unwrap_or_else(|| {
            default_book_proposal(book["title"].as_str().unwrap_or("Socartes book"))
        });
    book["proposal"] = proposal.clone();
    book["title"] = proposal["title"]
        .as_str()
        .map(|title| json!(title))
        .unwrap_or_else(|| book["title"].clone());
    book["description"] = proposal["description"]
        .as_str()
        .map(|description| json!(description))
        .unwrap_or_else(|| book["description"].clone());
    book["status"] = json!("spine_ready");
    book["updated_at"] = json!(now_seconds());
    let spine = default_spine(book_id, &proposal);
    book["chapter_count"] = json!(
        spine["chapters"]
            .as_array()
            .map(Vec::len)
            .unwrap_or_default()
    );
    write_book_manifest(state, book_id, &book)?;
    write_book_json(state, book_id, "spine.json", &spine)?;
    write_book_json(state, book_id, "progress.json", &default_progress(book_id))?;
    Ok((book, spine))
}

fn confirm_book_spine_record(
    state: &AppState,
    book_id: &str,
    spine: Option<&Value>,
    auto_compile: bool,
) -> Result<Vec<Value>, ApiError> {
    let mut book = load_book_manifest(state, book_id)?;
    let spine = spine
        .filter(|value| value.is_object())
        .cloned()
        .or_else(|| load_book_json(state, book_id, "spine.json").ok())
        .unwrap_or_else(|| default_spine(book_id, &book["proposal"]));
    write_book_json(state, book_id, "spine.json", &spine)?;
    let pages = if auto_compile {
        compile_pages_from_spine(state, book_id, true)?
    } else {
        load_book_pages(state, book_id).unwrap_or_default()
    };
    book["status"] = json!(if auto_compile { "ready" } else { "spine_ready" });
    book["page_count"] = json!(pages.len());
    book["chapter_count"] = json!(
        spine["chapters"]
            .as_array()
            .map(Vec::len)
            .unwrap_or_default()
    );
    book["updated_at"] = json!(now_seconds());
    write_book_manifest(state, book_id, &book)?;
    Ok(pages)
}

fn compile_pages_from_spine(
    state: &AppState,
    book_id: &str,
    mark_ready: bool,
) -> Result<Vec<Value>, ApiError> {
    let spine = load_book_json(state, book_id, "spine.json")?;
    let mut pages = Vec::new();
    for chapter in spine["chapters"].as_array().cloned().unwrap_or_default() {
        let page_ids = chapter["page_ids"]
            .as_array()
            .cloned()
            .unwrap_or_else(|| vec![json!(format!("page-{}", pages.len() + 1))]);
        for page_id in page_ids {
            let page_id = page_id.as_str().unwrap_or("page-1");
            let mut page = load_book_page(state, book_id, page_id).unwrap_or_else(|_| {
                new_book_page(
                    book_id,
                    page_id,
                    chapter["id"].as_str().unwrap_or("chapter-1"),
                    chapter["title"].as_str().unwrap_or("Socartes page"),
                    chapter["content_type"].as_str().unwrap_or("overview"),
                    chapter["order"].as_i64().unwrap_or(1),
                    "",
                )
            });
            if mark_ready {
                page["status"] = json!("ready");
                page["updated_at"] = json!(now_seconds());
            }
            write_book_page(state, book_id, &page)?;
            pages.push(page);
        }
    }
    refresh_book_counts(state, book_id)?;
    Ok(pages)
}

fn refresh_book_counts(state: &AppState, book_id: &str) -> Result<(), ApiError> {
    let mut book = load_book_manifest(state, book_id)?;
    let pages = load_book_pages(state, book_id).unwrap_or_default();
    let spine = load_book_json(state, book_id, "spine.json").unwrap_or(Value::Null);
    book["page_count"] = json!(pages.len());
    book["chapter_count"] = json!(
        spine["chapters"]
            .as_array()
            .map(Vec::len)
            .unwrap_or_default()
    );
    book["updated_at"] = json!(now_seconds());
    write_book_manifest(state, book_id, &book)
}

fn normalize_book_manifest(mut book: Value) -> Value {
    let now = now_seconds();
    if book["metadata"].is_null() {
        book["metadata"] = json!({ "page_chat_sessions": {} });
    }
    if book["kb_fingerprints"].is_null() {
        book["kb_fingerprints"] = json!({});
    }
    if book["stale_page_ids"].is_null() {
        book["stale_page_ids"] = json!([]);
    }
    if book["created_at"].is_null() {
        book["created_at"] = json!(now);
    }
    if book["updated_at"].is_null() {
        book["updated_at"] = json!(now);
    }
    book
}

fn default_book_proposal(user_intent: &str) -> Value {
    let title = title_from_content(user_intent);
    json!({
        "title": title,
        "description": format!("A Socartes study book about {title}."),
        "scope": "short",
        "target_level": "intermediate",
        "estimated_chapters": 1,
        "rationale": "Generated by the Rust compatibility BookEngine from the learner intent."
    })
}

fn default_spine(book_id: &str, proposal: &Value) -> Value {
    let title = proposal["title"].as_str().unwrap_or("Socartes chapter");
    json!({
        "book_id": book_id,
        "chapters": [{
            "id": "chapter-1",
            "title": title,
            "learning_objectives": [
                "Connect the learner intent to retrieved course evidence",
                "Explain the Planner, Executor, Critic, and Reflection loop"
            ],
            "content_type": "overview",
            "source_anchors": [],
            "prerequisites": [],
            "page_ids": ["page-1"],
            "summary": proposal["description"].as_str().unwrap_or("Socartes generated chapter"),
            "order": 1
        }],
        "version": 1,
        "updated_at": now_seconds(),
        "concept_graph": {
            "nodes": [],
            "edges": []
        },
        "exploration_summary": proposal["rationale"].as_str().unwrap_or("")
    })
}

fn default_progress(book_id: &str) -> Value {
    json!({
        "book_id": book_id,
        "current_page_id": "",
        "visited_page_ids": [],
        "bookmarked_page_ids": [],
        "quiz_attempts": [],
        "weak_chapters": [],
        "score": 0.0,
        "updated_at": now_seconds()
    })
}

fn new_book_page(
    book_id: &str,
    page_id: &str,
    chapter_id: &str,
    title: &str,
    content_type: &str,
    order: i64,
    parent_page_id: &str,
) -> Value {
    let now = now_seconds();
    json!({
        "id": page_id,
        "book_id": book_id,
        "chapter_id": chapter_id,
        "title": title,
        "learning_objectives": ["Understand the Socartes learning workflow"],
        "content_type": content_type,
        "status": "ready",
        "order": order,
        "blocks": [
            new_book_block(
                "text",
                "Socartes learning trace",
                &json!({ "body": format!("{title}\n\nThis Rust page is backed by file-backed Book data and can be edited through the Book API.") }),
                true
            )
        ],
        "links": [],
        "parent_page_id": parent_page_id,
        "error": "",
        "created_at": now,
        "updated_at": now
    })
}

fn new_book_block(block_type: &str, title: &str, params: &Value, ready: bool) -> Value {
    let now = now_seconds();
    let body = params["body"]
        .as_str()
        .or_else(|| params["topic"].as_str())
        .unwrap_or(title);
    json!({
        "id": format!("block-{}", unique_id()),
        "type": block_type,
        "status": if ready { "ready" } else { "pending" },
        "title": title,
        "params": params,
        "payload": {
            "body": body,
            "variant": "info"
        },
        "source_anchors": [],
        "metadata": {},
        "error": "",
        "created_at": now,
        "updated_at": now
    })
}

fn with_book_page_mut<F>(
    state: &AppState,
    request: &Value,
    mut update: F,
) -> Result<Value, ApiError>
where
    F: FnMut(&mut Value) -> Result<Value, ApiError>,
{
    let book_id = request["book_id"]
        .as_str()
        .ok_or_else(|| api_error(StatusCode::BAD_REQUEST, "book_id is required"))?;
    let page_id = request["page_id"]
        .as_str()
        .ok_or_else(|| api_error(StatusCode::BAD_REQUEST, "page_id is required"))?;
    let mut page = load_book_page(state, book_id, page_id)?;
    let output = update(&mut page)?;
    write_book_page(state, book_id, &page)?;
    refresh_book_counts(state, book_id)?;
    Ok(output)
}

fn with_book_block_mut<F>(
    state: &AppState,
    request: &Value,
    mut update: F,
) -> Result<Value, ApiError>
where
    F: FnMut(&mut Value) -> Result<Value, ApiError>,
{
    with_book_page_mut(state, request, |page| {
        let block_id = request["block_id"]
            .as_str()
            .ok_or_else(|| api_error(StatusCode::BAD_REQUEST, "block_id is required"))?;
        let blocks = page["blocks"]
            .as_array_mut()
            .ok_or_else(|| api_error(StatusCode::NOT_FOUND, "Block not found"))?;
        let index = blocks
            .iter()
            .position(|block| block["id"] == block_id)
            .ok_or_else(|| api_error(StatusCode::NOT_FOUND, "Block not found"))?;
        update(&mut blocks[index])
    })
}

fn append_book_block(state: &AppState, request: &Value, block: Value) -> Result<(), ApiError> {
    with_book_page_mut(state, request, |page| {
        if let Some(blocks) = page["blocks"].as_array_mut() {
            blocks.push(block.clone());
        } else {
            page["blocks"] = json!([block.clone()]);
        }
        page["updated_at"] = json!(now_seconds());
        Ok(Value::Null)
    })
    .map(|_| ())
}

fn remove_or_move_book_block(
    state: &AppState,
    request: &Value,
    new_position: Option<usize>,
) -> Result<(), ApiError> {
    with_book_page_mut(state, request, |page| {
        let block_id = request["block_id"]
            .as_str()
            .ok_or_else(|| api_error(StatusCode::BAD_REQUEST, "block_id is required"))?;
        let blocks = page["blocks"]
            .as_array_mut()
            .ok_or_else(|| api_error(StatusCode::NOT_FOUND, "Block not found"))?;
        let index = blocks
            .iter()
            .position(|block| block["id"] == block_id)
            .ok_or_else(|| api_error(StatusCode::NOT_FOUND, "Block not found"))?;
        let block = blocks.remove(index);
        if let Some(position) = new_position {
            blocks.insert(position.min(blocks.len()), block);
        }
        page["updated_at"] = json!(now_seconds());
        Ok(Value::Null)
    })
    .map(|_| ())
}

fn ensure_object(value: &mut Value) {
    if !value.is_object() {
        *value = json!({});
    }
}

fn merge_object_value(target: &mut Value, source: &Value) {
    ensure_object(target);
    let Some(target) = target.as_object_mut() else {
        return;
    };
    if let Some(source) = source.as_object() {
        for (key, value) in source {
            target.insert(key.clone(), value.clone());
        }
    }
}

fn safe_relative_output_path(value: &str) -> Option<PathBuf> {
    if value.trim().is_empty() || value.contains('\0') {
        return None;
    }
    let path = FsPath::new(value);
    if path.is_absolute() {
        return None;
    }
    let mut clean = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::Normal(part) => clean.push(part),
            _ => return None,
        }
    }
    if clean.as_os_str().is_empty() {
        None
    } else {
        Some(clean)
    }
}

fn is_allowed_output_path(path: &FsPath) -> bool {
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if matches!(
        extension.as_str(),
        "json" | "sqlite" | "db" | "md" | "yaml" | "yml" | "py" | "log"
    ) {
        return false;
    }
    let normalized = path.to_string_lossy().replace('\\', "/");
    normalized.starts_with("workspace/co-writer/audio/")
        || normalized.starts_with("workspace/chat/_detached_code_execution/")
        || (normalized.starts_with("workspace/chat/")
            && (normalized.contains("/artifacts/") || normalized.contains("/code_runs/")))
}

fn output_mime_type(path: &FsPath) -> &'static str {
    match path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "html" | "htm" => "text/html; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "js" => "application/javascript; charset=utf-8",
        "txt" => "text/plain; charset=utf-8",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "svg" => "image/svg+xml",
        "mp4" => "video/mp4",
        "mp3" => "audio/mpeg",
        "wav" => "audio/wav",
        _ => "application/octet-stream",
    }
}

fn settings_path(state: &AppState, filename: &str) -> PathBuf {
    state.settings_root.join(filename)
}

fn read_settings_json(state: &AppState, filename: &str) -> Option<Value> {
    let text = fs::read_to_string(settings_path(state, filename)).ok()?;
    serde_json::from_str(&text).ok()
}

fn write_settings_json(state: &AppState, filename: &str, value: &Value) -> Result<(), ApiError> {
    fs::create_dir_all(&*state.settings_root).map_err(|error| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("Failed to create settings directory: {error}"),
        )
    })?;
    let bytes = serde_json::to_vec_pretty(value).map_err(|error| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("Failed to serialize settings: {error}"),
        )
    })?;
    fs::write(settings_path(state, filename), bytes).map_err(|error| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("Failed to write settings: {error}"),
        )
    })
}

fn load_ui_settings(state: &AppState) -> Value {
    let mut defaults = default_ui_settings();
    if let Some(saved) = read_settings_json(state, "ui.json") {
        merge_object_value(&mut defaults, &saved);
    }
    defaults
}

fn load_settings_catalog(state: &AppState) -> Value {
    read_settings_json(state, "catalog.json").unwrap_or_else(default_settings_catalog)
}

fn default_ui_settings() -> Value {
    json!({
        "theme": "light",
        "language": "en",
        "sidebar_description": "✨ Data Intelligence Lab @ HKU",
        "sidebar_nav_order": {
            "start": ["/", "/history", "/knowledge", "/notebook"],
            "learnResearch": ["/question", "/solver", "/research", "/co_writer"]
        }
    })
}

fn default_settings_catalog() -> Value {
    json!({
        "version": 1,
        "services": {
            "llm": {
                "active_profile_id": "socartes-rust",
                "active_model_id": "deterministic-agent-loop",
                "profiles": [{
                    "id": "socartes-rust",
                    "name": "Socartes Rust",
                    "binding": "openai",
                    "base_url": "http://127.0.0.1:8810/v1",
                    "api_key": "",
                    "api_version": "",
                    "extra_headers": {},
                    "models": [{
                        "id": "deterministic-agent-loop",
                        "name": "Deterministic Agent Loop",
                        "model": "deterministic-agent-loop",
                        "context_window": "8192",
                        "context_window_source": "rust-default"
                    }]
                }]
            },
            "embedding": {
                "active_profile_id": "socartes-rust-embedding",
                "active_model_id": "deterministic-embedding",
                "profiles": [{
                    "id": "socartes-rust-embedding",
                    "name": "Socartes Rust Embedding",
                    "binding": "openai",
                    "base_url": "http://127.0.0.1:8810/v1",
                    "api_key": "",
                    "api_version": "",
                    "extra_headers": {},
                    "models": [{
                        "id": "deterministic-embedding",
                        "name": "Deterministic Embedding",
                        "model": "deterministic-embedding",
                        "dimension": "3072",
                        "send_dimensions": true,
                        "supported_dimensions": "1536,3072"
                    }]
                }]
            },
            "search": {
                "active_profile_id": "duckduckgo-local",
                "profiles": [{
                    "id": "duckduckgo-local",
                    "name": "DuckDuckGo Local",
                    "provider": "duckduckgo",
                    "base_url": "",
                    "api_key": "",
                    "api_version": "",
                    "proxy": "",
                    "max_results": 5,
                    "models": []
                }]
            }
        }
    })
}

fn settings_provider_choices() -> Value {
    json!({
        "llm": [
            { "value": "openai", "label": "OpenAI-compatible", "base_url": "http://127.0.0.1:8810/v1" },
            { "value": "custom", "label": "Custom (OpenAI API)", "base_url": "" },
            { "value": "custom_anthropic", "label": "Custom (Anthropic API)", "base_url": "" }
        ],
        "embedding": [
            { "value": "openai", "label": "OpenAI-compatible", "base_url": "http://127.0.0.1:8810/v1", "default_dim": "3072" },
            { "value": "local", "label": "Local deterministic", "base_url": "", "default_dim": "3072" }
        ],
        "search": [
            { "value": "brave", "label": "Brave", "base_url": "" },
            { "value": "tavily", "label": "Tavily", "base_url": "" },
            { "value": "jina", "label": "Jina", "base_url": "" },
            { "value": "searxng", "label": "SearXNG", "base_url": "" },
            { "value": "duckduckgo", "label": "DuckDuckGo", "base_url": "" },
            { "value": "perplexity", "label": "Perplexity", "base_url": "" }
        ]
    })
}

fn render_settings_env(catalog: &Value) -> Value {
    let llm_model = active_catalog_model_name(catalog, "llm")
        .unwrap_or_else(|| "deterministic-agent-loop".to_string());
    let embedding_model = active_catalog_model_name(catalog, "embedding")
        .unwrap_or_else(|| "deterministic-embedding".to_string());
    let search_provider =
        active_search_provider(catalog).unwrap_or_else(|| "duckduckgo".to_string());
    json!({
        "SOCARTES_LLM_MODEL": llm_model,
        "SOCARTES_EMBEDDING_MODEL": embedding_model,
        "SOCARTES_SEARCH_PROVIDER": search_provider
    })
}

fn active_catalog_model_name(catalog: &Value, service_name: &str) -> Option<String> {
    let service = &catalog["services"][service_name];
    let active_profile_id = service["active_profile_id"].as_str();
    let profile = service["profiles"]
        .as_array()?
        .iter()
        .find(|profile| Some(profile["id"].as_str().unwrap_or_default()) == active_profile_id)
        .or_else(|| service["profiles"].as_array()?.first())?;
    let active_model_id = service["active_model_id"].as_str();
    let model = profile["models"]
        .as_array()?
        .iter()
        .find(|model| Some(model["id"].as_str().unwrap_or_default()) == active_model_id)
        .or_else(|| profile["models"].as_array()?.first())?;
    model["model"]
        .as_str()
        .or_else(|| model["name"].as_str())
        .map(ToString::to_string)
}

fn active_search_provider(catalog: &Value) -> Option<String> {
    let service = &catalog["services"]["search"];
    let active_profile_id = service["active_profile_id"].as_str();
    let profile = service["profiles"]
        .as_array()?
        .iter()
        .find(|profile| Some(profile["id"].as_str().unwrap_or_default()) == active_profile_id)
        .or_else(|| service["profiles"].as_array()?.first())?;
    profile["provider"].as_str().map(ToString::to_string)
}

fn default_knowledge_root() -> PathBuf {
    env::var_os("SOCARTES_KNOWLEDGE_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            env::current_dir()
                .unwrap_or_else(|_| PathBuf::from("."))
                .join("data")
                .join("knowledge")
        })
}

fn safe_storage_component(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty()
        || trimmed == "."
        || trimmed == ".."
        || trimmed.contains('/')
        || trimmed.contains('\\')
        || trimmed.contains('\0')
    {
        return None;
    }
    Some(trimmed.to_string())
}

fn safe_filename(value: &str) -> Option<String> {
    let path = FsPath::new(value);
    let filename = path.file_name()?.to_str()?;
    safe_storage_component(filename)
}

fn tutorbot_dir(state: &AppState, bot_id: &str) -> PathBuf {
    let component = safe_storage_component(bot_id).unwrap_or_else(|| "invalid".to_string());
    state.tutorbot_root.join(component)
}

fn tutorbot_config_path(state: &AppState, bot_id: &str) -> PathBuf {
    tutorbot_dir(state, bot_id).join("config.json")
}

fn tutorbot_workspace_dir(state: &AppState, bot_id: &str) -> PathBuf {
    tutorbot_dir(state, bot_id).join("workspace")
}

fn tutorbot_sessions_dir(state: &AppState, bot_id: &str) -> PathBuf {
    tutorbot_workspace_dir(state, bot_id).join("sessions")
}

fn tutorbot_exists(state: &AppState, bot_id: &str) -> bool {
    tutorbot_config_path(state, bot_id).is_file()
}

fn read_tutorbot_config(state: &AppState, bot_id: &str) -> Option<Value> {
    let text = fs::read_to_string(tutorbot_config_path(state, bot_id)).ok()?;
    let parsed = serde_json::from_str::<Value>(&text).ok()?;
    Some(normalize_tutorbot_config(bot_id, parsed))
}

fn write_tutorbot_config(state: &AppState, bot_id: &str, config: &Value) -> Result<(), ApiError> {
    let dir = tutorbot_dir(state, bot_id);
    fs::create_dir_all(&dir).map_err(|error| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("Failed to create TutorBot directory: {error}"),
        )
    })?;
    let bytes = serde_json::to_vec_pretty(config).map_err(|error| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("Failed to serialize TutorBot config: {error}"),
        )
    })?;
    fs::write(tutorbot_config_path(state, bot_id), bytes).map_err(|error| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("Failed to write TutorBot config: {error}"),
        )
    })
}

fn default_tutorbot_config(bot_id: &str) -> Value {
    json!({
        "name": bot_id,
        "description": "",
        "persona": "",
        "channels": {
            "send_progress": true,
            "send_tool_hints": false
        },
        "model": null,
        "llm_selection": null,
        "running": false,
        "started_at": null,
        "last_reload_error": null
    })
}

fn normalize_tutorbot_config(bot_id: &str, mut config: Value) -> Value {
    if !config.is_object() {
        config = default_tutorbot_config(bot_id);
    }
    let default = default_tutorbot_config(bot_id);
    for key in [
        "name",
        "description",
        "persona",
        "channels",
        "model",
        "llm_selection",
        "running",
        "started_at",
        "last_reload_error",
    ] {
        if config.get(key).is_none() {
            config[key] = default[key].clone();
        }
    }
    if !config["channels"].is_object() {
        config["channels"] = default["channels"].clone();
    }
    config
}

fn apply_tutorbot_payload(config: &mut Value, payload: &Value, merge_channels: bool) {
    for key in ["name", "description", "persona", "model"] {
        if let Some(value) = payload.get(key).filter(|value| !value.is_null()) {
            config[key] = value.clone();
        }
    }
    if let Some(value) = payload.get("llm_selection") {
        config["llm_selection"] = value.clone();
    }
    if let Some(channels) = payload.get("channels").filter(|value| value.is_object()) {
        config["channels"] = if merge_channels {
            merge_masked_json_secrets(channels, &config["channels"])
        } else {
            channels.clone()
        };
    }
}

fn tutorbot_detail(bot_id: &str, config: &Value, mask_secrets: bool) -> Value {
    json!({
        "bot_id": bot_id,
        "name": config["name"].as_str().unwrap_or(bot_id),
        "description": config["description"].as_str().unwrap_or(""),
        "persona": config["persona"].as_str().unwrap_or(""),
        "channels": if mask_secrets { mask_json_secrets(&config["channels"], None) } else { config["channels"].clone() },
        "model": config["model"].clone(),
        "llm_selection": config["llm_selection"].clone(),
        "running": config["running"].as_bool().unwrap_or(false),
        "started_at": config["started_at"].clone(),
        "last_reload_error": config["last_reload_error"].clone()
    })
}

fn tutorbot_summary(bot_id: &str, config: &Value) -> Value {
    json!({
        "bot_id": bot_id,
        "name": config["name"].as_str().unwrap_or(bot_id),
        "description": config["description"].as_str().unwrap_or(""),
        "persona": config["persona"].as_str().unwrap_or(""),
        "channels": tutorbot_channel_names(&config["channels"]),
        "model": config["model"].clone(),
        "llm_selection": config["llm_selection"].clone(),
        "running": config["running"].as_bool().unwrap_or(false),
        "started_at": config["started_at"].clone(),
        "last_reload_error": config["last_reload_error"].clone()
    })
}

fn tutorbot_summaries(state: &AppState) -> Vec<Value> {
    let Ok(entries) = fs::read_dir(&*state.tutorbot_root) else {
        return Vec::new();
    };
    let mut bots = entries
        .flatten()
        .filter_map(|entry| {
            let bot_id = entry.file_name().to_string_lossy().to_string();
            if !entry.file_type().ok()?.is_dir() {
                return None;
            }
            read_tutorbot_config(state, &bot_id).map(|config| tutorbot_summary(&bot_id, &config))
        })
        .collect::<Vec<_>>();
    bots.sort_by(|left, right| {
        left["bot_id"]
            .as_str()
            .unwrap_or("")
            .cmp(right["bot_id"].as_str().unwrap_or(""))
    });
    bots
}

fn tutorbot_channel_names(channels: &Value) -> Vec<String> {
    channels
        .as_object()
        .map(|object| {
            object
                .iter()
                .filter(|(key, value)| {
                    key.as_str() != "send_progress"
                        && key.as_str() != "send_tool_hints"
                        && value.is_object()
                })
                .map(|(key, _)| key.clone())
                .collect()
        })
        .unwrap_or_default()
}

fn ensure_tutorbot_workspace(
    state: &AppState,
    bot_id: &str,
    config: &Value,
) -> Result<(), ApiError> {
    let workspace = tutorbot_workspace_dir(state, bot_id);
    fs::create_dir_all(tutorbot_sessions_dir(state, bot_id)).map_err(|error| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("Failed to create TutorBot workspace: {error}"),
        )
    })?;
    for filename in TUTORBOT_EDITABLE_FILES {
        let path = workspace.join(filename);
        if !path.exists() {
            let content = default_tutorbot_file(filename, bot_id, config);
            fs::write(path, content).map_err(|error| {
                api_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &format!("Failed to seed TutorBot profile file: {error}"),
                )
            })?;
        }
    }
    if let Some(persona) = config["persona"].as_str().filter(|value| !value.is_empty()) {
        let soul_path = workspace.join("SOUL.md");
        fs::write(soul_path, persona).map_err(|error| {
            api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("Failed to write SOUL.md: {error}"),
            )
        })?;
    }
    Ok(())
}

fn default_tutorbot_file(filename: &str, bot_id: &str, config: &Value) -> String {
    match filename {
        "SOUL.md" => config["persona"]
            .as_str()
            .filter(|value| !value.is_empty())
            .map(ToString::to_string)
            .unwrap_or_else(|| {
                format!("# Soul\n\nI am {bot_id}, a Socartes TutorBot focused on learning support.")
            }),
        "USER.md" => "# User\n\nLearning preferences and user context live here.\n".to_string(),
        "TOOLS.md" => {
            "# Tools\n\nAvailable tools are surfaced through Socartes compatibility APIs.\n"
                .to_string()
        }
        "AGENTS.md" => {
            "# Agent Instructions\n\nRespond with grounded, concise tutoring help.\n".to_string()
        }
        "HEARTBEAT.md" => {
            "# Heartbeat\n\nPeriodic proactive guidance can be recorded here.\n".to_string()
        }
        _ => String::new(),
    }
}

fn is_tutorbot_editable_file(filename: &str) -> bool {
    TUTORBOT_EDITABLE_FILES.contains(&filename)
}

fn read_tutorbot_workspace_file(state: &AppState, bot_id: &str, filename: &str) -> Option<String> {
    if !is_tutorbot_editable_file(filename) {
        return None;
    }
    fs::read_to_string(tutorbot_workspace_dir(state, bot_id).join(filename)).ok()
}

fn write_tutorbot_workspace_file(
    state: &AppState,
    bot_id: &str,
    filename: &str,
    content: &str,
) -> Result<(), ApiError> {
    if !is_tutorbot_editable_file(filename) {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            &format!("Not an editable file: {filename}"),
        ));
    }
    if !tutorbot_exists(state, bot_id) {
        return Err(api_error(StatusCode::NOT_FOUND, "Bot not found"));
    }
    let path = tutorbot_workspace_dir(state, bot_id).join(filename);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("Failed to create TutorBot workspace: {error}"),
            )
        })?;
    }
    fs::write(path, content).map_err(|error| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("Failed to write TutorBot file: {error}"),
        )
    })
}

fn secret_reveal_enabled() -> bool {
    env::var("ALLOW_SECRET_REVEAL")
        .map(|value| {
            matches!(
                value.to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn is_secret_key(key: &str) -> bool {
    let lowered = key.to_ascii_lowercase();
    [
        "token",
        "secret",
        "password",
        "api_key",
        "apikey",
        "encrypt_key",
    ]
    .iter()
    .any(|hint| lowered.contains(hint))
}

fn mask_json_secrets(value: &Value, key_hint: Option<&str>) -> Value {
    match value {
        Value::Object(object) => Value::Object(
            object
                .iter()
                .map(|(key, value)| (key.clone(), mask_json_secrets(value, Some(key))))
                .collect(),
        ),
        Value::Array(values) => Value::Array(
            values
                .iter()
                .map(|value| mask_json_secrets(value, key_hint))
                .collect(),
        ),
        Value::String(text) if key_hint.is_some_and(is_secret_key) && !text.is_empty() => {
            json!(SECRET_MASK)
        }
        _ => value.clone(),
    }
}

fn merge_masked_json_secrets(incoming: &Value, current: &Value) -> Value {
    merge_masked_json_secrets_with_key(incoming, current, None)
}

fn merge_masked_json_secrets_with_key(
    incoming: &Value,
    current: &Value,
    key_hint: Option<&str>,
) -> Value {
    match incoming {
        Value::Object(object) => Value::Object(
            object
                .iter()
                .map(|(key, value)| {
                    (
                        key.clone(),
                        merge_masked_json_secrets_with_key(value, &current[key], Some(key)),
                    )
                })
                .collect(),
        ),
        Value::Array(values) => Value::Array(
            values
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    merge_masked_json_secrets_with_key(value, &current[index], key_hint)
                })
                .collect(),
        ),
        Value::String(text)
            if key_hint.is_some_and(is_secret_key)
                && text == SECRET_MASK
                && current
                    .as_str()
                    .is_some_and(|existing| !existing.is_empty()) =>
        {
            current.clone()
        }
        _ => incoming.clone(),
    }
}

fn load_tutorbot_souls(state: &AppState) -> Vec<Value> {
    let path = state.tutorbot_root.join("_souls.json");
    if !path.exists() {
        let defaults = default_tutorbot_souls();
        let _ = save_tutorbot_souls(state, &defaults);
        return defaults;
    }
    fs::read_to_string(path)
        .ok()
        .and_then(|text| serde_json::from_str::<Value>(&text).ok())
        .and_then(|value| value.as_array().cloned())
        .unwrap_or_default()
}

fn save_tutorbot_souls(state: &AppState, souls: &[Value]) -> Result<(), ApiError> {
    fs::create_dir_all(&*state.tutorbot_root).map_err(|error| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("Failed to create TutorBot directory: {error}"),
        )
    })?;
    let bytes = serde_json::to_vec_pretty(souls).map_err(|error| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("Failed to serialize souls: {error}"),
        )
    })?;
    fs::write(state.tutorbot_root.join("_souls.json"), bytes).map_err(|error| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("Failed to write souls: {error}"),
        )
    })
}

fn default_tutorbot_souls() -> Vec<Value> {
    vec![
        json!({
            "id": "default-tutorbot",
            "name": "Default TutorBot",
            "content": "# Soul\n\nI am TutorBot, a personal learning companion.\n\n## Personality\n\n- Helpful and friendly\n- Clear, encouraging, and patient\n- Adapts explanations to the user's level\n\n## Values\n\n- Accuracy over speed\n- User privacy and safety\n- Transparency in actions"
        }),
        json!({
            "id": "math-tutor",
            "name": "Math Tutor",
            "content": "# Soul\n\nI am a math tutor specializing in clear, step-by-step problem solving.\n\n## Teaching Style\n\n- Break complex problems into small steps\n- Use visual representations when possible\n- Always verify final answers"
        }),
        json!({
            "id": "coding-assistant",
            "name": "Coding Assistant",
            "content": "# Soul\n\nI am a coding assistant focused on helping developers write better software.\n\n## Approach\n\n- Read before writing\n- Suggest tests alongside implementations\n- Prefer standard patterns over clever tricks"
        }),
        json!({
            "id": "research-helper",
            "name": "Research Helper",
            "content": "# Soul\n\nI am a research assistant helping users explore academic topics in depth.\n\n## Approach\n\n- Decompose broad questions\n- Distinguish facts from open questions\n- Suggest further reading"
        }),
    ]
}

fn tutorbot_history_messages(state: &AppState, bot_id: &str, limit: usize) -> Vec<Value> {
    let Ok(entries) = fs::read_dir(tutorbot_sessions_dir(state, bot_id)) else {
        return Vec::new();
    };
    let mut paths = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("jsonl"))
        .collect::<Vec<_>>();
    paths.sort();

    let mut messages = Vec::new();
    for path in paths {
        let Ok(text) = fs::read_to_string(path) else {
            continue;
        };
        for line in text.lines().filter(|line| !line.trim().is_empty()) {
            let Ok(mut value) = serde_json::from_str::<Value>(line) else {
                continue;
            };
            let role = value["role"].as_str().unwrap_or("");
            if role != "user" && role != "assistant" {
                continue;
            }
            let Some(content) = normalize_history_content(&value["content"]) else {
                continue;
            };
            value["content"] = json!(content);
            if let Some(object) = value.as_object_mut() {
                object.remove("reasoning_content");
            }
            messages.push(value);
        }
    }
    let start = messages.len().saturating_sub(limit);
    messages[start..].to_vec()
}

fn normalize_history_content(value: &Value) -> Option<String> {
    match value {
        Value::String(text) if !text.is_empty() => Some(text.clone()),
        Value::Array(values) => {
            let content = values
                .iter()
                .filter_map(normalize_history_content)
                .collect::<Vec<_>>()
                .join(" ");
            (!content.is_empty()).then_some(content)
        }
        Value::Object(object) => ["text", "content", "message", "alt"]
            .iter()
            .find_map(|key| object.get(*key).and_then(Value::as_str))
            .map(ToString::to_string)
            .or_else(|| {
                (object.get("type").and_then(Value::as_str) == Some("image"))
                    .then(|| "[image]".to_string())
            }),
        _ => None,
    }
}

fn recent_tutorbot_summaries(state: &AppState, limit: usize) -> Vec<Value> {
    let Ok(entries) = fs::read_dir(&*state.tutorbot_root) else {
        return Vec::new();
    };
    let mut recent = Vec::new();
    for entry in entries.flatten() {
        let bot_id = entry.file_name().to_string_lossy().to_string();
        if !entry.file_type().map(|ty| ty.is_dir()).unwrap_or(false) {
            continue;
        }
        let history = tutorbot_history_messages(state, &bot_id, usize::MAX);
        let Some(last_message) = history.last() else {
            continue;
        };
        let updated_at = last_message["timestamp"]
            .as_str()
            .map(ToString::to_string)
            .unwrap_or_else(now_rfc3339);
        let sort_key = last_message["timestamp_ms"].as_u64().unwrap_or_default();
        let config = read_tutorbot_config(state, &bot_id)
            .unwrap_or_else(|| default_tutorbot_config(&bot_id));
        recent.push((
            sort_key,
            json!({
                "bot_id": bot_id,
                "name": config["name"].as_str().unwrap_or(&bot_id),
                "running": config["running"].as_bool().unwrap_or(false),
                "last_message": last_message["content"].as_str().unwrap_or("").chars().take(200).collect::<String>(),
                "updated_at": updated_at
            }),
        ));
    }
    recent.sort_by(|left, right| right.0.cmp(&left.0));
    recent
        .into_iter()
        .take(limit)
        .map(|(_, item)| item)
        .collect()
}

fn append_tutorbot_history(
    state: &AppState,
    bot_id: &str,
    chat_id: &str,
    role: &str,
    content: &str,
) -> Result<(), ApiError> {
    let sessions = tutorbot_sessions_dir(state, bot_id);
    fs::create_dir_all(&sessions).map_err(|error| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("Failed to create TutorBot session directory: {error}"),
        )
    })?;
    let file = safe_storage_component(chat_id).unwrap_or_else(|| "web".to_string());
    let path = sessions.join(format!("{file}.jsonl"));
    let mut handle = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|error| {
            api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("Failed to open TutorBot history: {error}"),
            )
        })?;
    let timestamp = now_rfc3339();
    let timestamp_ms = current_epoch_millis();
    let line = json!({
        "role": role,
        "content": content,
        "timestamp": timestamp,
        "timestamp_ms": timestamp_ms
    })
    .to_string();
    writeln!(handle, "{line}").map_err(|error| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("Failed to write TutorBot history: {error}"),
        )
    })
}

async fn handle_tutorbot_socket(mut socket: WebSocket, state: AppState, bot_id: String) {
    let Some(bot_id) = safe_storage_component(&bot_id) else {
        let _ = socket
            .send(Message::Text(
                json!({"type": "error", "content": "Invalid bot id"})
                    .to_string()
                    .into(),
            ))
            .await;
        let _ = socket.send(Message::Close(None)).await;
        return;
    };

    let Some(mut config) = read_tutorbot_config(&state, &bot_id) else {
        let _ = socket
            .send(Message::Text(
                json!({"type": "error", "content": "Bot not found"})
                    .to_string()
                    .into(),
            ))
            .await;
        let _ = socket.send(Message::Close(None)).await;
        return;
    };

    if !config["running"].as_bool().unwrap_or(false) {
        config["running"] = json!(true);
        config["started_at"] = json!(now_rfc3339());
        let _ = write_tutorbot_config(&state, &bot_id, &config);
    }

    while let Some(Ok(message)) = socket.recv().await {
        let Message::Text(text) = message else {
            if matches!(message, Message::Close(_)) {
                break;
            }
            continue;
        };
        let Ok(payload) = serde_json::from_str::<Value>(&text) else {
            let _ = socket
                .send(Message::Text(
                    json!({"type": "error", "content": "Invalid JSON"})
                        .to_string()
                        .into(),
                ))
                .await;
            continue;
        };
        let content = payload["content"].as_str().unwrap_or("").trim();
        if content.is_empty() {
            continue;
        }
        let chat_id = payload["chat_id"].as_str().unwrap_or("web");
        let _ = append_tutorbot_history(&state, &bot_id, chat_id, "user", content);
        let _ = socket
            .send(Message::Text(
                json!({"type": "thinking", "content": "TutorBot is reading the local profile and conversation history."})
                    .to_string()
                    .into(),
            ))
            .await;
        let response = tutorbot_chat_response(&bot_id, &config, content);
        let _ = append_tutorbot_history(&state, &bot_id, chat_id, "assistant", &response);
        if socket
            .send(Message::Text(
                json!({"type": "content", "content": response})
                    .to_string()
                    .into(),
            ))
            .await
            .is_err()
        {
            break;
        }
        if socket
            .send(Message::Text(json!({"type": "done"}).to_string().into()))
            .await
            .is_err()
        {
            break;
        }
    }
}

fn tutorbot_chat_response(bot_id: &str, config: &Value, content: &str) -> String {
    let name = config["name"].as_str().unwrap_or(bot_id);
    let persona = config["persona"].as_str().unwrap_or("").trim();
    let persona_clause = if persona.is_empty() {
        "using the default Socartes tutor profile".to_string()
    } else {
        format!("using this profile: {}", compact_excerpt(persona, 220))
    };
    format!(
        "{name} received: \"{content}\". I am responding through the Rust TutorBot compatibility runtime, {persona_clause}."
    )
}

fn now_rfc3339() -> String {
    Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true)
}

fn current_epoch_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or_default()
}

fn knowledge_base_dir(state: &AppState, name: &str) -> PathBuf {
    let component = safe_storage_component(name).unwrap_or_else(|| "invalid".to_string());
    state.knowledge_root.join("bases").join(component)
}

fn knowledge_files_dir(state: &AppState, name: &str) -> PathBuf {
    knowledge_base_dir(state, name).join("files")
}

fn knowledge_metadata_path(state: &AppState, name: &str) -> PathBuf {
    knowledge_base_dir(state, name).join("metadata.json")
}

fn default_knowledge_path(state: &AppState) -> PathBuf {
    state.knowledge_root.join("default.txt")
}

fn read_default_knowledge_base(state: &AppState) -> String {
    fs::read_to_string(default_knowledge_path(state))
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| knowledge_base_exists(state, value))
        .unwrap_or_else(|| BUILTIN_KNOWLEDGE_BASE.to_string())
}

fn knowledge_base_exists(state: &AppState, name: &str) -> bool {
    name == BUILTIN_KNOWLEDGE_BASE || knowledge_base_dir(state, name).is_dir()
}

fn now_label() -> String {
    format!("{:.0}", now_seconds())
}

fn knowledge_base_summaries(state: &AppState) -> Vec<Value> {
    let default_name = read_default_knowledge_base(state);
    let mut summaries = vec![builtin_knowledge_summary(&default_name)];

    let bases_dir = state.knowledge_root.join("bases");
    if let Ok(entries) = fs::read_dir(bases_dir) {
        let mut local_names = entries
            .filter_map(Result::ok)
            .filter(|entry| entry.path().is_dir())
            .filter_map(|entry| entry.file_name().into_string().ok())
            .collect::<Vec<_>>();
        local_names.sort();
        for name in local_names {
            summaries.push(local_knowledge_summary(state, &name, &default_name));
        }
    }

    summaries
}

fn builtin_knowledge_summary(default_name: &str) -> Value {
    json!({
        "name": BUILTIN_KNOWLEDGE_BASE,
        "is_default": default_name == BUILTIN_KNOWLEDGE_BASE,
        "status": "ready",
        "path": "builtin://socartes-rust-rag",
        "metadata": {
            "description": "Built-in Socartes Rust RAG notes for local frontend smoke tests.",
            "rag_provider": DEFAULT_RAG_PROVIDER,
            "last_updated": "builtin"
        },
        "statistics": {
            "raw_documents": builtin_knowledge_file_entries().len(),
            "images": 0,
            "content_lists": 0,
            "chunks": knowledge_base().len(),
            "rag_provider": DEFAULT_RAG_PROVIDER,
            "rag_initialized": true,
            "needs_reindex": false,
            "status": "ready",
            "active_match": true,
            "index_versions": [
                {
                    "signature": "socartes-rust-builtin",
                    "model": "deterministic-agent-loop",
                    "dimension": 0,
                    "binding": "builtin",
                    "created_at": "builtin",
                    "ready": true,
                    "legacy": false
                }
            ],
            "active_signature": "socartes-rust-builtin"
        }
    })
}

fn local_knowledge_summary(state: &AppState, name: &str, default_name: &str) -> Value {
    let files = knowledge_files(state, name).unwrap_or_default();
    let metadata = read_knowledge_metadata(state, name).unwrap_or_else(|| {
        json!({
            "created_at": now_label(),
            "last_updated": now_label(),
            "rag_provider": DEFAULT_RAG_PROVIDER
        })
    });
    json!({
        "name": name,
        "is_default": default_name == name,
        "status": "ready",
        "path": knowledge_base_dir(state, name).to_string_lossy(),
        "metadata": metadata,
        "statistics": {
            "raw_documents": files.len(),
            "images": 0,
            "content_lists": 0,
            "rag_provider": metadata["rag_provider"].as_str().unwrap_or(DEFAULT_RAG_PROVIDER),
            "rag_initialized": true,
            "needs_reindex": false,
            "status": "ready",
            "active_match": true,
            "index_versions": [
                {
                    "signature": format!("socartes-rust-{name}"),
                    "model": "deterministic-agent-loop",
                    "dimension": 0,
                    "binding": "local-files",
                    "created_at": metadata["created_at"].as_str().unwrap_or("local"),
                    "ready": true,
                    "legacy": false
                }
            ],
            "active_signature": format!("socartes-rust-{name}")
        }
    })
}

fn read_knowledge_metadata(state: &AppState, name: &str) -> Option<Value> {
    let text = fs::read_to_string(knowledge_metadata_path(state, name)).ok()?;
    serde_json::from_str(&text).ok()
}

fn write_knowledge_metadata(
    state: &AppState,
    name: &str,
    provider: &str,
    extra: Option<Value>,
) -> Result<(), ApiError> {
    let dir = knowledge_base_dir(state, name);
    fs::create_dir_all(dir.join("files")).map_err(|error| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("Failed to create knowledge base: {error}"),
        )
    })?;

    let existing = read_knowledge_metadata(state, name);
    let created_at = existing
        .as_ref()
        .and_then(|value| value["created_at"].as_str())
        .map(ToString::to_string)
        .unwrap_or_else(now_label);

    let mut metadata = json!({
        "created_at": created_at,
        "last_updated": now_label(),
        "last_indexed_at": now_label(),
        "last_indexed_action": "upsert",
        "rag_provider": provider,
        "embedding_model": "deterministic-agent-loop",
        "embedding_dim": 0,
        "needs_reindex": false,
        "embedding_mismatch": false
    });

    if let Some(Value::Object(extra)) = extra
        && let Some(object) = metadata.as_object_mut()
    {
        for (key, value) in extra {
            object.insert(key, value);
        }
    }

    let bytes = serde_json::to_vec_pretty(&metadata).map_err(|error| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("Failed to serialize knowledge metadata: {error}"),
        )
    })?;
    fs::write(knowledge_metadata_path(state, name), bytes).map_err(|error| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("Failed to write knowledge metadata: {error}"),
        )
    })
}

fn builtin_knowledge_file_entries() -> Vec<(&'static str, &'static str)> {
    vec![
        (
            "agent-loop.md",
            "Socartes uses Planner, Retriever, Executor, Tool Adapter, Critic, and Reflection roles to make learner-facing answers auditable.",
        ),
        (
            "rag-notes.md",
            "RAG evidence grounds generated answers in retrieved course documents before the critic approves the response.",
        ),
    ]
}

fn builtin_knowledge_file(name: &str) -> Option<String> {
    builtin_knowledge_file_entries()
        .into_iter()
        .find(|(filename, _)| *filename == name)
        .map(|(_, content)| content.to_string())
}

fn builtin_knowledge_files() -> Vec<Value> {
    builtin_knowledge_file_entries()
        .into_iter()
        .map(|(name, content)| {
            json!({
                "name": name,
                "size": content.len(),
                "modified": 0,
                "mime_type": "text/markdown"
            })
        })
        .collect()
}

fn knowledge_files(state: &AppState, name: &str) -> Result<Vec<Value>, ApiError> {
    if name == BUILTIN_KNOWLEDGE_BASE {
        return Ok(builtin_knowledge_files());
    }
    if !knowledge_base_exists(state, name) {
        return Err(api_error(StatusCode::NOT_FOUND, "Knowledge base not found"));
    }

    let dir = knowledge_files_dir(state, name);
    let mut files = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.filter_map(Result::ok) {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let metadata = entry.metadata().ok();
            let modified = metadata
                .as_ref()
                .and_then(|meta| meta.modified().ok())
                .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
                .map(|duration| duration.as_secs_f64())
                .unwrap_or_default();
            let size = metadata.as_ref().map(|meta| meta.len()).unwrap_or_default();
            if let Some(name) = entry.file_name().to_str() {
                files.push(json!({
                    "name": name,
                    "size": size,
                    "modified": modified,
                    "mime_type": mime_type_for(name)
                }));
            }
        }
    }
    files.sort_by(|left, right| left["name"].as_str().cmp(&right["name"].as_str()));
    Ok(files)
}

fn mime_type_for(name: &str) -> &'static str {
    match FsPath::new(name)
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "md" | "markdown" => "text/markdown",
        "json" => "application/json",
        "csv" => "text/csv",
        "pdf" => "application/pdf",
        _ => "text/plain",
    }
}

fn is_supported_knowledge_file(filename: &str) -> bool {
    let extension = FsPath::new(filename)
        .extension()
        .and_then(|value| value.to_str())
        .map(|value| format!(".{}", value.to_ascii_lowercase()))
        .unwrap_or_default();
    SUPPORTED_KNOWLEDGE_EXTENSIONS.contains(&extension.as_str())
}

async fn save_knowledge_base_from_multipart(
    state: &AppState,
    mut multipart: Multipart,
    target_name: Option<String>,
) -> Result<(String, String), ApiError> {
    let mut name = target_name;
    let mut provider = DEFAULT_RAG_PROVIDER.to_string();
    let mut files = Vec::<(String, Vec<u8>)>::new();

    while let Some(field) = multipart.next_field().await.map_err(|error| {
        api_error(
            StatusCode::BAD_REQUEST,
            &format!("Invalid multipart form: {error}"),
        )
    })? {
        let field_name = field.name().unwrap_or_default().to_string();
        let file_name = field.file_name().and_then(safe_filename);
        let bytes = field.bytes().await.map_err(|error| {
            api_error(
                StatusCode::BAD_REQUEST,
                &format!("Failed to read multipart field: {error}"),
            )
        })?;

        match field_name.as_str() {
            "name" if name.is_none() => {
                let parsed = String::from_utf8(bytes.to_vec()).map_err(|_| {
                    api_error(StatusCode::BAD_REQUEST, "Knowledge base name must be UTF-8")
                })?;
                name = Some(parsed);
            }
            "rag_provider" => {
                provider = String::from_utf8(bytes.to_vec())
                    .unwrap_or_else(|_| DEFAULT_RAG_PROVIDER.to_string())
                    .trim()
                    .to_string();
                if provider.is_empty() {
                    provider = DEFAULT_RAG_PROVIDER.to_string();
                }
            }
            "files" => {
                let Some(file_name) = file_name else {
                    return Err(api_error(StatusCode::BAD_REQUEST, "Missing filename"));
                };
                if !is_supported_knowledge_file(&file_name) {
                    return Err(api_error(StatusCode::BAD_REQUEST, "Unsupported file type"));
                }
                files.push((file_name, bytes.to_vec()));
            }
            _ => {}
        }
    }

    let name = name
        .and_then(|value| safe_storage_component(&value))
        .ok_or_else(|| api_error(StatusCode::BAD_REQUEST, "Knowledge base name is required"))?;
    if name == BUILTIN_KNOWLEDGE_BASE {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "Cannot overwrite the built-in Socartes course",
        ));
    }
    if files.is_empty() {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "At least one supported file is required",
        ));
    }

    let files_dir = knowledge_files_dir(state, &name);
    fs::create_dir_all(&files_dir).map_err(|error| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("Failed to create knowledge file directory: {error}"),
        )
    })?;
    for (filename, bytes) in files {
        fs::write(files_dir.join(filename), bytes).map_err(|error| {
            api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("Failed to write uploaded file: {error}"),
            )
        })?;
    }

    write_knowledge_metadata(state, &name, &provider, None)?;
    Ok((name, format!("task-{}", unique_id())))
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

async fn get_settings(State(state): State<AppState>) -> Json<Value> {
    Json(json!({
        "ui": load_ui_settings(&state),
        "catalog": load_settings_catalog(&state),
        "providers": settings_provider_choices()
    }))
}

async fn get_settings_catalog(State(state): State<AppState>) -> Json<Value> {
    Json(json!({ "catalog": load_settings_catalog(&state) }))
}

async fn update_settings_catalog(
    State(state): State<AppState>,
    Json(payload): Json<Value>,
) -> impl IntoResponse {
    let catalog = payload
        .get("catalog")
        .filter(|value| value.is_object())
        .cloned()
        .unwrap_or_else(default_settings_catalog);
    match write_settings_json(&state, "catalog.json", &catalog) {
        Ok(()) => Json(json!({ "catalog": catalog })).into_response(),
        Err(error) => error.into_response(),
    }
}

async fn apply_settings_catalog(
    State(state): State<AppState>,
    Json(payload): Json<Value>,
) -> impl IntoResponse {
    let catalog = payload
        .get("catalog")
        .filter(|value| value.is_object())
        .cloned()
        .unwrap_or_else(|| load_settings_catalog(&state));
    if let Err(error) = write_settings_json(&state, "catalog.json", &catalog) {
        return error.into_response();
    }
    let env = render_settings_env(&catalog);
    match write_settings_json(&state, "applied-env.json", &env) {
        Ok(()) => Json(json!({
            "message": "Catalog applied to the active .env configuration.",
            "catalog": catalog,
            "env": env
        }))
        .into_response(),
        Err(error) => error.into_response(),
    }
}

async fn update_ui_settings_endpoint(
    State(state): State<AppState>,
    Json(payload): Json<Value>,
) -> impl IntoResponse {
    let mut ui = load_ui_settings(&state);
    if let Some(theme) = payload["theme"].as_str() {
        ui["theme"] = json!(theme);
    }
    if let Some(language) = payload["language"].as_str() {
        ui["language"] = json!(language);
    }
    if let Some(description) = payload["sidebar_description"].as_str() {
        ui["sidebar_description"] = json!(description);
    }
    if payload["sidebar_nav_order"].is_object() {
        ui["sidebar_nav_order"] = payload["sidebar_nav_order"].clone();
    }
    match write_settings_json(&state, "ui.json", &ui) {
        Ok(()) => Json(ui).into_response(),
        Err(error) => error.into_response(),
    }
}

async fn update_theme_endpoint(
    State(state): State<AppState>,
    Json(payload): Json<Value>,
) -> impl IntoResponse {
    let theme = payload["theme"].as_str().unwrap_or("light");
    let mut ui = load_ui_settings(&state);
    ui["theme"] = json!(theme);
    match write_settings_json(&state, "ui.json", &ui) {
        Ok(()) => Json(json!({ "theme": theme })).into_response(),
        Err(error) => error.into_response(),
    }
}

async fn update_language_endpoint(
    State(state): State<AppState>,
    Json(payload): Json<Value>,
) -> impl IntoResponse {
    let language = payload["language"].as_str().unwrap_or("en");
    let mut ui = load_ui_settings(&state);
    ui["language"] = json!(language);
    match write_settings_json(&state, "ui.json", &ui) {
        Ok(()) => Json(json!({ "language": language })).into_response(),
        Err(error) => error.into_response(),
    }
}

async fn reset_settings_endpoint(State(state): State<AppState>) -> impl IntoResponse {
    let ui = default_ui_settings();
    match write_settings_json(&state, "ui.json", &ui) {
        Ok(()) => Json(ui).into_response(),
        Err(error) => error.into_response(),
    }
}

async fn settings_themes() -> Json<Value> {
    Json(json!({
        "themes": [
            { "id": "snow", "name": "Snow" },
            { "id": "light", "name": "Light" },
            { "id": "dark", "name": "Dark" },
            { "id": "glass", "name": "Glass" }
        ]
    }))
}

async fn settings_sidebar(State(state): State<AppState>) -> Json<Value> {
    let ui = load_ui_settings(&state);
    Json(json!({
        "description": ui["sidebar_description"].clone(),
        "nav_order": ui["sidebar_nav_order"].clone()
    }))
}

async fn start_settings_test(
    Path(service): Path<String>,
    Json(payload): Json<Value>,
) -> Json<Value> {
    let _ = payload;
    Json(json!({ "run_id": format!("{service}-{}", unique_id()) }))
}

async fn settings_test_events(
    Path((service, run_id)): Path<(String, String)>,
) -> impl IntoResponse {
    let service = service.replace('"', "");
    let run_id = run_id.replace('"', "");
    let body = format!(
        "data: {{\"type\":\"started\",\"message\":\"Socartes Rust {service} diagnostics started\",\"run_id\":\"{run_id}\"}}\n\n\
data: {{\"type\":\"capabilities\",\"message\":\"Detected deterministic Rust compatibility capabilities\",\"detected_dim\":3072,\"default_dim\":3072,\"supported_dimensions\":[1536,3072],\"supports_variable_dimensions\":true,\"model_known\":true,\"active_dim\":3072,\"active_dim_source\":\"catalog\"}}\n\n\
data: {{\"type\":\"completed\",\"message\":\"{service} diagnostics completed\"}}\n\n"
    );
    ([(header::CONTENT_TYPE, "text/event-stream")], body)
}

async fn cancel_settings_test(Path((_service, _run_id)): Path<(String, String)>) -> Json<Value> {
    Json(json!({ "message": "Cancelled" }))
}

async fn system_status(State(state): State<AppState>) -> Json<Value> {
    let catalog = load_settings_catalog(&state);
    let llm_model = active_catalog_model_name(&catalog, "llm");
    let embedding_model = active_catalog_model_name(&catalog, "embedding");
    let search_provider = active_search_provider(&catalog);
    Json(json!({
        "backend": {
            "status": "online",
            "timestamp": now_label()
        },
        "llm": {
            "status": if llm_model.is_some() { "configured" } else { "not_configured" },
            "model": llm_model,
            "testable": true
        },
        "embeddings": {
            "status": if embedding_model.is_some() { "configured" } else { "not_configured" },
            "model": embedding_model,
            "testable": true
        },
        "search": {
            "status": if search_provider.is_some() { "configured" } else { "optional" },
            "provider": search_provider,
            "testable": true
        }
    }))
}

async fn system_runtime_topology() -> Json<Value> {
    Json(json!({
        "primary_runtime": {
            "transport": "/api/v1/ws",
            "manager": "RustTurnRuntime",
            "orchestrator": "SocartesOrchestrator",
            "session_store": "FileSessionStore",
            "capability_entry": "Rust compatibility endpoints",
            "tool_entry": "Deterministic local adapters"
        },
        "compatibility_routes": [
            {"router": "book", "mode": "file_backed_compatibility"},
            {"router": "knowledge", "mode": "file_backed_compatibility"},
            {"router": "settings", "mode": "file_backed_compatibility"}
        ],
        "isolated_subsystems": []
    }))
}

async fn system_test_llm(State(state): State<AppState>) -> Json<Value> {
    let model = active_catalog_model_name(&load_settings_catalog(&state), "llm")
        .unwrap_or_else(|| "deterministic-agent-loop".to_string());
    Json(json!({
        "success": true,
        "message": "Socartes Rust deterministic LLM test completed.",
        "model": model,
        "response_time_ms": 1.0,
        "error": null
    }))
}

async fn system_test_embeddings(State(state): State<AppState>) -> Json<Value> {
    let model = active_catalog_model_name(&load_settings_catalog(&state), "embedding")
        .unwrap_or_else(|| "deterministic-embedding".to_string());
    Json(json!({
        "success": true,
        "message": "Socartes Rust deterministic embedding test completed.",
        "model": model,
        "response_time_ms": 1.0,
        "error": null
    }))
}

async fn system_test_search(State(state): State<AppState>) -> Json<Value> {
    let provider = active_search_provider(&load_settings_catalog(&state))
        .unwrap_or_else(|| "duckduckgo".to_string());
    Json(json!({
        "success": true,
        "message": "Socartes Rust deterministic search test completed.",
        "model": provider,
        "response_time_ms": 1.0,
        "error": null
    }))
}

async fn list_tutorbots(State(state): State<AppState>) -> Json<Value> {
    Json(Value::Array(tutorbot_summaries(&state)))
}

async fn start_tutorbot(
    State(state): State<AppState>,
    Json(payload): Json<Value>,
) -> impl IntoResponse {
    let Some(bot_id) = payload["bot_id"].as_str().and_then(safe_storage_component) else {
        return api_error(StatusCode::UNPROCESSABLE_ENTITY, "bot_id is required").into_response();
    };

    let mut config =
        read_tutorbot_config(&state, &bot_id).unwrap_or_else(|| default_tutorbot_config(&bot_id));
    apply_tutorbot_payload(&mut config, &payload, false);
    config["running"] = json!(true);
    config["started_at"] = json!(now_rfc3339());
    config["last_reload_error"] = Value::Null;

    if let Err(error) = ensure_tutorbot_workspace(&state, &bot_id, &config) {
        return error.into_response();
    }
    match write_tutorbot_config(&state, &bot_id, &config) {
        Ok(()) => Json(tutorbot_detail(&bot_id, &config, true)).into_response(),
        Err(error) => error.into_response(),
    }
}

async fn recent_tutorbots(
    State(state): State<AppState>,
    Query(query): Query<TutorBotRecentQuery>,
) -> Json<Value> {
    let limit = query.limit.unwrap_or(3).clamp(1, 50);
    Json(Value::Array(recent_tutorbot_summaries(&state, limit)))
}

async fn tutorbot_channel_schema() -> Json<Value> {
    Json(json!({
        "channels": {
            "telegram": {
                "name": "telegram",
                "display_name": "Telegram",
                "default_config": {
                    "enabled": false,
                    "token": "",
                    "allow_from": ["*"]
                },
                "secret_fields": ["token"],
                "json_schema": {
                    "type": "object",
                    "description": "Telegram bot channel settings.",
                    "properties": {
                        "enabled": { "type": "boolean", "title": "Enabled", "default": false },
                        "token": { "type": "string", "title": "Bot token", "default": "" },
                        "allow_from": {
                            "type": "array",
                            "title": "Allowed chats",
                            "items": { "type": "string" },
                            "default": ["*"]
                        }
                    }
                }
            },
            "web": {
                "name": "web",
                "display_name": "Web Chat",
                "default_config": { "enabled": true },
                "secret_fields": [],
                "json_schema": {
                    "type": "object",
                    "description": "Built-in browser chat channel.",
                    "properties": {
                        "enabled": { "type": "boolean", "title": "Enabled", "default": true }
                    }
                }
            }
        },
        "global": {
            "secret_fields": [],
            "json_schema": {
                "type": "object",
                "properties": {
                    "send_progress": { "type": "boolean", "title": "Stream progress text", "default": true },
                    "send_tool_hints": { "type": "boolean", "title": "Stream tool hints", "default": false }
                }
            }
        }
    }))
}

async fn list_tutorbot_souls(State(state): State<AppState>) -> Json<Value> {
    Json(Value::Array(load_tutorbot_souls(&state)))
}

async fn create_tutorbot_soul(
    State(state): State<AppState>,
    Json(payload): Json<Value>,
) -> impl IntoResponse {
    let id = payload["id"].as_str().unwrap_or("").trim();
    let name = payload["name"].as_str().unwrap_or("").trim();
    let content = payload["content"].as_str().unwrap_or("");
    if safe_storage_component(id).is_none() || name.is_empty() {
        return api_error(StatusCode::UNPROCESSABLE_ENTITY, "id and name are required")
            .into_response();
    }

    let mut souls = load_tutorbot_souls(&state);
    if souls.iter().any(|soul| soul["id"] == id) {
        return api_error(StatusCode::CONFLICT, &format!("Soul '{id}' already exists"))
            .into_response();
    }

    let entry = json!({ "id": id, "name": name, "content": content });
    souls.push(entry.clone());
    match save_tutorbot_souls(&state, &souls) {
        Ok(()) => Json(entry).into_response(),
        Err(error) => error.into_response(),
    }
}

async fn get_tutorbot_soul(
    State(state): State<AppState>,
    Path(soul_id): Path<String>,
) -> impl IntoResponse {
    let souls = load_tutorbot_souls(&state);
    match souls.into_iter().find(|soul| soul["id"] == soul_id) {
        Some(soul) => Json(soul).into_response(),
        None => api_error(StatusCode::NOT_FOUND, "Soul not found").into_response(),
    }
}

async fn update_tutorbot_soul(
    State(state): State<AppState>,
    Path(soul_id): Path<String>,
    Json(payload): Json<Value>,
) -> impl IntoResponse {
    let mut souls = load_tutorbot_souls(&state);
    let Some(index) = souls.iter().position(|soul| soul["id"] == soul_id) else {
        return api_error(StatusCode::NOT_FOUND, "Soul not found").into_response();
    };
    if let Some(name) = payload["name"].as_str() {
        souls[index]["name"] = json!(name);
    }
    if let Some(content) = payload["content"].as_str() {
        souls[index]["content"] = json!(content);
    }
    let updated = souls[index].clone();
    match save_tutorbot_souls(&state, &souls) {
        Ok(()) => Json(updated).into_response(),
        Err(error) => error.into_response(),
    }
}

async fn delete_tutorbot_soul(
    State(state): State<AppState>,
    Path(soul_id): Path<String>,
) -> impl IntoResponse {
    let mut souls = load_tutorbot_souls(&state);
    let before = souls.len();
    souls.retain(|soul| soul["id"] != soul_id);
    if souls.len() == before {
        return api_error(StatusCode::NOT_FOUND, "Soul not found").into_response();
    }
    match save_tutorbot_souls(&state, &souls) {
        Ok(()) => Json(json!({ "id": soul_id, "deleted": true })).into_response(),
        Err(error) => error.into_response(),
    }
}

async fn get_tutorbot(
    State(state): State<AppState>,
    Path(bot_id): Path<String>,
    Query(query): Query<TutorBotDetailQuery>,
) -> impl IntoResponse {
    let Some(bot_id) = safe_storage_component(&bot_id) else {
        return api_error(StatusCode::BAD_REQUEST, "Invalid bot id").into_response();
    };
    match read_tutorbot_config(&state, &bot_id) {
        Some(config) => Json(tutorbot_detail(
            &bot_id,
            &config,
            !(query.include_secrets.unwrap_or(false) && secret_reveal_enabled()),
        ))
        .into_response(),
        None => api_error(StatusCode::NOT_FOUND, "Bot not found").into_response(),
    }
}

async fn update_tutorbot(
    State(state): State<AppState>,
    Path(bot_id): Path<String>,
    Json(payload): Json<Value>,
) -> impl IntoResponse {
    let Some(bot_id) = safe_storage_component(&bot_id) else {
        return api_error(StatusCode::BAD_REQUEST, "Invalid bot id").into_response();
    };
    let Some(mut config) = read_tutorbot_config(&state, &bot_id) else {
        return api_error(StatusCode::NOT_FOUND, "Bot not found").into_response();
    };

    apply_tutorbot_payload(&mut config, &payload, true);
    if let Some(persona) = payload["persona"].as_str()
        && let Err(error) = write_tutorbot_workspace_file(&state, &bot_id, "SOUL.md", persona)
    {
        return error.into_response();
    }
    match write_tutorbot_config(&state, &bot_id, &config) {
        Ok(()) => Json(tutorbot_detail(&bot_id, &config, true)).into_response(),
        Err(error) => error.into_response(),
    }
}

async fn stop_tutorbot(
    State(state): State<AppState>,
    Path(bot_id): Path<String>,
) -> impl IntoResponse {
    let Some(bot_id) = safe_storage_component(&bot_id) else {
        return api_error(StatusCode::BAD_REQUEST, "Invalid bot id").into_response();
    };
    let Some(mut config) = read_tutorbot_config(&state, &bot_id) else {
        return api_error(StatusCode::NOT_FOUND, "Bot not found or not running").into_response();
    };
    if !config["running"].as_bool().unwrap_or(false) {
        return api_error(StatusCode::NOT_FOUND, "Bot not found or not running").into_response();
    }
    config["running"] = json!(false);
    config["started_at"] = Value::Null;
    match write_tutorbot_config(&state, &bot_id, &config) {
        Ok(()) => Json(json!({ "bot_id": bot_id, "stopped": true })).into_response(),
        Err(error) => error.into_response(),
    }
}

async fn destroy_tutorbot(
    State(state): State<AppState>,
    Path(bot_id): Path<String>,
) -> impl IntoResponse {
    let Some(bot_id) = safe_storage_component(&bot_id) else {
        return api_error(StatusCode::BAD_REQUEST, "Invalid bot id").into_response();
    };
    let path = tutorbot_dir(&state, &bot_id);
    if !path.exists() {
        return api_error(StatusCode::NOT_FOUND, "Bot not found").into_response();
    }
    match fs::remove_dir_all(path) {
        Ok(()) => Json(json!({ "bot_id": bot_id, "destroyed": true })).into_response(),
        Err(error) => api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("Failed to delete bot: {error}"),
        )
        .into_response(),
    }
}

async fn list_tutorbot_files(
    State(state): State<AppState>,
    Path(bot_id): Path<String>,
) -> impl IntoResponse {
    let Some(bot_id) = safe_storage_component(&bot_id) else {
        return api_error(StatusCode::BAD_REQUEST, "Invalid bot id").into_response();
    };
    if !tutorbot_exists(&state, &bot_id) {
        return api_error(StatusCode::NOT_FOUND, "Bot not found").into_response();
    }
    Json(Value::Object(
        TUTORBOT_EDITABLE_FILES
            .iter()
            .map(|filename| {
                (
                    (*filename).to_string(),
                    json!(
                        read_tutorbot_workspace_file(&state, &bot_id, filename).unwrap_or_default()
                    ),
                )
            })
            .collect(),
    ))
    .into_response()
}

async fn read_tutorbot_file(
    State(state): State<AppState>,
    Path((bot_id, filename)): Path<(String, String)>,
) -> impl IntoResponse {
    let Some(bot_id) = safe_storage_component(&bot_id) else {
        return api_error(StatusCode::BAD_REQUEST, "Invalid bot id").into_response();
    };
    if !is_tutorbot_editable_file(&filename) {
        return api_error(
            StatusCode::BAD_REQUEST,
            &format!("Not an editable file: {filename}"),
        )
        .into_response();
    }
    if !tutorbot_exists(&state, &bot_id) {
        return api_error(StatusCode::NOT_FOUND, "Bot not found").into_response();
    }
    Json(json!({
        "filename": filename,
        "content": read_tutorbot_workspace_file(&state, &bot_id, &filename).unwrap_or_default()
    }))
    .into_response()
}

async fn write_tutorbot_file(
    State(state): State<AppState>,
    Path((bot_id, filename)): Path<(String, String)>,
    Json(payload): Json<Value>,
) -> impl IntoResponse {
    let Some(bot_id) = safe_storage_component(&bot_id) else {
        return api_error(StatusCode::BAD_REQUEST, "Invalid bot id").into_response();
    };
    let content = payload["content"].as_str().unwrap_or("");
    match write_tutorbot_workspace_file(&state, &bot_id, &filename, content) {
        Ok(()) => {
            if filename == "SOUL.md"
                && let Some(mut config) = read_tutorbot_config(&state, &bot_id)
            {
                config["persona"] = json!(content);
                let _ = write_tutorbot_config(&state, &bot_id, &config);
            }
            Json(json!({ "filename": filename, "saved": true })).into_response()
        }
        Err(error) => error.into_response(),
    }
}

async fn tutorbot_history(
    State(state): State<AppState>,
    Path(bot_id): Path<String>,
    Query(query): Query<TutorBotHistoryQuery>,
) -> impl IntoResponse {
    let Some(bot_id) = safe_storage_component(&bot_id) else {
        return api_error(StatusCode::BAD_REQUEST, "Invalid bot id").into_response();
    };
    if !tutorbot_exists(&state, &bot_id) {
        return api_error(StatusCode::NOT_FOUND, "Bot not found").into_response();
    }
    let limit = query.limit.unwrap_or(100).clamp(1, 1000);
    Json(Value::Array(tutorbot_history_messages(
        &state, &bot_id, limit,
    )))
    .into_response()
}

async fn tutorbot_ws(
    State(state): State<AppState>,
    Path(bot_id): Path<String>,
    ws: WebSocketUpgrade,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_tutorbot_socket(socket, state, bot_id))
}

async fn list_sessions(State(state): State<AppState>) -> Json<Value> {
    Json(json!({ "sessions": session_summaries(&state) }))
}

async fn get_session(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
) -> impl IntoResponse {
    match read_session(&state, &session_id) {
        Ok(session) => Json(session).into_response(),
        Err(error) => error.into_response(),
    }
}

async fn update_session_title(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
    Json(payload): Json<Value>,
) -> impl IntoResponse {
    let title = payload["title"]
        .as_str()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("Untitled chat");
    match read_session(&state, &session_id).and_then(|mut session| {
        session["title"] = json!(title);
        session["updated_at"] = json!(now_seconds());
        write_session(&state, &session_id, &session)?;
        Ok(session)
    }) {
        Ok(session) => Json(json!({ "session": session })).into_response(),
        Err(error) => error.into_response(),
    }
}

async fn delete_session(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
) -> impl IntoResponse {
    let path = session_path(&state, &session_id);
    if !path.exists() {
        return api_error(StatusCode::NOT_FOUND, "Session not found").into_response();
    }
    match fs::remove_file(path) {
        Ok(()) => Json(json!({ "deleted": true })).into_response(),
        Err(error) => api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("Failed to delete session: {error}"),
        )
        .into_response(),
    }
}

async fn record_quiz_results(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
    Json(payload): Json<Value>,
) -> impl IntoResponse {
    match read_session(&state, &session_id).and_then(|mut session| {
        session["quiz_results"] = payload["answers"].clone();
        session["updated_at"] = json!(now_seconds());
        write_session(&state, &session_id, &session)?;
        Ok(())
    }) {
        Ok(()) => Json(json!({ "recorded": true })).into_response(),
        Err(error) => error.into_response(),
    }
}

async fn run_test_chat_turn(
    State(state): State<AppState>,
    Json(payload): Json<Value>,
) -> Json<Value> {
    let (session_id, turn_id, _) = execute_chat_turn(&state, &payload);
    Json(json!({ "session_id": session_id, "turn_id": turn_id }))
}

async fn chat_ws(State(state): State<AppState>, ws: WebSocketUpgrade) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_chat_socket(socket, state))
}

async fn handle_chat_socket(mut socket: WebSocket, state: AppState) {
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
                run_chat_turn(&mut socket, &state, &payload).await;
            }
            "regenerate" => {
                let fallback = json!({
                    "type": "start_turn",
                    "content": "Regenerate the previous Socartes answer.",
                    "session_id": payload["session_id"].as_str()
                });
                run_chat_turn(&mut socket, &state, &fallback).await;
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

async fn run_chat_turn(socket: &mut WebSocket, state: &AppState, payload: &Value) {
    let (_, _, events) = execute_chat_turn(state, payload);
    for event in events {
        if send_stream_event(socket, event).await.is_err() {
            break;
        }
    }
}

fn execute_chat_turn(state: &AppState, payload: &Value) -> (String, String, Vec<Value>) {
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
    let knowledge_bases = as_string_array(&payload["knowledge_bases"]);
    let retrieved_context = retrieve_chat_context(state, content, &knowledge_bases);
    let trace = SocartesOrchestrator::new().run_with_retrieved_context(
        content,
        &learner_context,
        retrieved_context,
    );
    let ids = StreamIds::new(&session_id, &turn_id);

    let events = vec![
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

    let _ = persist_chat_turn(state, payload, &session_id, &turn_id, &trace, &events);
    (session_id, turn_id, events)
}

fn session_path(state: &AppState, session_id: &str) -> PathBuf {
    let component = safe_storage_component(session_id).unwrap_or_else(|| "invalid".to_string());
    state.session_root.join(format!("{component}.json"))
}

fn read_session(state: &AppState, session_id: &str) -> Result<Value, ApiError> {
    let text = fs::read_to_string(session_path(state, session_id))
        .map_err(|_| api_error(StatusCode::NOT_FOUND, "Session not found"))?;
    serde_json::from_str(&text).map_err(|error| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("Failed to parse session: {error}"),
        )
    })
}

fn write_session(state: &AppState, session_id: &str, session: &Value) -> Result<(), ApiError> {
    fs::create_dir_all(&*state.session_root).map_err(|error| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("Failed to create session store: {error}"),
        )
    })?;
    let bytes = serde_json::to_vec_pretty(session).map_err(|error| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("Failed to serialize session: {error}"),
        )
    })?;
    fs::write(session_path(state, session_id), bytes).map_err(|error| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("Failed to write session: {error}"),
        )
    })
}

fn session_summaries(state: &AppState) -> Vec<Value> {
    let mut summaries = Vec::new();
    if let Ok(entries) = fs::read_dir(&*state.session_root) {
        for entry in entries.filter_map(Result::ok) {
            let path = entry.path();
            if path.extension().and_then(|value| value.to_str()) != Some("json") {
                continue;
            }
            let Ok(text) = fs::read_to_string(path) else {
                continue;
            };
            let Ok(session) = serde_json::from_str::<Value>(&text) else {
                continue;
            };
            summaries.push(session_summary(&session));
        }
    }
    summaries.sort_by(|left, right| {
        right["updated_at"]
            .as_f64()
            .partial_cmp(&left["updated_at"].as_f64())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    summaries
}

fn session_summary(session: &Value) -> Value {
    let messages = session["messages"].as_array().cloned().unwrap_or_default();
    let last_message = messages
        .iter()
        .rev()
        .find_map(|message| message["content"].as_str())
        .unwrap_or_default();
    json!({
        "id": session["id"].clone(),
        "session_id": session["session_id"].clone(),
        "title": session["title"].clone(),
        "created_at": session["created_at"].clone(),
        "updated_at": session["updated_at"].clone(),
        "message_count": messages.len(),
        "last_message": last_message,
        "status": session["status"].clone(),
        "active_turn_id": Value::Null,
        "preferences": session["preferences"].clone()
    })
}

fn persist_chat_turn(
    state: &AppState,
    payload: &Value,
    session_id: &str,
    turn_id: &str,
    trace: &StudyTrace,
    events: &[Value],
) -> Result<(), ApiError> {
    let now = now_seconds();
    let mut session = read_session(state, session_id).unwrap_or_else(|_| {
        json!({
            "id": session_id,
            "session_id": session_id,
            "title": title_from_content(payload["content"].as_str().unwrap_or("Socartes chat")),
            "created_at": now,
            "updated_at": now,
            "status": "completed",
            "preferences": {},
            "messages": [],
            "active_turns": []
        })
    });

    let messages_len = session["messages"]
        .as_array()
        .map(|messages| messages.len())
        .unwrap_or_default();
    let capability = payload["capability"].as_str().unwrap_or_default();
    let attachments = if payload["attachments"].is_array() {
        payload["attachments"].clone()
    } else {
        json!([])
    };
    let user_message = json!({
        "id": messages_len + 1,
        "session_id": session_id,
        "role": "user",
        "content": payload["content"].as_str().unwrap_or_default(),
        "capability": capability,
        "events": [],
        "attachments": attachments,
        "metadata": { "turn_id": turn_id },
        "created_at": now
    });
    let assistant_message = json!({
        "id": messages_len + 2,
        "session_id": session_id,
        "role": "assistant",
        "content": trace.final_answer,
        "capability": capability,
        "events": events
            .iter()
            .filter(|event| event["type"] != "session" && event["type"] != "done")
            .cloned()
            .collect::<Vec<_>>(),
        "attachments": [],
        "metadata": { "turn_id": turn_id },
        "created_at": now
    });

    if let Some(messages) = session["messages"].as_array_mut() {
        messages.push(user_message);
        messages.push(assistant_message);
    } else {
        session["messages"] = json!([user_message, assistant_message]);
    }
    session["updated_at"] = json!(now);
    session["status"] = json!("completed");
    session["active_turns"] = json!([]);
    session["preferences"] = json!({
        "capability": payload["capability"].clone(),
        "tools": as_string_array(&payload["tools"]),
        "knowledge_bases": as_string_array(&payload["knowledge_bases"]),
        "language": payload["language"].as_str().unwrap_or("en"),
        "llm_selection": payload["llm_selection"].clone()
    });

    write_session(state, session_id, &session)
}

fn as_string_array(value: &Value) -> Vec<String> {
    value
        .as_array()
        .map(|values| {
            values
                .iter()
                .filter_map(|value| value.as_str().map(ToString::to_string))
                .collect()
        })
        .unwrap_or_default()
}

fn title_from_content(content: &str) -> String {
    let mut title = content.trim().replace('\n', " ");
    if title.is_empty() {
        return "Socartes chat".to_string();
    }
    if title.chars().count() > 60 {
        title = title.chars().take(57).collect::<String>();
        title.push_str("...");
    }
    title
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

fn retrieve_chat_context(
    state: &AppState,
    goal: &str,
    selected_knowledge_bases: &[String],
) -> Vec<RetrievalChunk> {
    if selected_knowledge_bases.is_empty() {
        return retrieve(goal);
    }

    let mut chunks = Vec::new();
    for name in selected_knowledge_bases {
        if name == BUILTIN_KNOWLEDGE_BASE {
            chunks.extend(retrieve(goal));
        } else {
            chunks.extend(retrieve_uploaded_knowledge_base(state, goal, name));
        }
    }

    if chunks.is_empty() {
        return retrieve(goal);
    }

    dedupe_chunks(chunks).into_iter().take(4).collect()
}

fn retrieve_uploaded_knowledge_base(
    state: &AppState,
    goal: &str,
    name: &str,
) -> Vec<RetrievalChunk> {
    if !knowledge_base_exists(state, name) {
        return Vec::new();
    }

    let query_terms = tokenize(goal);
    let mut scored = Vec::new();
    let Ok(entries) = fs::read_dir(knowledge_files_dir(state, name)) else {
        return Vec::new();
    };

    for entry in entries.flatten() {
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if !file_type.is_file() {
            continue;
        }

        let filename = entry.file_name().to_string_lossy().to_string();
        if !is_supported_knowledge_file(&filename) {
            continue;
        }

        let Ok(bytes) = fs::read(entry.path()) else {
            continue;
        };
        let text = String::from_utf8_lossy(&bytes);
        let excerpt = compact_excerpt(&text, 900);
        if excerpt.is_empty() {
            continue;
        }

        let score = query_terms.intersection(&tokenize(&excerpt)).count();
        scored.push((
            score,
            RetrievalChunk {
                source_id: format!("{name}/{filename}"),
                title: format!("{name} / {filename}"),
                content: excerpt,
                confidence: confidence_for_score(score).to_string(),
            },
        ));
    }

    scored.sort_by(|left, right| {
        right
            .0
            .cmp(&left.0)
            .then_with(|| left.1.source_id.cmp(&right.1.source_id))
    });
    scored.into_iter().take(3).map(|(_, chunk)| chunk).collect()
}

fn dedupe_chunks(chunks: Vec<RetrievalChunk>) -> Vec<RetrievalChunk> {
    let mut seen = HashSet::new();
    let mut deduped = Vec::new();

    for chunk in chunks {
        if seen.insert(chunk.source_id.clone()) {
            deduped.push(chunk);
        }
    }

    deduped
}

fn compact_excerpt(text: &str, max_chars: usize) -> String {
    let normalized = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut excerpt = normalized.chars().take(max_chars).collect::<String>();
    if normalized.chars().count() > max_chars {
        excerpt.push_str("...");
    }
    excerpt
}

fn confidence_for_score(score: usize) -> &'static str {
    match score {
        0 => "low",
        1 => "medium",
        _ => "high",
    }
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
    let evidence_clause = chunks.first().map_or_else(String::new, |chunk| {
        format!(
            " The strongest retrieved course note says: {}",
            chunk.content
        )
    });

    DraftAnswer {
        agent: "executor".to_string(),
        content: format!(
            "Socartes answers the goal '{goal}' through a visible agent loop. \
             The Planner decomposes the request, the Retriever supplies RAG \
             evidence, the Executor combines that evidence with MCP-style tool \
             outputs, and the Critic checks whether the answer is cited and \
             complete. RAG evidence comes from {}, while MCP tool use is \
             represented by {}.{}{}",
            citations.join(", "),
            tool_names.join(", "),
            context_clause,
            evidence_clause
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
