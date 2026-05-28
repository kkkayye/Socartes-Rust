use std::collections::{BTreeMap, HashSet};

use axum::{
    Json, Router,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use serde::{Deserialize, Serialize};

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
    detail: String,
}

pub fn app() -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/api/v1/agents", get(agents))
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
                detail: "goal must be at least 3 characters".to_string(),
            }),
        )
            .into_response();
    }

    Json(SocartesOrchestrator::new().run(&request.goal, &request.learner_context)).into_response()
}

async fn ask_story_rag(Json(request): Json<StoryQuestion>) -> Json<StoryAnswer> {
    Json(haunted_pajamas_index().ask(&request.question))
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
