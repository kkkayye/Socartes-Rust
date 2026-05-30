# Socartes Rust Backend

Socartes Rust Backend is a clean Rust rewrite of the Socartes backend prototype. It keeps the same behavior as the previous backend contract while replacing the backend implementation with Rust, Axum, Tokio, and Serde.

This repository is intentionally separate from the original Socartes repository. It contains only the Rust backend implementation and its Rust contract tests.

## Capabilities

| Capability | Rust Implementation |
| --- | --- |
| Multi-Agent: Planner / Executor / Critic role separation | `SocartesOrchestrator` builds the same visible Planner -> Retriever -> Tool Adapter -> Executor -> Critic -> Reflection workflow. |
| RAG (Retrieval-Augmented Generation) | The local RAG index returns cited chunks or refuses when source evidence is missing. |
| MCP / Tool Use | Tool adapter records model external API, knowledge database, and filesystem operations through auditable outputs. |
| Reflection / Self-Correction | Reflection events record critic approval and planning constraints for future answer cycles. |

## Backend API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Returns backend status, service name, and version. |
| `GET` | `/api/v1/agents` | Returns the role boundary and implementation contract for each agent. |
| `POST` | `/api/v1/learn` | Runs the Planner -> Retriever -> Tool Adapter -> Executor -> Critic -> Reflection loop. |
| `POST` | `/api/v1/story-rag/ask` | Tests source-grounded RAG answers against an obscure public-domain novel. |
| `GET` | `/openapi.json` | Returns the OpenAPI schema for the Rust backend. |
| `GET` | `/docs` | Serves a Swagger UI-compatible documentation page. |
| `GET` | `/docs/oauth2-redirect` | Serves the Swagger UI OAuth2 redirect helper page. |
| `GET` | `/redoc` | Serves a ReDoc-compatible documentation page. |

## Run Locally

```bash
cd backend
cargo run
```

The service listens on `0.0.0.0:8000` by default. Set `PORT` to choose a different port:

```bash
PORT=8080 cargo run
```

Open `http://127.0.0.1:8000/docs` for Swagger UI-compatible API documentation, or `http://127.0.0.1:8000/redoc` for ReDoc-compatible documentation.

Example request:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/learn \
  -H 'Content-Type: application/json' \
  -d '{
    "goal": "Compare RAG agents with MCP tool-using agents.",
    "learner_context": "Prefer a concise, citation-backed explanation."
  }'
```

Story RAG grounding test:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/story-rag/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "What did Jenkins say was in the pajama leg?"}'
```

Expected behavior:

- If the database contains the supporting chunk, the answer includes `grounded: true` and the matching `source_ids`.
- If the database does not contain evidence, the answer refuses with `grounded: false` instead of guessing from general model knowledge.

## CLI

The Rust workspace includes a Clap-based CLI implementation in
`backend/src/bin/socartes.rs`. It mirrors the Python `socartes_cli` command
surface and also keeps the old DeepTutor command names as compatibility
aliases.

```bash
cd backend
cargo run --bin socartes -- --help
cargo run --bin socartes-cli -- --help
cargo run --bin socartes_cli -- --help
cargo run --bin deeptutor -- --help
cargo run --bin deeptutor-cli -- --help
cargo run --bin deeptutor_cli -- --help
cargo run --bin socartes -- run chat "Explain retrieval-augmented generation" --tool rag --kb course-ai
cargo run --bin socartes -- chat --session <session-id>
```

Implemented command groups:

- `run`, `start`, `serve`, `chat`
- `book list|health|refresh-fingerprints`
- `bot list|start|stop|create`
- `kb list|info|set-default|create|add|delete|search`
- `notebook list|create|show|remove-record|add-md|replace-md`
- `memory show|clear`
- `plugin list|info`
- `config show`
- `session list|show|open|delete|rename`
- `provider login`
- `init wizard`

Compatibility binary names:

- `socartes`, `socartes-cli`, and `socartes_cli`
- `deeptutor`, `deeptutor-cli`, and `deeptutor_cli`

`chat --session` reloads saved session preferences before entering the REPL, matching the Python CLI behavior for capability, tools, knowledge bases, language, notebook references, and history references.

`memory show|clear` can use the API when it is available, and falls back to
local `SOCARTES_MEMORY_ROOT` or `SOCARTES_DATA_DIR/memory` files when the API
is offline, matching the Python CLI's local memory workflow for `SUMMARY.md`
and `PROFILE.md`.

Selected OpenAI-compatible chat providers now stream visible `/api/v1/ws`
assistant content chunks when no native tool call is pending. Provider
reasoning chunks remain hidden from visible content and are persisted in
assistant metadata.

The contract suite for this surface is `backend/tests/cli_contract.rs`. It
checks more than help text: API paths, payload shapes, SSE rendering, REPL state
mutation, `init wizard` filesystem side effects, provider login behavior, and
`start` launcher cleanup and port-conflict diagnostics are covered there.

## Repository Structure

```text
.
+-- backend/
|   +-- Cargo.toml
|   +-- Cargo.lock
|   +-- src/
|   |   +-- lib.rs
|   |   +-- main.rs
|   |   +-- bin/
|   |       +-- socartes.rs
|   |       +-- socartes_cli.rs
|   |       +-- socartes_cli_underscore.rs
|   |       +-- deeptutor.rs
|   |       +-- deeptutor_cli.rs
|   |       +-- deeptutor_cli_underscore.rs
|   +-- tests/
|       +-- api_contract.rs
|       +-- cli_contract.rs
|       +-- orchestrator_contract.rs
+-- .gitignore
+-- LICENSE
+-- README.md
```

## Verification

```bash
cd backend
cargo fmt --check
cargo test
cargo check --release
```

Optional HTTP smoke checks while the server is running:

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/openapi.json
curl -fsS http://127.0.0.1:8000/docs
curl -fsS http://127.0.0.1:8000/docs/oauth2-redirect
curl -fsS http://127.0.0.1:8000/redoc
```

## License

MIT
