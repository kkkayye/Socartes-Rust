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

## Python Strangler Proxy

The Rust server can sit in front of the existing Python backend while
capabilities are migrated one at a time. Keep Python bound to an internal port
such as `127.0.0.1:8001`, run Rust on the public port, then enable
`migration.toml`:

```toml
enabled = true
python_base_url = "http://127.0.0.1:8001"
python_ws_base_url = "ws://127.0.0.1:8001"
fallback = "proxy"

[routes]
chat = "proxy"
book = "proxy"
knowledge = "proxy"
```

Known capabilities can be switched between `native`, `proxy`, and `shadow`.
`proxy` forwards the request to Python, `native` keeps the Rust handler, and
HTTP/SSE `shadow` returns Python's response while a Rust native copy runs in the
background and logs status, content type, SSE event-type sequence, body length,
hash, truncation, and error differences. The user-visible response is marked
with `x-socartes-migration-mode: shadow`. Reload the file without restarting
Rust:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/admin/migration/reload
```

HTTP and SSE responses are streamed byte-for-byte from Python. WebSocket
endpoints such as `/api/v1/ws`, `/api/v1/book/ws`,
`/api/v1/knowledge/{name}/progress/ws`, tutorbot, quiz, and vision routes use a
bidirectional splice. WebSocket routes in `shadow` mode currently keep the
Python splice as the user-visible transport; native WebSocket double-run/diff is
tracked as the next migration step because Axum's upgraded `WebSocket` cannot be
shared by the Python splice and the native handler at the same time.

During migration, Rust and Python must read the same state. For a systemd
deployment, load the Python `.env` first so auth, PocketBase, model, and search
settings use one source of truth, then pin the Rust process to the same data
directories Python mounts:

```ini
EnvironmentFile=-/path/to/DeepTutor/.env
Environment=SOCARTES_MIGRATION_CONFIG=/path/to/Socartes-Rust/migration.toml
Environment=SOCARTES_KNOWLEDGE_ROOT=/path/to/DeepTutor/data/knowledge_bases
Environment=SOCARTES_SETTINGS_ROOT=/path/to/DeepTutor/data/user/settings
Environment=SOCARTES_MEMORY_ROOT=/path/to/DeepTutor/data/memory
Environment=SOCARTES_BOOK_ROOT=/path/to/DeepTutor/data/user/workspace/book
Environment=SOCARTES_OUTPUT_ROOT=/path/to/DeepTutor/data/user
Environment=SOCARTES_TUTORBOT_ROOT=/path/to/DeepTutor/data/tutorbot
Environment=SOCARTES_NOTEBOOK_ROOT=/path/to/DeepTutor/data/user/workspace/notebook
Environment=SOCARTES_QUESTION_NOTEBOOK_ROOT=/path/to/DeepTutor/data/user/question_notebook
Environment=SOCARTES_SKILLS_ROOT=/path/to/DeepTutor/data/user/workspace/skills
Environment=SOCARTES_CO_WRITER_DOCS_ROOT=/path/to/DeepTutor/data/user/workspace/co-writer/documents
Environment=CHAT_ATTACHMENT_DIR=/path/to/DeepTutor/data/user/workspace/chat/attachments
```

If `AUTH_ENABLED=true`, both processes must use the same `AUTH_SECRET`; if
`POCKETBASE_URL` is configured, both processes must point at the same
PocketBase instance. Otherwise sessions, users, course access, and chat history
can split between the two backends.

## Docker

Build and run the Rust backend image from the repository root:

```bash
docker build -t socartes-rust:local .
docker run --rm -p 8000:8000 -v "$PWD/runtime/data:/app/data" socartes-rust:local
```

The image contains the backend server binary plus CLI compatibility binaries:
`socartes`, `socartes-cli`, `socartes_cli`, `deeptutor`, `deeptutor-cli`, and
`deeptutor_cli`. Runtime data is stored under `/app/data` by default via
`SOCARTES_DATA_DIR`.

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
cargo check --locked
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked --test cli_contract
cargo test --locked --test api_contract
cargo test --locked --test orchestrator_contract
cargo check --release --locked
```

GitHub Actions runs these Rust checks and a Docker image build on pushes and
pull requests to `main`. Published GitHub releases build and push a GHCR image
from `Dockerfile`.

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
