# Backend Parity Audit

## 用途

`tools/api_parity_scan.py` 是一个轻量级接口盘点工具，用来给 Socartes-Rust 的后续全量后端替代提供现状对照。

它会同时扫描三侧：

- Rust 替代后端当前显式注册的 `axum` 路由
- DeepTutor Python 原后端的 FastAPI router 路径
- DeepTutor web 前端里常见的 `apiUrl(...)`、`wsUrl(...)`、`fetch(apiUrl(...))` 调用

输出是单个 JSON，适合做这些事情：

- 快速看 Rust 当前覆盖了哪些接口
- 找前端正在调用、但 Rust 还没接住的路径
- 找 Python 原后端存在、但 Rust 还未替代的路径
- 给后续更细的接口语义比对提供起点

## 当前替代目标

当前目标不是精确生成 OpenAPI，也不是验证请求体/响应体完全一致，而是先回答下面几个问题：

- Rust 后端当前暴露了哪些 HTTP 路径
- Python 原后端有哪些 router 路径仍然是基线能力
- 前端实际还在依赖哪些 API / WS 路径
- 三者交集和缺口大致在哪

这更适合做迁移前的 coverage 审计，而不是做最终兼容性证明。

## 如何运行

在 `/home/coobabm/Socartes-Rust` 下运行：

```bash
python3 tools/api_parity_scan.py \
  --rust-root /home/coobabm/Socartes-Rust \
  --deeptutor-root /home/coobabm/.gitnexus/repos/DeepTutor
```

如果只想把结果保存下来继续分析：

```bash
python3 tools/api_parity_scan.py \
  --rust-root /home/coobabm/Socartes-Rust \
  --deeptutor-root /home/coobabm/.gitnexus/repos/DeepTutor \
  > /tmp/socartes-api-parity.json
```

## 输出内容

输出 JSON 主要包含四部分：

- `rust.routes`: 从 `backend/src/lib.rs` 提取的 Rust 路由
- `python.routes`: 从 `deeptutor/api/main.py` 的 `include_router(...)` 前缀和各 router decorator 合成出的 Python 路径
- `frontend.calls`: 从 `web/app`、`web/components`、`web/lib` 中提取的前端调用路径
- `summary`: 基于路径集合生成的计数、交集和缺口摘要

其中 `summary.parity` 会重点列出：

- `rust_vs_python_missing_in_rust`
- `rust_vs_frontend_missing_in_rust`
- `shared_http_paths_all_three`
- WebSocket 相关缺口

## 2026-05-29 当前快照

本快照来自：

```bash
python3 tools/api_parity_scan.py \
  --rust-root /home/coobabm/Socartes-Rust \
  --deeptutor-root /home/coobabm/.gitnexus/repos/DeepTutor \
  > /tmp/socartes-api-parity.json
```

当前计数：

- 前端扫描到的调用：`163`
- Python 原后端路由：`167`
- Rust 后端路由：`170`
- Python 原后端仍缺 Rust 覆盖的路径：`0`
- 前端仍缺 Rust 覆盖的路径：`1`，为 `/api/v1/book{param}` 扫描伪影

本轮已补齐并验证的主要区域：

- `/api/v1/knowledge/*` 课程/知识库启动、创建、上传、设默认、reindex、任务流、删除、health、configs/config sync、default get、per-KB config、progress、linked-folder/sync-folder 管理入口
- 课程 RAG index-version 状态：本地课程 reindex 会创建 Python-like `version-N/meta.json`，返回 `signature`/`noop`，并在 list/detail statistics 暴露 `index_versions`、`active_signature`、`active_match`、`rag_initialized`、`needs_reindex`
- 课程 RAG 持久化索引和向量检索：本地课程 reindex 现在会读取已保存的 embedding catalog，为 chunk 批量生成 embedding，并写入 `version-N/chunks.json` 与 `default__vector_store.json.embedding_dict`；检索优先读取 active signature 的 chunk vectors，对 query 使用同一 embedding 配置生成向量后按 cosine similarity 排序，source 输出保留旧 `source_id/title/content/confidence` 并补充 Python/LlamaIndex 风格 `provider/source/chunk_id/page/score`
- 课程资料解析：`/api/v1/knowledge/supported-file-types` 现在对齐 Python `FileTypeRouter` 的 parser + text-like 扩展白名单，包含 `.pdf/.docx/.xlsx/.pptx` 和常见文本/代码格式；索引与未索引 fallback 检索共用同一抽取入口，文本文件使用 UTF-8/BOM、GBK、Windows-1252 fallback，PDF 保留 `--- Page N ---` marker，DOCX/XLSX/PPTX 通过 OOXML ZIP/XML 抽取正文、sheet 和 slide 文本；坏 parser 文件在 KB loader 语义下跳过，不会把二进制垃圾写入 chunk index
- 课程上传增量索引：`/api/v1/knowledge/{name}/upload` 现在要求课程已有 active index；当课程配置或 metadata 标记 `needs_reindex`、或课程尚未 reindex 出 active index 时返回 `409`，避免在 stale/missing index 上假成功；完成 reindex 后上传会重写 active `chunks.json`、维护 `metadata.json.file_hashes`、追加 `update_history`，并跳过已存在课程内容的重复上传而不改 metadata
- 课程 RAG 检索边界：选定上传课程时只从该课程返回匹配 source；无匹配时返回空 sources，不再回退到内置 `socartes-rust-rag` 资料造成来源串库
- `/api/v1/sessions/*` 和 `/api/v1/chat/sessions/*` 会话列表、详情、改名、删除、quiz results
- `/api/v1/ws` unified chat WebSocket：`start_turn/message` 会持久化完整 turn event 序列，`llm_selection` 会按 catalog 中同一 profile 下的 model id 精确校验，无效 selection 返回 `error(status=rejected)` 且不创建 turn；`subscribe_turn`/`resume_from` 可按 `seq` 回放已完成 turn 的尾部事件，也可在 turn 运行中接上 live events；`cancel_turn` 支持同连接、跨连接和已落盘 running turn 取消，并广播/补写 `error(status=cancelled)` + `done(status=cancelled)`；`regenerate` 复用上一条 user、删除尾部 assistant、写入 regenerate metadata 且不重复 user message；无效 regenerate `llm_selection` 会在删改历史前 rejected，保留原 user/assistant 历史；`subscribe_session` 可回放 session 最新 turn
- `/api/v1/ws` selected LLM runtime：当 turn 选择非 `deterministic-agent-loop` 的 OpenAI-compatible profile/model 时，Rust 会从 settings catalog 解析 `base_url/api_key/api_version/extra_headers/model`，调用 `{base_url}/chat/completions` 的 non-streaming 成功路径，发送 `system + bounded history + current user` messages 和 `stream:false`，并把 provider answer 写入 `content` event 与持久化 assistant message；provider HTTP/JSON/content 错误会发 terminal `error(status=failed)` + `done(status=failed)`，session/turn 标为 `failed`，不写入假的 assistant completed 回答
- `/api/v1/book/*` Book 首页、file-backed 书籍读取、创建、确认 proposal、确认 spine、编译页面、块编辑、deep dive、quiz attempt、supplement、page chat session、rebuild、health、fingerprint refresh、WebSocket 入口
- `/api/outputs/{path}` 兼容原 DeepTutor public output 静态文件白名单
- `/api/v1/settings*` UI/catalog/apply/themes/sidebar/test SSE/llm-options 兼容入口
- `/api/v1/system*` status/runtime-topology/test LLM/test embeddings/test search 兼容入口；`/api/v1/system/test/embeddings` 现在按 Python 语义执行真实 batch probe，校验两条向量返回、非空和维度一致，并区分配置错误、连接错误和 invalid response；OpenAI-compatible/Azure 会处理 `api-version`、`api-key` 和 `extra_headers`，Cohere v2/Ollama 会使用各自的 payload 与 response shape
- `/api/v1/settings/tests/embedding/start` + `/events` 现在按 Python settings runner 语义保存 catalog 快照、返回 `embedding-<10 hex>` run id，SSE 执行真实 embedding probe，成功时发送 `capabilities`/`response`/`catalog`/`completed`，失败时发送 terminal `failed` 且不误报 completed；成功 probe 会从已知模型能力表刷新 `default_dim`、`supported_dimensions`、`supports_variable_dimensions`、`model_known` 并写回 catalog
- `/api/v1/tutorbot*` TutorBot 列表、创建/启动、停止/销毁、详情/PATCH、recent、souls CRUD、channels schema、profile 文件、history、WebSocket 最小聊天入口
- `/api/v1/notebook/*` 普通 Notebook 列表、统计、创建、详情、更新、删除、记录增删改、带 summary 的 SSE 保存入口
- `/api/v1/question-notebook/*` 题目 Notebook entries、lookup/upsert、分类 CRUD、entry/category 关联与筛选
- `/api/v1/memory*` two-file public memory：`SUMMARY.md`/`PROFILE.md` 快照、单文件保存/清空、从会话刷新、缺失 session/非法 file 错误兼容
- `/api/v1/skills*` file-backed `SKILL.md` 管理：技能 list/detail/create/update/rename/delete、默认 tag 词表、tag create/rename/delete、tag 级联重写、frontmatter scalar 转义、symlink 目录拒绝、前端 JSON DELETE 合同
- `/api/v1/plugins/*` Playground plugins list、tool execute、tool SSE、capability SSE 兼容入口
- `/api/v1/page-agent/openai/v1/chat/completions` Page agent OpenAI-compatible fallback，返回 `AgentOutput` tool call，保持前端 pet/page-agent 可解析
- `/api/v1/co_writer/documents*` Co-Writer file-backed 文档列表、创建、读取、更新、删除，兼容 12 位 hex id、标题推导、preview、`updated_at` 倒序和 `Document not found` 错误
- `/api/v1/co_writer/edit`、`/api/v1/co_writer/automark`、`/api/v1/co_writer/edit_react`、`/api/v1/co_writer/edit_react/stream` Co-Writer 编辑、自动标注、selection ReAct 普通响应与 SSE 兼容入口
- `/api/v1/co_writer/history`、`/api/v1/co_writer/history/{operation_id}`、`/api/v1/co_writer/tool_calls/{operation_id}`、`/api/v1/co_writer/export/markdown` Co-Writer 历史、tool call JSON、Markdown 导出兼容入口
- `/api/attachments/{session_id}/{attachment_id}/{filename}` chat attachment preview/download 兼容入口，并兼容旧扫描器看到的四段 alias 形态
- `/api/v1/settings/sidebar/description`、`/api/v1/settings/sidebar/nav-order`、`/api/v1/settings/tour/status|complete|reopen` 旧设置和 tour 入口
- `/api/v1/dashboard/recent`、`/api/v1/dashboard/{entry_id}`、`/api/v1/agent-config/agents*`、`/api/v1/solve/sessions*` legacy dashboard/agent-config/solve session 兼容入口
- `/api/v1/vision/analyze` legacy REST surface：参数校验、无图请求、base64/url 输入优先级、valid data URI metadata-only 降级响应、`image_url` 真实下载、HTTP status/content-type/10MB 大小限制和 unsupported format 错误兼容
- `/api/v1/vision/solve` legacy WebSocket surface：真实握手、无图 `session -> no_image -> done` 事件，以及 valid image metadata-only `analysis_start -> bbox_complete -> analysis_complete -> ggbscript_complete -> reflection_complete -> analysis_message_complete -> answer_start -> done` 事件兼容

已运行的验证：

- `cargo fmt --check`：通过
- `cargo test`：`69` 个 API contract 测试 + `5` 个 orchestrator 测试通过
- `cargo clippy --all-targets --all-features -- -D warnings`：通过
- `cargo check --release`：通过
- `python3 tools/api_parity_scan.py --rust-root /home/coobabm/Socartes-Rust --deeptutor-root /home/coobabm/.gitnexus/repos/DeepTutor`：Python missing `0`，frontend missing 仅 `/api/v1/book{param}` 扫描伪影
- `cargo test --test api_contract course`：`3` 个课程/知识库契约测试通过
- `cargo test --test api_contract knowledge_reindex_creates_signature_version_and_reports_active_match`：课程 reindex signature/version/noop/active_match 契约测试通过
- `cargo test --test api_contract knowledge_reindex_persists_chunk_index_and_rag_uses_indexed_chunks`：课程 reindex 持久化 chunk index、plugin RAG 和 chat sources 使用 chunk 级 source 的契约测试通过
- `cargo test --test api_contract knowledge_rag_uses_embedding_similarity_over_keyword_overlap`：课程 RAG 会按 embedding/cosine similarity 选中语义匹配 chunk，而不是按关键词重叠选中 decoy chunk；同时验证 `chunks.json` 和 `default__vector_store.json.embedding_dict` 写入向量
- `cargo test --test api_contract office_documents_upload_reindex_and_rag_extract_text`：DOCX/XLSX/PPTX 上传、文件 MIME、reindex 后 chunk 文本抽取、sheet/slide marker、plugin RAG source 回指 Office 文件的契约测试通过
- `cargo test --test api_contract corrupt_parser_documents_are_skipped_instead_of_indexing_binary_garbage`：坏 DOCX/PDF parser 文件会被跳过，混合 KB 仍索引好文本文件且不写入坏 parser source 的契约测试通过
- `cargo test --test api_contract knowledge_upload`：`2` 个课程上传契约测试通过，覆盖 missing/stale active index 返回 `409`、上传后 active chunk index 可检索、`file_hashes`/`update_history` 写入，以及重复内容不写入索引也不改 metadata
- `cargo test --test api_contract rag_selected_uploaded_kb_returns_no_builtin_fallback_when_no_match`：选定上传课程无命中时不回退内置 RAG 的语义契约测试通过
- `cargo test knowledge_python_config_progress_and_linked_folder_endpoints_match_contract`：Python-only knowledge 管理兼容端点契约测试通过
- `cargo test --test api_contract co_writer`：`3` 个 Co-Writer 文档、编辑、历史/tool-calls/export 契约测试通过
- `cargo test --test api_contract attachment_preview_route_serves_local_chat_files_like_python_contract`：chat attachment preview/download 契约测试通过
- `cargo test --test api_contract legacy_settings_dashboard_agent_config_and_solve_routes_match_python_contracts`：legacy settings/dashboard/agent-config/solve 契约测试通过
- `cargo test --test api_contract vision_`：`4` 个 Vision REST/WebSocket 校验、REST image URL 下载/content-type 错误、真实 WS 握手和事件序列契约测试通过
- `cargo test --test api_contract chat_ws_`：`12` 个 unified chat WebSocket 真实握手、turn event 持久化、`llm_selection` 无效 selection rejected、selected OpenAI-compatible LLM `/chat/completions` 成功路径、provider 失败 terminal error、`subscribe_turn`/`resume_from` 回放、运行中 live subscribe、发起 socket 断开后完整 replay、同连接/跨连接/落盘 running turn 取消、regenerate 消息替换和无效 selection 不破坏历史契约测试通过
- `cargo test --test api_contract embedding_test`：`9` 个 system/settings embedding diagnostic 契约测试通过，覆盖不可达 provider 失败、system 固定 Python batch probe、Azure `api-version`/`api-key`/`extra_headers`、Cohere v2 payload/response、Ollama payload/response、settings detected dimension 写回且不发送 `dimensions`、已知模型能力表刷新、provider 失败 SSE、空 vector terminal failed 文案
- `cargo test skills_`：`5` 个 Skills API 契约测试通过
- `cargo test plugins_`：`3` 个 Playground plugins API 契约测试通过
- `cargo test page_agent_chat_completion`：`2` 个 Page agent OpenAI-compatible 契约测试通过
- `cargo test co_writer_documents_crud_matches_frontend_contract`：Co-Writer 文档 CRUD 契约测试通过
- `cargo test book_`：`4` 个 Book/Notebook 相关契约测试通过
- Chromium 本地打开 `http://127.0.0.1:3011/book?book=<id>`：页面 `200`，reader 正文渲染，控制台 `0` 个 error
- Chromium 本地打开 `http://127.0.0.1:3011/settings`：页面 `200`，settings/status/catalog 渲染，控制台 `0` 个 error
- Chromium 本地打开 `http://127.0.0.1:3011/agents`：页面 `200`，TutorBot 列表和 tabs 渲染，控制台 `0` 个 error
- Node WebSocket 直连 `ws://127.0.0.1:8810/api/v1/tutorbot/<id>/ws`：收到 `thinking,content,done`

当前剩余高风险差距：

- 前端扫描里的 `/api/v1/book{param}` 是 `web/lib/book-api.ts` 中 `BASE + path` 包装导致的模板扫描伪影；真实运行路径是 `/api/v1/book/books`、`/api/v1/book/books/{book_id}` 等，Rust 已覆盖这些 Book 路由
- `/api/v1/vision/analyze` 和 `/api/v1/vision/solve` 的路由、校验、REST image URL 下载、真实 WS 握手和 metadata-only 事件序列已补齐；但真实 VisionSolver/GeoGebra/LLM 图像分析流水线还没有移植到 Rust，这仍是语义级差距，不是路由级缺口
- parity scan 已无 Python-only 路由缺口，但这不等于所有端点都完成了完整业务语义迁移；课程 RAG 已补 selected-KB no-fallback、本地 index-version 状态合同、embedding 持久化和 query/vector cosine retrieval；仍未做到完整 LlamaIndex node/docstore JSON schema、精确 token chunking 和 legacy index 迁移全兼容；unified WS 已补 replay/live subscribe/cancel/regenerate 的协议级语义，并具备 selected OpenAI-compatible LLM non-streaming 成功路径和 provider failure terminal failed 语义，但仍未完整移植 Python ChatOrchestrator 的 streaming 阶段、Responses API、tool_calls/tool execution loop、reasoning/usage 解析、provider retry/error mapping、attachments 多模态注入、memory/skills/context builder 全量消息构造；LLM、Vision、TutorBot 等仍需要继续做真实回放和语义 contract test

## 已知限制

这是轻量审计，不是完整语义分析器，当前限制包括：

- Rust 只扫描 `backend/src/lib.rs` 里的 `.route("...", get/post/put/patch/delete(...))`，并有限识别 `get(...).post(...)` 这类 axum 链式 method router
- Python 只合成 `main.py` 里 `include_router(...)` 能看见的 prefix，并提取 `@router.get/post/put/delete/patch/websocket(...)`
- 前端只扫描 `web/app`、`web/components`、`web/lib` 下常见 TS/JS 文件
- 前端对模板字符串只做有限展开：会解析简单常量，复杂表达式会保留成 `{expr}`
- 前端 `apiUrl(...)` 本身不一定代表真实发起了请求；只有 `fetch(apiUrl(...))` 会尽量推断 HTTP method
- 不校验 query 参数、请求体、响应体、状态码、认证要求、流式协议或业务语义
- 不跟踪 Next rewrite、代理规则或运行时注入的动态 URL 构造

如果后续要做“真正的 parity 验证”，还需要再补：

- 请求/响应 schema 对比
- 认证与错误语义对比
- 流式接口与 WebSocket 事件格式对比
- 逐路径的真实回放或 contract test
