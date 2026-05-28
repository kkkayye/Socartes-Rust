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
  > /tmp/socartes-api-parity-after-book2.json
```

当前计数：

- 前端扫描到的调用：`163`
- Python 原后端路由：`167`
- Rust 后端路由：`72`
- 前端仍缺 Rust 覆盖的路径：`39`

本轮已补齐并验证的主要区域：

- `/api/v1/knowledge/*` 课程/知识库启动、创建、上传、设默认、reindex、任务流、删除
- `/api/v1/sessions/*` 和 `/api/v1/chat/sessions/*` 会话列表、详情、改名、删除、quiz results
- `/api/v1/book/*` Book 首页、file-backed 书籍读取、创建、确认 proposal、确认 spine、编译页面、块编辑、deep dive、quiz attempt、supplement、page chat session、rebuild、health、fingerprint refresh、WebSocket 入口
- `/api/outputs/{path}` 兼容原 DeepTutor public output 静态文件白名单
- `/api/v1/settings*` UI/catalog/apply/themes/sidebar/test SSE/llm-options 兼容入口
- `/api/v1/system*` status/runtime-topology/test LLM/test embeddings/test search 兼容入口

已运行的验证：

- `cargo test`：`19` 个测试通过
- `cargo clippy --all-targets --all-features -- -D warnings`：通过
- `cargo check --release`：通过
- Chromium 本地打开 `http://127.0.0.1:3011/book?book=<id>`：页面 `200`，reader 正文渲染，控制台 `0` 个 error
- Chromium 本地打开 `http://127.0.0.1:3011/settings`：页面 `200`，settings/status/catalog 渲染，控制台 `0` 个 error

下一批 P0 仍未完成：

- `/api/v1/tutorbot/*` 和 `/api/v1/tutorbot/{id}/ws`
- `/api/v1/plugins/list` 与 tool/capability stream
- `/api/v1/co_writer/*`
- `/api/v1/notebook/*`
- `/api/v1/question-notebook/*`
- `/api/v1/memory*`
- `/api/v1/skills*`

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
