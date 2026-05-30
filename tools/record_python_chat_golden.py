#!/usr/bin/env python3
"""Record normalized Python Socartes chat events into a Rust golden fixture."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any


STAGE_CHUNKS = [
    ["Checking the selected course before answering.\n"],
    ["No external tools were needed."],
    ["The current course is orbital mirrors."],
]


def _load_fixture(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize(events: list[Any], projection: dict[str, Any]) -> list[dict[str, Any]]:
    event_types = set(projection.get("event_types") or [])
    fields = list(projection.get("fields") or ["type", "stage", "content"])
    normalized: list[dict[str, Any]] = []
    for event in events:
        event_dict = event.to_dict() if hasattr(event, "to_dict") else dict(event)
        if event_types and event_dict.get("type") not in event_types:
            continue
        normalized.append({field: event_dict.get(field) for field in fields})
    return normalized


async def _record_python_events(python_root: Path, fixture: dict[str, Any]) -> list[dict[str, Any]]:
    sys.path.insert(0, str(python_root))

    import socartes.agents.chat.agentic_pipeline as pipeline_mod
    from socartes.core.context import UnifiedContext
    from socartes.core.stream_bus import StreamBus

    class FakeRegistry:
        def build_prompt_text(self, _enabled_tools: Any, **_kwargs: Any) -> str:
            return "- none"

        def get_enabled(self, selected: Any) -> list[Any]:
            return [SimpleNamespace(name=name) for name in selected or []]

    call_index = 0

    async def fake_llm_stream(**_kwargs: Any):
        nonlocal call_index
        chunks = STAGE_CHUNKS[min(call_index, len(STAGE_CHUNKS) - 1)]
        call_index += 1
        for chunk in chunks:
            yield chunk

    pipeline_mod.get_llm_config = lambda: SimpleNamespace(
        binding="openai",
        model="gpt-fixture",
        api_key="sk-fixture",
        base_url="http://fixture.invalid/v1",
        api_version=None,
        extra_headers={},
    )
    pipeline_mod.get_tool_registry = lambda: FakeRegistry()
    pipeline_mod.llm_stream = fake_llm_stream

    pipeline = pipeline_mod.AgenticChatPipeline(language="en")
    pipeline.registry = FakeRegistry()
    payload = fixture["request"]["payload"]
    context = UnifiedContext(
        session_id="python-golden-session",
        user_message=payload["content"],
        enabled_tools=payload.get("tools") or [],
        knowledge_bases=payload.get("knowledge_bases") or [],
        language=payload.get("language") or "en",
        metadata={"turn_id": "python-golden-turn"},
    )
    bus = StreamBus()
    events: list[Any] = []

    async def consume() -> None:
        async for event in bus.subscribe():
            events.append(event)

    consumer = asyncio.create_task(consume())
    await asyncio.sleep(0)
    await pipeline.run(context, bus)
    await bus.close()
    await consumer
    return _normalize(events, fixture["projection"])


async def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--python-root",
        default="/home/coobabm/DeepTutor",
        help="Path containing the Python Socartes package.",
    )
    parser.add_argument(
        "--fixture",
        default="backend/tests/fixtures/llm_golden/python_labelled_thinking_chat.json",
        help="Golden fixture to update or check.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the recorded Python projection differs from the fixture.",
    )
    args = parser.parse_args()

    fixture_path = Path(args.fixture)
    fixture = _load_fixture(fixture_path)
    recorded = await _record_python_events(Path(args.python_root), fixture)
    if args.check:
        expected = fixture.get("expected_events") or []
        if recorded != expected:
            print("Recorded Python events differ from fixture.", file=sys.stderr)
            print(json.dumps({"expected": expected, "recorded": recorded}, indent=2), file=sys.stderr)
            return 1
        return 0

    fixture["expected_events"] = recorded
    fixture_path.write_text(
        json.dumps(fixture, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"updated {fixture_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
