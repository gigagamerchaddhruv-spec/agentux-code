"""Tests for the Agentux Solo autonomous research engine.

These exercise the deterministic enforcement of the research cycle: the
engine must search memory before answering, research the web when memory is
insufficient, save what it learns, and verify the save. The model and the web
are mocked so the tests are deterministic and run without network access.
"""

from __future__ import annotations

import asyncio
import os
from typing import List, Optional

import pytest

from agentux.knowledge import KnowledgeStore
from agentux.models import Message, ToolCall
from agentux.providers.base import BaseProvider
from agentux.solo_engine import SoloEngine, SoloEvent


class ScriptedProvider(BaseProvider):
    """A provider that returns a scripted list of messages in order."""

    def __init__(self, responses: List[Message], config=None):
        if config is None:
            from agentux.config import Config

            config = Config(api_key="mock-key")
        super().__init__(config)
        self._responses = list(responses)
        self.calls: List[List[Message]] = []

    async def chat(self, messages, tools=None):
        self.calls.append(list(messages))
        if not self._responses:
            return Message(role="assistant", content="out of responses")
        return self._responses.pop(0)

    def supports_tool_calling(self) -> bool:
        return True


def tool_call(name: str, **arguments) -> Message:
    return Message(
        role="assistant",
        content=None,
        tool_calls=[ToolCall(id="call-1", name=name, arguments=arguments)],
    )


class StubWebSearch:
    """Stands in for WebSearchTool.search() in the engine.

    It is a real ``BaseTool`` so the registry accepts it, but its ``execute``
    delegates to ``search`` so the engine's ``web_search`` tool calls land on
    the same stub the tests assert against.
    """

    def __init__(self, output: str = "1. **Example** - a fact about the topic"):
        self._output = output
        self.queries: List[str] = []

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "stub"

    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer"},
            },
            "required": ["query"],
        }

    def search(self, query: str, max_results: int = 5) -> str:
        self.queries.append(query)
        return self._output

    async def execute(self, query: str, max_results: int = 5, **kwargs) -> str:
        return self.search(query, max_results)

    def to_definition(self):
        from agentux.models import ToolDefinition

        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )


def make_engine(tmp_path, provider, web=None, **kwargs):
    knowledge = KnowledgeStore(path=str(tmp_path / "knowledge.db"))
    engine = SoloEngine(provider, knowledge, web_search=web, **kwargs)
    return engine, knowledge


def collect_events(result):
    return [e.kind for e in result.events]


# ----------------------------------------------------------------------
# Test A: Unknown information -> research -> save -> answer
# ----------------------------------------------------------------------
async def test_unknown_information_triggers_research(tmp_path):
    provider = ScriptedProvider([
        tool_call("web_search", query="who invented the transistor"),
        tool_call("get_knowledge", id="dummy"),
        Message(role="assistant", content="The transistor was invented by Bardeen, Brattain, and Shockley at Bell Labs in 1947."),
    ])
    web = StubWebSearch()
    engine, knowledge = make_engine(tmp_path, provider, web=web)

    result = await engine.run("Who invented the transistor?")

    assert result.researched is True
    assert result.saved is True
    assert "transistor" in result.answer.lower()
    assert "tool_call" in collect_events(result)
    assert web.queries == ["who invented the transistor"]
    # The engine saves on the model's behalf; the model does not need to.
    assert knowledge.count() == 1
    knowledge.close()


# ----------------------------------------------------------------------
# Test B: Repeated question -> memory hit, no web research
# ----------------------------------------------------------------------
async def test_repeated_question_uses_memory(tmp_path):
    knowledge = KnowledgeStore(path=str(tmp_path / "knowledge.db"))
    knowledge.save(
        "Transistor",
        "The transistor was invented by Bardeen, Brattain, and Shockley at Bell Labs in 1947.",
        category="history",
        confidence=0.9,
        verification="verified",
        sources=[{"url": "https://example.com/transistor"}],
    )
    web = StubWebSearch()
    provider = ScriptedProvider([
        Message(role="assistant", content="The transistor was invented by Bardeen, Brattain, and Shockley at Bell Labs in 1947."),
    ])
    engine = SoloEngine(provider, knowledge, web_search=web)
    result = await engine.run("Who invented the transistor?")

    assert web.queries == []
    assert "memory_hit" in collect_events(result)
    assert result.researched is False
    knowledge.close()


# ----------------------------------------------------------------------
# Test C: Partially known information
# ----------------------------------------------------------------------
async def test_partially_known_information_researches_only_the_gap(tmp_path):
    knowledge = KnowledgeStore(path=str(tmp_path / "knowledge.db"))
    knowledge.save(
        "Transistor",
        "The transistor was invented by Bardeen, Brattain, and Shockley at Bell Labs.",
        category="history",
        confidence=0.9,
        verification="verified",
        sources=[{"url": "https://example.com/transistor"}],
    )
    web = StubWebSearch(output="1. **Transistor** - first demonstrated in 1947")
    provider = ScriptedProvider([
        tool_call("web_search", query="when was the transistor first demonstrated"),
        Message(role="assistant",
               content="The transistor was invented by Bardeen, Brattain, and Shockley at Bell Labs, and first demonstrated in December 1947."),
    ])
    engine = SoloEngine(provider, knowledge, web_search=web)
    result = await engine.run("Who invented the transistor, and when was it first demonstrated?")

    assert web.queries == ["when was the transistor first demonstrated"]
    assert "memory_hit" in collect_events(result)
    assert result.researched is True
    assert result.saved is True
    # One pre-existing entry + one auto-saved research result.
    assert knowledge.count() == 2
    knowledge.close()


# ----------------------------------------------------------------------
# Test D: Conflicting information
# ----------------------------------------------------------------------
async def test_conflicting_information_preserves_uncertainty(tmp_path):
    knowledge = KnowledgeStore(path=str(tmp_path / "knowledge.db"))
    knowledge.save(
        "Transistor",
        "The transistor was invented in 1947.",
        category="history",
        confidence=0.5,
        verification="unverified",
        sources=[{"url": "https://example.com/a"}],
    )
    web = StubWebSearch(output="1. **Transistor** - some sources say 1948")
    provider = ScriptedProvider([
        tool_call("web_search", query="transistor invention year"),
        tool_call("save_knowledge", title="Transistor invention year",
                  content="Sources disagree on whether the transistor was invented in 1947 or 1948.",
                  verification="conflicting",
                  sources=[{"url": "https://example.com/b"}]),
        Message(role="assistant",
               content="Sources disagree on whether the transistor was invented in 1947 or 1948."),
    ])
    engine = SoloEngine(provider, knowledge, web_search=web)
    result = await engine.run("When was the transistor invented?")

    assert "disagree" in result.answer.lower()
    assert result.saved is True
    knowledge.close()


# ----------------------------------------------------------------------
# Test E: Research failure
# ----------------------------------------------------------------------
async def test_research_failure_does_not_invent(tmp_path):
    class FailingWeb(StubWebSearch):
        def __init__(self):
            super().__init__()

        def search(self, query, max_results=5):
            self.queries.append(query)
            raise RuntimeError("offline")

    web = FailingWeb()
    provider = ScriptedProvider([
        tool_call("web_search", query="who invented the transistor"),
        Message(role="assistant", content="I could not verify this because the search failed."),
    ])
    engine, knowledge = make_engine(tmp_path, provider, web=web, max_research=1)
    result = await engine.run("Who invented the transistor?")

    assert "could not" in result.answer.lower()
    assert web.queries == ["who invented the transistor"]
    assert result.researched is True
    knowledge.close()


# ----------------------------------------------------------------------
# Test F: Persistence
# ----------------------------------------------------------------------
async def test_knowledge_persists_across_restarts(tmp_path):
    path = str(tmp_path / "knowledge.db")
    store = KnowledgeStore(path=path)
    store.save(
        "Fact",
        "A persistent fact",
        category="general",
        confidence=0.8,
        verification="verified",
        sources=[{"url": "https://example.com"}],
    )
    store.close()

    store2 = KnowledgeStore(path=path)
    hits = store2.search("persistent", limit=5)
    assert len(hits) == 1
    assert hits[0].content == "A persistent fact"
    assert hits[0].verification == "verified"
    store2.close()


# ----------------------------------------------------------------------
# Engine behaviour: mandatory retrieval + enforcement
# ----------------------------------------------------------------------
async def test_engine_searches_memory_before_answering(tmp_path):
    web = StubWebSearch()
    provider = ScriptedProvider([
        Message(role="assistant", content="answer"),
    ])
    engine, knowledge = make_engine(tmp_path, provider, web=web)
    result = await engine.run("anything at all")

    assert "memory_search" in collect_events(result)
    assert web.queries == []
    knowledge.close()


async def test_engine_enforces_research_when_model_skips_it(tmp_path):
    """A model that answers a factual question without researching gets a nudge."""
    web = StubWebSearch()
    provider = ScriptedProvider([
        # Model tries to answer without researching.
        Message(role="assistant", content="I think the answer is 42."),
        # After the nudge, it researches.
        tool_call("web_search", query="the answer to everything"),
        tool_call("save_knowledge", title="Answer", content="The answer is 42.",
                  sources=[{"url": "https://example.com/42"}]),
        Message(role="assistant", content="The answer is 42."),
    ])
    engine, knowledge = make_engine(tmp_path, provider, web=web, max_research=3)
    result = await engine.run("What is the answer to everything?")

    assert result.researched is True
    assert "nudge" in collect_events(result)
    assert web.queries == ["the answer to everything"]
    knowledge.close()


async def test_engine_enforces_save_when_model_skips_it(tmp_path):
    web = StubWebSearch()
    provider = ScriptedProvider([
        tool_call("web_search", query="who discovered gravity"),
        # Model answers without saving.
        Message(role="assistant", content="Newton discovered gravity."),
    ])
    engine, knowledge = make_engine(tmp_path, provider, web=web, max_research=3)
    result = await engine.run("Who discovered gravity?")

    # The engine saves on the model's behalf, so no nudge is needed and the
    # cycle completes without the model ever calling save_knowledge.
    assert result.saved is True
    assert knowledge.count() == 1
    knowledge.close()


async def test_engine_stops_at_max_iterations(tmp_path):
    web = StubWebSearch()
    provider = ScriptedProvider([
        tool_call("web_search", query="x"),
    ] * 50)
    engine, knowledge = make_engine(tmp_path, provider, web=web, max_iterations=2, max_research=5)
    result = await engine.run("tell me something")

    assert result.iterations <= 2
    knowledge.close()


async def test_engine_stops_at_max_tool_calls(tmp_path):
    web = StubWebSearch()
    provider = ScriptedProvider([
        tool_call("web_search", query="x"),
    ] * 100)
    engine, knowledge = make_engine(tmp_path, provider, web=web, max_tool_calls=3)
    result = await engine.run("tell me something")

    assert result.tool_calls <= 3
    knowledge.close()


async def test_engine_records_research_history(tmp_path):
    web = StubWebSearch()
    provider = ScriptedProvider([
        tool_call("web_search", query="who invented the transistor"),
        Message(role="assistant", content="Bardeen, Brattain, Shockley."),
    ])
    engine, knowledge = make_engine(tmp_path, provider, web=web)
    await engine.run("Who invented the transistor?")

    history = knowledge.recent_research(limit=5)
    assert len(history) == 1
    assert "transistor" in history[0].query
    knowledge.close()


async def test_engine_events_are_emitted_in_order(tmp_path):
    web = StubWebSearch()
    provider = ScriptedProvider([
        tool_call("web_search", query="who invented the transistor"),
        tool_call("save_knowledge", title="Transistor",
                  content="The transistor was invented by Bardeen, Brattain, and Shockley.",
                  sources=[{"url": "https://example.com"}]),
        Message(role="assistant", content="Bardeen, Brattain, Shockley."),
    ])
    engine, knowledge = make_engine(tmp_path, provider, web=web)
    result = await engine.run("Who invented the transistor?")

    kinds = [e.kind for e in result.events]
    assert kinds[0] == "start"
    assert kinds[-1] == "done"
    assert "memory_search" in kinds
    assert "tool_call" in kinds
    assert "answer" in kinds
    knowledge.close()


async def test_engine_handles_non_question_without_research(tmp_path):
    web = StubWebSearch()
    provider = ScriptedProvider([
        Message(role="assistant", content="Hello! How can I help?"),
    ])
    engine, knowledge = make_engine(tmp_path, provider, web=web)
    result = await engine.run("hello")

    assert web.queries == []
    assert result.answer
    knowledge.close()


# ----------------------------------------------------------------------
# Knowledge store unit tests
# ----------------------------------------------------------------------
def test_knowledge_store_save_and_get(tmp_path):
    store = KnowledgeStore(path=str(tmp_path / "k.db"))
    entry = store.save("Title", "Content here", category="science", confidence=0.8,
                       verification="verified", sources=[{"url": "https://x"}])
    fetched = store.get(entry.id)
    assert fetched is not None
    assert fetched.title == "Title"
    assert fetched.content == "Content here"
    assert fetched.category == "science"
    assert fetched.confidence == 0.8
    assert fetched.verification == "verified"
    assert len(fetched.sources) == 1
    store.close()


def test_knowledge_store_dedupes_by_content(tmp_path):
    store = KnowledgeStore(path=str(tmp_path / "k.db"))
    a = store.save("T1", "same content")
    b = store.save("T2", "same content")
    assert a.id == b.id
    assert store.count() == 1
    store.close()


def test_knowledge_store_search(tmp_path):
    store = KnowledgeStore(path=str(tmp_path / "k.db"))
    store.save("Transistor", "Invented at Bell Labs in 1947", category="history")
    store.save("Python", "A programming language", category="tech")
    hits = store.search("invented", limit=5)
    assert len(hits) == 1
    assert hits[0].title == "Transistor"
    store.close()


def test_knowledge_store_delete(tmp_path):
    store = KnowledgeStore(path=str(tmp_path / "k.db"))
    entry = store.save("T", "C")
    assert store.delete(entry.id) is True
    assert store.get(entry.id) is None
    assert store.delete(entry.id) is False
    store.close()


def test_knowledge_store_metadata(tmp_path):
    store = KnowledgeStore(path=str(tmp_path / "k.db"))
    store.set_meta("key1", "value1")
    assert store.get_meta("key1") == "value1"
    assert store.get_meta("missing", "default") == "default"
    store.close()


def test_knowledge_store_research_history(tmp_path):
    store = KnowledgeStore(path=str(tmp_path / "k.db"))
    store.record_research("query one", "summary", status="ok")
    store.record_research("query two", "", status="failed")
    history = store.recent_research(limit=10)
    assert len(history) == 2
    assert history[0].query == "query two"
    assert history[1].status == "ok"
    store.close()


def test_knowledge_store_invalid_verification_rejected(tmp_path):
    store = KnowledgeStore(path=str(tmp_path / "k.db"))
    with pytest.raises(ValueError):
        store.save("T", "C", verification="bogus")
    store.close()


# ----------------------------------------------------------------------
# Tool unit tests
# ----------------------------------------------------------------------
async def test_save_knowledge_tool_requires_content(tmp_path):
    from agentux.tools.knowledge_tool import SaveKnowledgeTool

    store = KnowledgeStore(path=str(tmp_path / "k.db"))
    tool = SaveKnowledgeTool(store)
    with pytest.raises(Exception):
        await tool.execute(title="t", content="")
    store.close()


async def test_search_knowledge_tool_returns_no_matches(tmp_path):
    from agentux.tools.knowledge_tool import SearchKnowledgeTool

    store = KnowledgeStore(path=str(tmp_path / "k.db"))
    tool = SearchKnowledgeTool(store)
    result = await tool.execute(query="nothing here")
    assert "No matching" in result
    store.close()


async def test_extract_information_tool_extracts_facts():
    from agentux.tools.knowledge_tool import ExtractInformationTool

    tool = ExtractInformationTool()
    result = await tool.execute(
        material="The sky is blue. Water is wet. This is a sentence fragment.",
        sources=[{"url": "https://example.com"}],
    )
    assert "sky is blue" in result
    assert "Source:" in result


async def test_calculate_tool_works():
    from agentux.tools.knowledge_tool import CalculateTool

    tool = CalculateTool()
    result = await tool.execute(expression="2 + 3 * 4")
    assert "14" in result


async def test_get_datetime_tool_returns_iso():
    from agentux.tools.knowledge_tool import GetDateTimeTool

    tool = GetDateTimeTool()
    result = await tool.execute()
    assert "Current date and time" in result
    assert "UTC" in result


async def test_list_knowledge_tool_empty(tmp_path):
    from agentux.tools.knowledge_tool import ListKnowledgeTool

    store = KnowledgeStore(path=str(tmp_path / "k.db"))
    tool = ListKnowledgeTool(store)
    result = await tool.execute()
    assert "No knowledge entries" in result
    store.close()


async def test_delete_knowledge_tool(tmp_path):
    from agentux.tools.knowledge_tool import DeleteKnowledgeTool

    store = KnowledgeStore(path=str(tmp_path / "k.db"))
    entry = store.save("T", "C")
    tool = DeleteKnowledgeTool(store)
    result = await tool.execute(id=entry.id)
    assert "Deleted" in result
    assert store.get(entry.id) is None
    store.close()


async def test_update_knowledge_tool(tmp_path):
    from agentux.tools.knowledge_tool import UpdateKnowledgeTool

    store = KnowledgeStore(path=str(tmp_path / "k.db"))
    entry = store.save("T", "C", confidence=0.5)
    tool = UpdateKnowledgeTool(store)
    result = await tool.execute(id=entry.id, content="updated content", confidence=0.9)
    assert "Updated" in result
    fetched = store.get(entry.id)
    assert fetched.content == "updated content"
    assert fetched.confidence == 0.9
    store.close()


async def test_get_knowledge_tool(tmp_path):
    from agentux.tools.knowledge_tool import GetKnowledgeTool

    store = KnowledgeStore(path=str(tmp_path / "k.db"))
    entry = store.save("T", "C")
    tool = GetKnowledgeTool(store)
    result = await tool.execute(id=entry.id)
    assert entry.title in result
    store.close()


async def test_search_multiple_tool(tmp_path):
    from agentux.tools.knowledge_tool import SearchMultipleTool

    web = StubWebSearch(output="result")
    tool = SearchMultipleTool(web)
    result = await tool.execute(queries=["a", "b"])
    assert "### a" in result
    assert "### b" in result