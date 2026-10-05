"""Shared test fixtures.

Agentux's core is async, but the suite runs on a bare pytest install. This
module supplies the small hook that executes ``async def`` tests, so the
project needs no pytest plugin (``pytest-asyncio``) and no asyncio-mode
configuration to be maintained alongside the code.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

from agentux.config import Config
from agentux.providers.mock import MockProvider
from agentux.tools.filesystem import FilesystemTool
from agentux.tools.project import ProjectTool
from agentux.tools.registry import ToolRegistry
from agentux.tools.shell import ShellTool


def pytest_pyfunc_call(pyfuncitem):
    """Run coroutine tests on a fresh event loop."""
    test_function = pyfuncitem.obj
    if not inspect.iscoroutinefunction(test_function):
        return None

    kwargs = {
        name: pyfuncitem.funcargs[name]
        for name in pyfuncitem._fixtureinfo.argnames
    }
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(test_function(**kwargs))
    finally:
        asyncio.set_event_loop(None)
        loop.close()
    return True


def run_async(coroutine):
    """Run a coroutine from a synchronous test."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coroutine)
    finally:
        loop.close()


@pytest.fixture
def config(tmp_path):
    return Config(
        api_key="test-key",
        base_url="http://api.test/v1",
        model="test-model",
        working_dir=str(tmp_path),
        max_steps=10,
    )


@pytest.fixture
def provider():
    return MockProvider([])


@pytest.fixture
def registry(tmp_path):
    return ToolRegistry(
        [
            FilesystemTool(root=str(tmp_path)),
            ShellTool(cwd=str(tmp_path)),
            ProjectTool(root=str(tmp_path)),
        ]
    )
