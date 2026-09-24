import asyncio

import pytest

from automl_agent.config import Settings
from automl_agent.llm import LLMError, create_llm, extract_json
from automl_agent.llm.base import LLMClient, LLMResponse
from automl_agent.schemas.plan import CodeDraft


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('Sure!\n```json\n{"a": 2}\n```') == {"a": 2}
    assert extract_json('noise {"a": 3} trailing') == {"a": 3}
    with pytest.raises(ValueError):
        extract_json("no json here")


class ScriptedLLM(LLMClient):
    provider = "scripted"

    def __init__(self, replies):
        super().__init__("scripted")
        self.replies = list(replies)
        self.seen = []

    async def _complete(self, messages, *, temperature, json_mode):
        self.seen.append(messages)
        return LLMResponse(text=self.replies.pop(0), model=self.model, input_tokens=10, output_tokens=5)


def test_complete_json_repairs_invalid_output():
    llm = ScriptedLLM(["not json", '{"code": "print(1)"}'])
    draft = asyncio.run(llm.complete_json([{"role": "user", "content": "go"}], CodeDraft))
    assert draft.code == "print(1)"
    assert llm.usage.calls == 2 and llm.usage.input_tokens == 20
    assert "invalid" in llm.seen[1][-1]["content"]


def test_complete_json_gives_up():
    llm = ScriptedLLM(["x", "y", "z"])
    with pytest.raises(LLMError):
        asyncio.run(llm.complete_json([{"role": "user", "content": "go"}], CodeDraft, max_repairs=2))


def test_factory_builds_fake():
    llm = create_llm(Settings(llm_provider="fake"))
    assert llm.provider == "fake"


def test_factory_openai_compatible_requires_base_url():
    with pytest.raises(ValueError):
        create_llm(Settings(llm_provider="openai_compatible", llm_model="m"))
