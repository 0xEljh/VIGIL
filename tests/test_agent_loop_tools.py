from __future__ import annotations

import unittest

from agent.inference import ChatCompletionResult, ToolCall
from agent.loop import AgentLoop
from agent.prompt_builder import PromptBundle


class _FakeDispatcher:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def dispatch(self, tool_name: str, arguments: str) -> str:
        self.calls.append((tool_name, arguments))
        return '{"entity_count": 1}'


class _InferenceWithToolCall:
    def __init__(self) -> None:
        self.complete_messages: list[list[dict[str, object]]] = []

    def generate_with_tools(
        self,
        prompt: PromptBundle,
        *,
        tools: list[dict[str, object]],
        on_stderr=None,
    ) -> ChatCompletionResult:
        del prompt, tools, on_stderr
        return ChatCompletionResult(
            content=None,
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name="describe_scene",
                    arguments='{"region": "left"}',
                )
            ],
            finish_reason="tool_calls",
        )

    def complete(
        self,
        messages: list[dict[str, object]],
        *,
        max_tokens=None,
        temperature=None,
        on_stdout=None,
        on_stderr=None,
    ) -> str:
        del max_tokens, temperature, on_stdout, on_stderr
        self.complete_messages.append(messages)
        return "final answer after tool"

    def generate(self, prompt, on_stdout=None, on_stderr=None) -> str:
        del prompt, on_stdout, on_stderr
        raise AssertionError("generate() should not be called in tool path")


class _InferenceWithoutToolCall:
    def __init__(self) -> None:
        self.complete_called = False

    def generate_with_tools(
        self,
        prompt: PromptBundle,
        *,
        tools: list[dict[str, object]],
        on_stderr=None,
    ) -> ChatCompletionResult:
        del prompt, tools, on_stderr
        return ChatCompletionResult(
            content="direct response",
            tool_calls=[],
            finish_reason="stop",
        )

    def complete(
        self,
        messages,
        *,
        max_tokens=None,
        temperature=None,
        on_stdout=None,
        on_stderr=None,
    ):  # noqa: ANN001, E501
        del messages, max_tokens, temperature, on_stdout, on_stderr
        self.complete_called = True
        return "should not happen"

    def generate(self, prompt, on_stdout=None, on_stderr=None) -> str:
        del prompt, on_stdout, on_stderr
        raise AssertionError("generate() should not be called in tool-enabled path")


def _prompt_builder(*args, **kwargs) -> PromptBundle:  # noqa: ANN002, ANN003
    del args, kwargs
    return PromptBundle(
        system_prompt="system",
        user_prompt="user",
        messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ],
    )


class AgentLoopToolTests(unittest.TestCase):
    def test_runs_tool_call_then_requests_final_response(self) -> None:
        inference = _InferenceWithToolCall()
        dispatcher = _FakeDispatcher()
        loop = AgentLoop(
            inference=inference,
            prompt_builder=_prompt_builder,
            tool_dispatcher=dispatcher,
            tool_schemas=[{"type": "function", "function": {"name": "describe_scene"}}],
        )

        turn = loop.step(scene_state={"world_version": 1, "timestamp": 2})

        self.assertEqual(turn.response, "final answer after tool")
        self.assertEqual(dispatcher.calls, [("describe_scene", '{"region": "left"}')])
        self.assertEqual(len(inference.complete_messages), 1)

        completion_messages = inference.complete_messages[0]
        self.assertEqual(completion_messages[-2]["role"], "assistant")
        self.assertEqual(completion_messages[-1]["role"], "tool")
        self.assertEqual(completion_messages[-1]["tool_call_id"], "call_1")
        self.assertEqual(completion_messages[-1]["name"], "describe_scene")

    def test_returns_model_text_when_no_tool_call(self) -> None:
        inference = _InferenceWithoutToolCall()
        dispatcher = _FakeDispatcher()
        loop = AgentLoop(
            inference=inference,
            prompt_builder=_prompt_builder,
            tool_dispatcher=dispatcher,
            tool_schemas=[{"type": "function", "function": {"name": "describe_scene"}}],
        )

        turn = loop.step(scene_state={"world_version": 3, "timestamp": 4})

        self.assertEqual(turn.response, "direct response")
        self.assertEqual(dispatcher.calls, [])
        self.assertFalse(inference.complete_called)

    def test_emits_tool_call_callback_with_result(self) -> None:
        inference = _InferenceWithToolCall()
        dispatcher = _FakeDispatcher()
        loop = AgentLoop(
            inference=inference,
            prompt_builder=_prompt_builder,
            tool_dispatcher=dispatcher,
            tool_schemas=[{"type": "function", "function": {"name": "describe_scene"}}],
        )

        emitted_calls: list[tuple[str, str, str]] = []
        turn = loop.step(
            scene_state={"world_version": 5, "timestamp": 6},
            on_tool_call=lambda name, arguments, result: emitted_calls.append(
                (name, arguments, result)
            ),
        )

        self.assertEqual(turn.response, "final answer after tool")
        self.assertEqual(
            emitted_calls,
            [("describe_scene", '{"region": "left"}', '{"entity_count": 1}')],
        )


if __name__ == "__main__":
    unittest.main()
