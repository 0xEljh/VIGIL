from __future__ import annotations

import asyncio
import threading
import unittest

from agent.loop import AgentTurn
from agent.prompt_builder import PromptBundle
from conversation.manager import ConversationManager


class _BlockingFakeAgentLoop:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self._first_call_started = threading.Event()
        self._release_first_call = threading.Event()

    @property
    def first_call_started(self) -> threading.Event:
        return self._first_call_started

    @property
    def release_first_call(self) -> threading.Event:
        return self._release_first_call

    def step(
        self,
        scene_state,
        *,
        user_prompt: str = "",
        auxiliary_context=None,
        conversation_history=None,
        on_model_stdout=None,
        on_model_stderr=None,
        on_tool_call=None,
    ) -> AgentTurn:
        call_index = len(self.calls)
        self.calls.append(
            {
                "scene_state": dict(scene_state),
                "user_prompt": user_prompt,
                "auxiliary_context": auxiliary_context,
                "conversation_history": list(conversation_history or []),
            }
        )

        if call_index == 0:
            self._first_call_started.set()
            self._release_first_call.wait(timeout=2.0)

        return AgentTurn(
            world_version=int(scene_state.get("world_version", -1)),
            scene_timestamp=int(scene_state.get("timestamp", -1)),
            response=f"response-{call_index}",
            prompt=PromptBundle(
                system_prompt="system",
                user_prompt=user_prompt,
                messages=[],
            ),
        )


class _RecordingFakeAgentLoop:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def step(
        self,
        scene_state,
        *,
        user_prompt: str = "",
        auxiliary_context=None,
        conversation_history=None,
        on_model_stdout=None,
        on_model_stderr=None,
        on_tool_call=None,
    ) -> AgentTurn:
        call_index = len(self.calls)
        history = list(conversation_history or [])
        self.calls.append(
            {
                "scene_state": dict(scene_state),
                "user_prompt": user_prompt,
                "auxiliary_context": auxiliary_context,
                "conversation_history": history,
            }
        )

        if on_model_stdout is not None:
            on_model_stdout(f"stream-{call_index}")

        if on_tool_call is not None:
            on_tool_call(
                "describe_scene",
                '{"region": "left"}',
                '{"entity_count": 1}',
            )

        return AgentTurn(
            world_version=int(scene_state.get("world_version", -1)),
            scene_timestamp=int(scene_state.get("timestamp", -1)),
            response=f"response-{call_index}",
            prompt=PromptBundle(
                system_prompt="system",
                user_prompt=user_prompt,
                messages=[],
            ),
        )


class ConversationManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_coalesces_critical_vision_events_during_long_turn(self) -> None:
        state_counter = 0
        aux_calls: list[tuple[int, str]] = []
        fake_agent = _BlockingFakeAgentLoop()

        def state_provider() -> dict[str, int]:
            nonlocal state_counter
            state_counter += 1
            return {
                "world_version": state_counter,
                "timestamp": state_counter * 10,
            }

        def auxiliary_context_builder(scene_state, query_text: str):
            aux_calls.append((int(scene_state.get("world_version", 0)), query_text))
            return {"query": query_text}

        manager = ConversationManager(
            agent_loop=fake_agent,
            state_provider=state_provider,
            auxiliary_context_builder=auxiliary_context_builder,
            history_window_turns=3,
            max_steps=2,
            stream_llm_output=False,
        )

        run_task = asyncio.create_task(manager.run())

        await manager.submit_user_message("user-turn")
        started = await asyncio.to_thread(fake_agent.first_call_started.wait, 2.0)
        self.assertTrue(started)

        manager.notify_critical_vision_event(
            description="critical-event-1",
            world_version=11,
            frame_index=101,
        )
        manager.notify_critical_vision_event(
            description="critical-event-2",
            world_version=12,
            frame_index=102,
        )
        manager.notify_critical_vision_event(
            description="critical-event-3",
            world_version=13,
            frame_index=103,
        )

        fake_agent.release_first_call.set()
        await asyncio.wait_for(run_task, timeout=2.0)

        self.assertEqual(len(fake_agent.calls), 2)
        self.assertEqual(fake_agent.calls[0]["user_prompt"], "user-turn")
        self.assertEqual(fake_agent.calls[1]["user_prompt"], "critical-event-3")
        self.assertEqual(aux_calls[1][1], "critical-event-3")

    async def test_prioritizes_user_messages_over_pending_vision_trigger(self) -> None:
        fake_agent = _RecordingFakeAgentLoop()
        state_counter = 0

        def state_provider() -> dict[str, int]:
            nonlocal state_counter
            state_counter += 1
            return {
                "world_version": state_counter,
                "timestamp": state_counter,
            }

        def auxiliary_context_builder(scene_state, query_text: str):
            return {"query": query_text}

        manager = ConversationManager(
            agent_loop=fake_agent,
            state_provider=state_provider,
            auxiliary_context_builder=auxiliary_context_builder,
            history_window_turns=2,
            max_steps=2,
            stream_llm_output=False,
        )

        run_task = asyncio.create_task(manager.run())
        manager.notify_critical_vision_event(description="vision-priority-check")
        await manager.submit_user_message("user-priority-check")
        await asyncio.wait_for(run_task, timeout=2.0)

        self.assertEqual(len(fake_agent.calls), 2)
        self.assertEqual(fake_agent.calls[0]["user_prompt"], "user-priority-check")
        self.assertEqual(fake_agent.calls[1]["user_prompt"], "vision-priority-check")

    async def test_sliding_window_limits_conversation_history_pairs(self) -> None:
        fake_agent = _RecordingFakeAgentLoop()
        state_counter = 0

        def state_provider() -> dict[str, int]:
            nonlocal state_counter
            state_counter += 1
            return {
                "world_version": state_counter,
                "timestamp": state_counter,
            }

        def auxiliary_context_builder(scene_state, query_text: str):
            return {"query": query_text}

        manager = ConversationManager(
            agent_loop=fake_agent,
            state_provider=state_provider,
            auxiliary_context_builder=auxiliary_context_builder,
            history_window_turns=1,
            max_steps=3,
            stream_llm_output=True,
        )

        streamed_chunks: list[str] = []
        manager.on_turn_chunk = streamed_chunks.append

        run_task = asyncio.create_task(manager.run())
        await manager.submit_user_message("turn-1")
        await manager.submit_user_message("turn-2")
        await manager.submit_user_message("turn-3")
        await asyncio.wait_for(run_task, timeout=2.0)

        self.assertEqual(len(fake_agent.calls), 3)
        self.assertEqual(fake_agent.calls[0]["conversation_history"], [])

        history_turn_2 = fake_agent.calls[1]["conversation_history"]
        self.assertEqual(
            history_turn_2,
            [
                {"role": "user", "content": "turn-1"},
                {"role": "assistant", "content": "response-0"},
            ],
        )

        history_turn_3 = fake_agent.calls[2]["conversation_history"]
        self.assertEqual(
            history_turn_3,
            [
                {"role": "user", "content": "turn-2"},
                {"role": "assistant", "content": "response-1"},
            ],
        )

        self.assertEqual(streamed_chunks, ["stream-0", "stream-1", "stream-2"])

    async def test_emits_tool_call_callback(self) -> None:
        fake_agent = _RecordingFakeAgentLoop()
        state_counter = 0

        def state_provider() -> dict[str, int]:
            nonlocal state_counter
            state_counter += 1
            return {
                "world_version": state_counter,
                "timestamp": state_counter,
            }

        def auxiliary_context_builder(scene_state, query_text: str):
            return {"query": query_text}

        manager = ConversationManager(
            agent_loop=fake_agent,
            state_provider=state_provider,
            auxiliary_context_builder=auxiliary_context_builder,
            history_window_turns=1,
            max_steps=1,
            stream_llm_output=False,
        )

        observed_tool_calls: list[tuple[str, str, str]] = []
        manager.on_tool_call = lambda name, arguments, result: (
            observed_tool_calls.append((name, arguments, result))
        )

        run_task = asyncio.create_task(manager.run())
        await manager.submit_user_message("tool-test")
        await asyncio.wait_for(run_task, timeout=2.0)

        self.assertEqual(
            observed_tool_calls,
            [("describe_scene", '{"region": "left"}', '{"entity_count": 1}')],
        )


if __name__ == "__main__":
    unittest.main()
