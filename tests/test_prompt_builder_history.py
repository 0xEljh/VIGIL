from __future__ import annotations

import unittest

from agent.prompt_builder import build_prompt


class PromptBuilderHistoryTests(unittest.TestCase):
    def test_includes_normalized_conversation_history_before_current_turn(self) -> None:
        prompt = build_prompt(
            scene_state={"world_version": 7, "timestamp": 42, "objects": []},
            user_prompt="current request",
            conversation_history=[
                {"role": "user", "content": "  previous question  "},
                {"role": "assistant", "content": " previous answer "},
                {"role": "system", "content": "ignored"},
                {"role": "assistant", "content": "   "},
            ],
        )

        self.assertEqual(prompt.messages[0]["role"], "system")
        self.assertEqual(
            prompt.messages[1], {"role": "user", "content": "previous question"}
        )
        self.assertEqual(
            prompt.messages[2],
            {"role": "assistant", "content": "previous answer"},
        )
        self.assertEqual(prompt.messages[3]["role"], "user")
        self.assertIn("Task:", prompt.messages[3]["content"])
        self.assertIn("current request", prompt.messages[3]["content"])


if __name__ == "__main__":
    unittest.main()
