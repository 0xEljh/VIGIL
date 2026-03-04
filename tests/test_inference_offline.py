from __future__ import annotations

import unittest
from pathlib import Path

from agent.inference import LlamaCppServerConfig, LlamaCppServerInference


class LlamaCppServerOfflineTests(unittest.TestCase):
    def test_server_command_includes_offline_flag_when_enabled(self) -> None:
        inference = LlamaCppServerInference(
            LlamaCppServerConfig(
                binary_path=Path(__file__),
                offline=True,
            )
        )

        command = inference._build_server_command(cpu_only=False)

        self.assertIn("--offline", command)

    def test_server_command_omits_offline_flag_when_disabled(self) -> None:
        inference = LlamaCppServerInference(
            LlamaCppServerConfig(
                binary_path=Path(__file__),
                offline=False,
            )
        )

        command = inference._build_server_command(cpu_only=False)

        self.assertNotIn("--offline", command)


if __name__ == "__main__":
    unittest.main()
