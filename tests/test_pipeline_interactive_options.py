from __future__ import annotations

import unittest
from pathlib import Path

from orchestration.pipeline import _build_arg_parser, _resolve_interactive_log_path


class PipelineInteractiveOptionsTests(unittest.TestCase):
    def test_resolve_interactive_log_path_defaults_to_artifacts_logs(self) -> None:
        resolved = _resolve_interactive_log_path("")

        self.assertTrue(resolved.is_absolute())
        self.assertEqual(resolved.parent.name, "logs")
        self.assertEqual(resolved.parent.parent.name, "artifacts")
        self.assertTrue(resolved.name.startswith("interactive-"))
        self.assertEqual(resolved.suffix, ".log")

    def test_arg_parser_supports_interactive_transcript_cap_and_log_file(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(
            [
                "--interactive-user-input",
                "--interactive-transcript-max-lines",
                "321",
                "--interactive-log-file",
                "tmp/interactive.log",
            ]
        )

        self.assertTrue(args.interactive_user_input)
        self.assertEqual(args.interactive_transcript_max_lines, 321)
        self.assertEqual(Path(args.interactive_log_file), Path("tmp/interactive.log"))

    def test_arg_parser_supports_offline_mode(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(["--offline"])

        self.assertTrue(args.offline)


if __name__ == "__main__":
    unittest.main()
