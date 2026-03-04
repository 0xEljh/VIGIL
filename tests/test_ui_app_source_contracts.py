from __future__ import annotations

import ast
import unittest
from pathlib import Path


def _load_ui_app_ast() -> ast.Module:
    source_path = Path(__file__).resolve().parents[1] / "ui" / "app.py"
    source_text = source_path.read_text(encoding="utf-8")
    return ast.parse(source_text)


def _find_class(tree: ast.Module, class_name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise AssertionError(f"Class {class_name!r} not found")


def _find_method(class_node: ast.ClassDef, method_name: str) -> ast.FunctionDef:
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return node
        if isinstance(node, ast.AsyncFunctionDef) and node.name == method_name:
            return node  # type: ignore[return-value]
    raise AssertionError(f"Method {method_name!r} not found")


class UIAppSourceContractTests(unittest.TestCase):
    def test_transcript_log_wrap_compatibility_guard(self) -> None:
        tree = _load_ui_app_ast()
        app_class = _find_class(tree, "InteractiveConversationApp")
        init_method = _find_method(app_class, "__init__")
        compose = _find_method(app_class, "compose")

        sets_support_flag = False
        for node in ast.walk(init_method):
            if not isinstance(node, ast.Assign):
                continue

            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                    and target.attr == "_log_supports_wrap"
                ):
                    sets_support_flag = True
                    break

        uses_support_flag_in_compose = False
        assigns_wrap_kwarg = False
        for node in ast.walk(compose):
            if isinstance(node, ast.If):
                test_node = node.test
                if (
                    isinstance(test_node, ast.Attribute)
                    and isinstance(test_node.value, ast.Name)
                    and test_node.value.id == "self"
                    and test_node.attr == "_log_supports_wrap"
                ):
                    uses_support_flag_in_compose = True

            if not isinstance(node, ast.Assign):
                continue

            for target in node.targets:
                if not isinstance(target, ast.Subscript):
                    continue
                if not isinstance(target.value, ast.Name):
                    continue
                if target.value.id != "transcript_kwargs":
                    continue

                key_value = None
                if isinstance(target.slice, ast.Constant):
                    key_value = target.slice.value

                if (
                    key_value == "wrap"
                    and isinstance(node.value, ast.Constant)
                    and node.value.value is True
                ):
                    assigns_wrap_kwarg = True
                    break

        self.assertTrue(sets_support_flag)
        self.assertTrue(uses_support_flag_in_compose)
        self.assertTrue(assigns_wrap_kwarg)

    def test_on_unmount_marks_app_as_closing(self) -> None:
        tree = _load_ui_app_ast()
        app_class = _find_class(tree, "InteractiveConversationApp")
        on_unmount = _find_method(app_class, "on_unmount")

        sets_closing_true = False
        for node in ast.walk(on_unmount):
            if not isinstance(node, ast.Assign):
                continue

            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                    and target.attr == "_closing"
                    and isinstance(node.value, ast.Constant)
                    and node.value.value is True
                ):
                    sets_closing_true = True
                    break

        self.assertTrue(sets_closing_true, "on_unmount() must set self._closing = True")

    def test_action_quit_cancels_manager_task(self) -> None:
        tree = _load_ui_app_ast()
        app_class = _find_class(tree, "InteractiveConversationApp")
        action_quit = _find_method(app_class, "action_quit")

        calls_cancel_helper = False
        for node in ast.walk(action_quit):
            if not isinstance(node, ast.Call):
                continue
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "self"
                and node.func.attr == "_cancel_manager_task"
            ):
                calls_cancel_helper = True
                break

        self.assertTrue(
            calls_cancel_helper,
            "action_quit() must cancel the background manager task",
        )

    def test_write_line_handles_transcript_lookup_failures(self) -> None:
        tree = _load_ui_app_ast()
        app_class = _find_class(tree, "InteractiveConversationApp")
        write_line = _find_method(app_class, "_write_line")

        has_try = any(isinstance(node, ast.Try) for node in ast.walk(write_line))
        self.assertTrue(
            has_try,
            "_write_line() must guard against shutdown-time widget lookup failures",
        )

    def test_on_mount_registers_tool_call_callback(self) -> None:
        tree = _load_ui_app_ast()
        app_class = _find_class(tree, "InteractiveConversationApp")
        on_mount = _find_method(app_class, "on_mount")

        registers_tool_call_callback = False
        for node in ast.walk(on_mount):
            if not isinstance(node, ast.Assign):
                continue

            if len(node.targets) != 1:
                continue

            target = node.targets[0]
            if not (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Attribute)
                and isinstance(target.value.value, ast.Name)
                and target.value.value.id == "self"
                and target.value.attr == "_conversation_manager"
                and target.attr == "on_tool_call"
            ):
                continue

            if (
                isinstance(node.value, ast.Attribute)
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "self"
                and node.value.attr == "_on_tool_call"
            ):
                registers_tool_call_callback = True
                break

        self.assertTrue(
            registers_tool_call_callback,
            "on_mount() must register on_tool_call callback",
        )

    def test_defines_on_tool_call_handler(self) -> None:
        tree = _load_ui_app_ast()
        app_class = _find_class(tree, "InteractiveConversationApp")
        _find_method(app_class, "_on_tool_call")


if __name__ == "__main__":
    unittest.main()
