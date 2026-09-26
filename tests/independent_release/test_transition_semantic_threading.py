import ast
from pathlib import Path


def test_controller_threads_resolved_semantic_into_transition():
    path = Path(__file__).parents[2] / "packages/analytics_core/src/runtime/controller.py"
    tree = ast.parse(path.read_text())
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "apply_post_execution_transition":
                calls.append(node)
    assert calls, "Controller must call the canonical ScientificTransitionService"
    assert any(
        kw.arg == "semantic" and isinstance(kw.value, ast.Name) and kw.value.id == "semantic"
        for call in calls for kw in call.keywords
    ), "Resolved semantic world model must be threaded into the canonical transition"
