"""Compare source ASTs only; no real study arrays/plots are evaluated."""

import ast
import hashlib
import json
from pathlib import Path


def ast_string(node):
    return ast.dump(node, include_attributes=False)


class IgnoreApprovedPresentationChanges(ast.NodeTransformer):
    def visit_Expr(self, node):
        call = node.value
        if isinstance(call, ast.Call):
            function = call.func
            if isinstance(function, ast.Name) and function.id == "finish_figure_layout":
                return None
            if (
                isinstance(function, ast.Attribute)
                and isinstance(function.value, ast.Name)
                and function.value.id == "figure"
                and function.attr in {"colorbar", "suptitle", "subplots_adjust"}
            ):
                return None
        return self.generic_visit(node)

    def visit_Constant(self, node):
        if node.value == "aggregate-comparison-v2/final":
            return ast.copy_location(
                ast.Constant(value="aggregate-comparison-v1/final"), node
            )
        return node


def proof(code):
    previous = code.with_name("comparison-v1") / "aggregate.py"
    current = code / "aggregate.py"
    old, new = ast.parse(previous.read_text()), ast.parse(current.read_text())
    before = {node.name: node for node in old.body if isinstance(node, ast.FunctionDef)}
    after = {node.name: node for node in new.body if isinstance(node, ast.FunctionDef)}
    assert set(after) == set(before) | {"finish_figure_layout"}
    unchanged = [
        name for name in before if ast_string(before[name]) == ast_string(after[name])
    ]
    assert set(before) - set(unchanged) == {"draw_grid_matrix", "main"}
    for name in ("draw_grid_matrix", "main"):
        first = IgnoreApprovedPresentationChanges().visit(before[name])
        second = IgnoreApprovedPresentationChanges().visit(after[name])
        assert ast_string(first) == ast_string(
            second
        ), f"Non-presentation source changed: {name}"
    new.body = [
        node
        for node in new.body
        if not isinstance(node, ast.FunctionDef) or node.name != "finish_figure_layout"
    ]
    assert ast_string(IgnoreApprovedPresentationChanges().visit(old)) == ast_string(
        IgnoreApprovedPresentationChanges().visit(new)
    )
    return {
        "state": "passed",
        "aggregate_v1_sha256": hashlib.sha256(previous.read_bytes()).hexdigest(),
        "aggregate_v2_sha256": hashlib.sha256(current.read_bytes()).hexdigest(),
        "exactly_unchanged_function_asts": unchanged,
        "only_changed_existing_functions": ["draw_grid_matrix", "main"],
        "approved_ast_differences": [
            "Dedicated colorbar/panel/suptitle physical layout helper calls replacing only prior layout statements",
            "Default output directory aggregate-comparison-v1/final to aggregate-comparison-v2/final",
        ],
        "remaining_entire_module_ast_equal": True,
        "paired_rendering": "Calls unchanged frozen canonical checkpoint_surface_data/checkpoint_surface_figure/checkpoint_trajectory_figure; only copied display labels, title linebreaks/font, global header position and publication paths change. Runtime surface-data digests must be identical.",
    }


if __name__ == "__main__":
    code = Path(__file__).resolve().parent
    result = proof(code)
    (code / "scientific-equivalence.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))
