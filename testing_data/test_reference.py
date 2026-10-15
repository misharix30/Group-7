"""Verify literal prompts, unchanged generation functions and reference notebooks."""
import ast
import hashlib
import json
from pathlib import Path

from src import prompts
from src.config import NOTEBOOK_PATH, ROOT


def notebook_trees():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    trees = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            try:
                trees.append(ast.parse("".join(cell["source"])))
            except SyntaxError:
                pass
    return trees


def test_prompts_match_reference_verbatim():
    values = {}
    users = {}
    for tree in notebook_trees():
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if not isinstance(target, ast.Name):
                        continue
                    if target.id in ("VERIFY_SYSTEM_PROMPT", "MEDGEMMA_SYSTEM_PROMPT"):
                        # Only evaluate a known literal followed by .strip().
                        assert isinstance(node.value, ast.Call)
                        assert node.value.func.attr == "strip"
                        values[target.id] = ast.literal_eval(node.value.func.value).strip()
                    if target.id in ("verify_prompt", "medgemma_prompt"):
                        for item in ast.walk(node.value):
                            if isinstance(item, ast.Tuple):
                                try:
                                    role, text = ast.literal_eval(item)
                                    if role == "human":
                                        users[target.id] = text
                                except (ValueError, TypeError):
                                    pass
    assert prompts.VERIFY_SYSTEM_PROMPT == values["VERIFY_SYSTEM_PROMPT"]
    assert prompts.MEDGEMMA_SYSTEM_PROMPT == values["MEDGEMMA_SYSTEM_PROMPT"]
    assert prompts.VERIFY_USER_PROMPT == users["verify_prompt"]
    assert prompts.MEDGEMMA_USER_PROMPT == users["medgemma_prompt"]


def test_generation_functions_match_notebook():
    reference = {node.name: ast.dump(node, include_attributes=False)
                 for tree in notebook_trees() for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name.startswith("generate_")}
    source = ast.parse((ROOT / "src" / "generation.py").read_text(encoding="utf-8"))
    copied = {node.name: ast.dump(node, include_attributes=False)
              for node in ast.walk(source) if isinstance(node, ast.FunctionDef)
              and node.name.startswith("generate_")}
    assert reference == copied


def test_generation_module_imports_and_role_map_matches_notebook():
    # Import exercises module initialization, which AST-only checks miss.
    from src.generation import ROLE_MAP, build_generators
    reference = None
    for tree in notebook_trees():
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "ROLE_MAP"
                for target in node.targets
            ):
                reference = ast.literal_eval(node.value)
    assert reference is not None and ROLE_MAP == reference
    assert callable(build_generators)


def test_reference_notebooks_and_csv_unchanged():
    hashes = json.loads((ROOT / "results" / "reference_hashes.json").read_text(encoding="utf-8"))
    for name, expected in hashes.items():
        if name.endswith(".ipynb") or name.endswith(".csv"):
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
