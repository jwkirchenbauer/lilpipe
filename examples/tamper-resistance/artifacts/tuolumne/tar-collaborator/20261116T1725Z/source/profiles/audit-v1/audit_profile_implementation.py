"""Check the complete local call graph of the old and current profile routines."""
import ast
import hashlib
import json
import sys
from pathlib import Path

run, original = map(Path, sys.argv[1:])
old_source = original / "source/jobs/analysis-v6/misc/tar_dynamics.py"
new_source = run / "source/profiles/analysis-v1/tar_dynamics.py"
old_meta = json.loads((original / "cases/baseline-lr1e-6/surfaces/step-profiles.json").read_text())
new_metas = [json.loads(path.read_text()) for path in sorted((run / "cases").glob("*/surfaces/step-profiles.json"))]
assert len(new_metas) == 2 and all(value["state"] == "complete" for value in new_metas)
hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in (old_source, new_source)]
assert hashes[0] == old_meta["source_sha256"]
assert all(hashes[1] == value["source_sha256"] for value in new_metas)
definitions = []
for path in (old_source, new_source):
    definitions.append({node.name: node for node in ast.parse(path.read_text()).body if isinstance(node, (ast.FunctionDef, ast.ClassDef))})
selected, todo = set(), ["step_profiles"]
while todo:
    name = todo.pop()
    if name in selected:
        continue
    selected.add(name)
    assert name in definitions[0] and name in definitions[1], name
    for nodes in definitions:
        todo.extend(node.func.id for node in ast.walk(nodes[name]) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in nodes and node.func.id not in selected)
records = {}
for name in sorted(selected):
    canonical = [ast.dump(nodes[name], include_attributes=False) for nodes in definitions]
    assert canonical[0] == canonical[1], f"Profile implementation differs: {name}"
    records[name] = hashlib.sha256(canonical[0].encode()).hexdigest()
assert all(value["fixed_text_sha256"] == old_meta["fixed_text_sha256"] for value in new_metas)
assert all(value["fixed_source_indices"] == old_meta["fixed_source_indices"] for value in new_metas)
assert all(value["forget_revision"] == old_meta["forget_revision"] for value in new_metas)
result = {
    "state": "passed", "old_source": str(old_source), "new_source": str(new_source),
    "old_module_sha256": hashes[0], "new_module_sha256": hashes[1],
    "profile_call_graph_ast_identical": True, "canonical_ast_sha256": records,
    "fixed_probe_texts_and_revision_identical": True,
    "interpretation": "Full numerical-module files differ, but the profile entry point and every locally called helper/class have exactly equal syntax trees. Full source identities remain separately checked for each cached run. This audit does not replace native endpoint/protocol/input validation.",
}
(run / "profile-implementation-validation.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
