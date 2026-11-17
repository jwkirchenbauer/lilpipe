"""Add the reviewed benchmark scope without changing frozen training inputs."""

import hashlib
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path


run = Path(sys.argv[1]).resolve()
path = run / "study.json"
original = path.read_bytes()
program = json.loads(original)
assert program["kind"] == "fixed-sampler-hparam-study"
selected = list(program["bundles"]["attacks"])
assert len(selected) == 4 and all(
    program["cases"][name]["role"] == "attack" for name in selected
)
assert (
    "evaluation_cases" not in program
), "Scope was already declared; inspect instead of overwriting"
program["evaluation_cases"] = selected
updated = (json.dumps(program, indent=2) + "\n").encode()
temporary = path.with_name(f".study-evaluation-scope-{uuid.uuid4().hex}.tmp")
try:
    with temporary.open("xb") as stream:
        stream.write(updated)
    assert (
        path.read_bytes() == original
    ), "Study metadata changed concurrently; retry after inspection"
    temporary.replace(path)
finally:
    temporary.unlink(missing_ok=True)
amendment = {
    "time_utc": datetime.now(timezone.utc).isoformat(),
    "changed_fields": ["evaluation_cases"],
    "evaluation_cases": selected,
    "before_sha256": hashlib.sha256(original).hexdigest(),
    "after_sha256": hashlib.sha256(updated).hexdigest(),
    "training_configs_and_frozen_sources_changed": False,
    "reason": "Attack step zero benchmarks the paired defense endpoint. Native CE and landscapes cover defense checkpoint histories; repeated defense benchmarks are outside this prespecified scope.",
}
(run / "evaluation-scope-amendment.json").write_text(
    json.dumps(amendment, indent=2) + "\n"
)
print(json.dumps(amendment, indent=2))
