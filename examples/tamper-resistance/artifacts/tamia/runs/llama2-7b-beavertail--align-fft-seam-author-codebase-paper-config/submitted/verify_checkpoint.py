import json
import sys
from pathlib import Path

checkpoint = Path(sys.argv[1])
step = int(sys.argv[2])
state = json.loads((checkpoint / "trainer_state.json").read_text())
assert state["global_step"] == step, state["global_step"]
assert (checkpoint / "config.json").is_file()
weights = list(checkpoint.glob("*.safetensors"))
assert weights and all(path.stat().st_size > 0 for path in weights)
index = checkpoint / "model.safetensors.index.json"
if index.exists():
    shards = set(json.loads(index.read_text())["weight_map"].values())
    assert all((checkpoint / shard).is_file() for shard in shards)
validation = {
    "checkpoint": str(checkpoint),
    "global_step": step,
    "weights": sorted(path.name for path in weights),
    "save_only_model": True,
}
Path(sys.argv[3]).write_text(json.dumps(validation, indent=2) + "\n")
print(json.dumps(validation, indent=2), flush=True)
