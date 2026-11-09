"""Counts why each failed pallet failed, from results/eval_pallets{TAG}.jsonl -> results/failure_breakdown{TAG}.json (TAG = argv[1], e.g. _v1)."""
import collections
import json
import sys

TAG = sys.argv[1] if len(sys.argv) > 1 else ""

P = [json.loads(line) for line in open(f"results/eval_pallets{TAG}.jsonl")]
c = collections.Counter()
for p in P:
    if p["success"]:
        continue
    r = []
    if any(e["event"].startswith("drop") for e in p["events"]):
        r.append("drop")
    if any(e["event"].startswith("fail") for e in p["events"]):
        r.append("ik_fail")
    if p["unstable"]:
        r.append("unstable")
    if p["not_found_by_vision"]:
        r.append("not_found_by_vision")
    c["+".join(r) or "other"] += 1
out = {"failed_pallets": sum(not p["success"] for p in P), "reasons_per_failed_pallet": dict(c),
       "pallets_with_any_unstable": sum(bool(p["unstable"]) for p in P)}
open(f"results/failure_breakdown{TAG}.json", "w").write(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
