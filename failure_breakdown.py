"""Counts why each failed pallet failed, from results/eval_pallets.jsonl -> results/failure_breakdown.json."""
import collections
import json

P = [json.loads(line) for line in open("results/eval_pallets.jsonl")]
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
open("results/failure_breakdown.json", "w").write(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
