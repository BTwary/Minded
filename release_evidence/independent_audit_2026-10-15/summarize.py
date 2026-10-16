import json
from collections import Counter

with open("audit_matrix_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total test cases: {len(data)}")
sev_counts = Counter(x["severity"] for x in data)
eval_counts = Counter(x["eval_category"] for x in data)
print(f"Severities: {dict(sev_counts)}")
print(f"Eval Categories: {dict(eval_counts)}")

by_fam = {}
for x in data:
    fam = x["family"]
    if fam not in by_fam:
        by_fam[fam] = {
            "name": x["family_name"],
            "total": 0,
            "PASS": 0,
            "P1": 0,
            "P2": 0,
            "P0": 0,
            "evals": Counter()
        }
    by_fam[fam]["total"] += 1
    by_fam[fam][x["severity"]] += 1
    by_fam[fam]["evals"][x["eval_category"]] += 1

print("\n| Family ID | Family Name | Total | Pass | P1 | P2 | Pass % | Dominant Evaluation Category |")
print("|---|---|---|---|---|---|---|---|")
for fam in sorted(by_fam.keys()):
    f = by_fam[fam]
    pct = (f["PASS"] / f["total"]) * 100
    top_eval = f["evals"].most_common(1)[0][0]
    print(f"| F{fam:02d} | {f['name']} | {f['total']} | {f['PASS']} | {f['P1']} | {f['P2']} | {pct:.1f}% | {top_eval} |")
