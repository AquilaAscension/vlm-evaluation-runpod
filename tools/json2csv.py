# tools/json2csv.py
import csv, glob, json, os, sys
root = os.path.expanduser("~/prismatic-vlms/results")
rows = []
for model in glob.glob(f"{root}/*"):
    for ds in glob.glob(f"{model}/*"):
        score_file = f"{ds}/_scores.json"  # produced by score.py
        if not os.path.isfile(score_file):
            continue
        with open(score_file) as fh:
            metrics = json.load(fh)
        for metric, val in metrics.items():
            rows.append({"model": os.path.basename(model),
                         "dataset": os.path.basename(ds),
                         "metric": metric,
                         "value": val})
out = sys.argv[1] if len(sys.argv) > 1 else "vlm_scores.csv"
with open(out, "w", newline="") as fh:
    csv.DictWriter(fh, fieldnames=rows[0].keys()).writeheader()
    csv.DictWriter(fh, fieldnames=rows[0].keys()).writerows(rows)
print("✓ wrote", out)
