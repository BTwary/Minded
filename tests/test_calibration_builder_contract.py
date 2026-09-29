import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        good = td / "good.jsonl"
        bad = td / "bad.jsonl"
        good_records = []
        for i in range(20):
            good_records.append({
                "predicted_probability": 0.1, "outcome": 0, "out_of_sample": True, "population_scope": "primary_dataset"
            })
        for i in range(20):
            good_records.append({
                "predicted_probability": 0.9, "outcome": 1, "out_of_sample": True, "population_scope": "primary_dataset"
            })
        good.write_text("".join(json.dumps(r) + "\n" for r in good_records))
        out = td / "profile.json"
        result = subprocess.run([
            sys.executable, str(root / "scripts" / "build_calibration_profile.py"),
            str(good), str(out), "--population-scope", "primary_dataset",
        ], cwd=root, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        payload = json.loads(out.read_text())
        assert payload["out_of_sample_verified"] is True
        assert payload["population_scope"] == "primary_dataset"
        assert payload["fitting_protocol"] == "labeled_holdout_out_of_sample_v1"
        assert len(payload["calibration_source_fingerprint"]) == 64

        bad_records = list(good_records)
        bad_records[0] = {"predicted_probability": 0.1, "outcome": 0, "out_of_sample": True}
        bad.write_text("".join(json.dumps(r) + "\n" for r in bad_records))
        result = subprocess.run([
            sys.executable, str(root / "scripts" / "build_calibration_profile.py"),
            str(bad), str(td / "bad_profile.json"), "--population-scope", "primary_dataset",
        ], cwd=root, capture_output=True, text=True)
        assert result.returncode != 0
        assert "population_scope" in result.stderr

    print("PASS: calibration builder contract 2/2")


if __name__ == "__main__":
    main()
