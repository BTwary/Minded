"""Build an empirical AA-OS probability calibration profile from labeled, out-of-sample predictions.

Input JSONL records must contain: predicted_probability (0..1), outcome (0/1),
out_of_sample=true, and population_scope matching the requested profile population.
The output profile is marked usable only after those provenance requirements are
validated and its canonical source fingerprint is computed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from packages.analytics_core.src.statistics.probability_calibration import fit_isotonic_calibration, save_calibration_profile


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="JSONL with predicted_probability and outcome")
    ap.add_argument("output", help="output calibration profile JSON")
    ap.add_argument("--scope", default="labeled_out_of_sample")
    ap.add_argument("--model-id", default="AAOS_BELIEF_V1")
    ap.add_argument("--population-scope", required=True, help="Explicit population/sampling scope represented by the calibration data")
    args = ap.parse_args()
    y, p = [], []
    canonical_rows = []
    with Path(args.input).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("out_of_sample") is not True:
                raise ValueError("Every calibration record must explicitly declare out_of_sample=true.")
            if "population_scope" not in row:
                raise ValueError("Every calibration record must explicitly declare population_scope.")
            row_population = str(row["population_scope"])
            if row_population != args.population_scope:
                raise ValueError(
                    f"Calibration record population_scope={row_population!r} does not match the requested profile population_scope={args.population_scope!r}."
                )
            y.append(int(row["outcome"]))
            p.append(float(row["predicted_probability"]))
            canonical_rows.append({
                "predicted_probability": float(row["predicted_probability"]),
                "outcome": int(row["outcome"]),
                "population_scope": row_population,
                "out_of_sample": True,
            })
    if not canonical_rows:
        raise ValueError("No calibration records were supplied.")
    source_payload = json.dumps(canonical_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    source_fingerprint = hashlib.sha256(source_payload).hexdigest()
    profile = fit_isotonic_calibration(
        y, p, scope=args.scope, model_id=args.model_id, population_scope=args.population_scope,
        out_of_sample_verified=True,
        calibration_source_fingerprint=source_fingerprint,
        fitting_protocol="labeled_holdout_out_of_sample_v1",
    )
    save_calibration_profile(profile, args.output)
    print(json.dumps({"status": "CALIBRATION_PROFILE_BUILT", "sample_size": profile.sample_size, "scope": profile.scope, "output": args.output}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
