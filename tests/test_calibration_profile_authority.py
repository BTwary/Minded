import json
import tempfile
from pathlib import Path
import numpy as np

from packages.analytics_core.src.statistics.probability_calibration import CalibrationProfile, fit_isotonic_calibration, save_calibration_profile, load_calibration_profile


def main():
    y = ([0] * 20) + ([1] * 20)
    p = np.linspace(0.05, 0.95, 40)
    profile = fit_isotonic_calibration(
        y, p, scope="benchmark_a", population_scope="synthetic_churn_v1",
        out_of_sample_verified=True,
        calibration_source_fingerprint="a" * 64,
        fitting_protocol="labeled_holdout_out_of_sample_v1",
    )
    assert profile.is_usable
    assert profile.profile_id
    assert profile.to_dict()["profile_id"] == profile.profile_id

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "profile.json"
        save_calibration_profile(profile, path)
        loaded = load_calibration_profile(path)
        assert loaded is not None
        assert loaded.profile_id == profile.profile_id
        assert loaded.population_scope == "synthetic_churn_v1"

        data = json.loads(path.read_text())
        data["population_scope"] = "different_population"
        path.write_text(json.dumps(data))
        try:
            load_calibration_profile(path)
        except ValueError:
            pass
        else:
            raise AssertionError("Tampered calibration profile must fail integrity validation")

    assert not CalibrationProfile((0.0, 1.0), (0.2, 0.8), 40, scope="benchmark_a", population_scope="known").is_usable
    assert profile.is_authorized_for(
        model_id="AAOS_BELIEF_V1", scope="benchmark_a", population_scope="synthetic_churn_v1"
    )
    assert not profile.is_authorized_for(
        model_id="AAOS_BELIEF_V1", scope="benchmark_a", population_scope="other_population"
    )
    print("PASS: calibration profile authority tests 4/4")

if __name__ == "__main__":
    main()
