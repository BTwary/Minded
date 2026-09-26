import json
import tempfile
from pathlib import Path

from packages.analytics_core.src.statistics.probability_calibration import fit_isotonic_calibration, save_calibration_profile, load_calibration_profile


def main():
    y = [0] * 20 + [1] * 20
    p = [0.1] * 20 + [0.9] * 20

    # A profile without explicit out-of-sample provenance must remain unusable.
    unverified = fit_isotonic_calibration(y, p, scope="s", population_scope="pop")
    assert not unverified.is_usable

    verified = fit_isotonic_calibration(
        y, p, scope="s", population_scope="pop",
        out_of_sample_verified=True,
        calibration_source_fingerprint="c" * 64,
        fitting_protocol="labeled_holdout_out_of_sample_v1",
    )
    assert verified.is_usable

    assert verified.is_authorized_for(model_id="AAOS_BELIEF_V1", scope="s", population_scope="pop")
    assert not verified.is_authorized_for(model_id="AAOS_BELIEF_V1", scope="s", population_scope="other")


    # A textual "false" flag must never coerce to True.
    payload = verified.to_dict()
    payload["out_of_sample_verified"] = "false"
    from packages.analytics_core.src.statistics.probability_calibration import CalibrationProfile
    try:
        CalibrationProfile.from_dict(payload)
    except ValueError:
        pass
    else:
        raise AssertionError("String out_of_sample_verified must be rejected")

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "profile.json"
        save_calibration_profile(verified, path)
        loaded = load_calibration_profile(path)
        assert loaded is not None
        assert loaded.is_usable
        payload = json.loads(path.read_text())
        payload["fitting_protocol"] = "in_sample"
        path.write_text(json.dumps(payload))
        try:
            load_calibration_profile(path)
        except ValueError:
            pass
        else:
            raise AssertionError("Tampered fitting protocol must fail integrity validation")

    print("PASS: calibration population authority tests 4/4")


if __name__ == "__main__":
    main()
