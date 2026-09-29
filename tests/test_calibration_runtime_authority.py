from packages.analytics_core.src.intelligence.epistemic_calibration import EpistemicCalibrationEngine
from packages.analytics_core.src.statistics.probability_calibration import CalibrationProfile


def _profile():
    return CalibrationProfile(
        x=(0.0, 0.5, 1.0),
        y=(0.0, 0.6, 1.0),
        sample_size=40,
        scope="scope-A",
        model_id="AAOS_BELIEF_V1",
        population_scope="population-A",
        out_of_sample_verified=True,
        calibration_source_fingerprint="b" * 64,
        fitting_protocol="labeled_holdout_out_of_sample_v1",
    )


def main():
    import os
    previous_scope = os.environ.get("AAOS_CALIBRATION_SCOPE")
    previous_model = os.environ.get("AAOS_CALIBRATION_MODEL")
    profile = _profile()
    os.environ["AAOS_CALIBRATION_SCOPE"] = "scope-A"
    os.environ["AAOS_CALIBRATION_MODEL"] = "AAOS_BELIEF_V1"
    # The profile object is independently valid; runtime authorization must
    # happen before it is supplied to the epistemic engine.
    calibrated = EpistemicCalibrationEngine.calibrate_epistemic_state(
        posterior_probability=0.5,
        sample_size=100,
        calibration_profile=profile,
        runtime_model_id="AAOS_BELIEF_V1",
        runtime_scope="scope-A",
        runtime_population_scope="population-A",
    )
    assert calibrated.calibration_status == "CALIBRATED"
    assert calibrated.calibrated_probability == 0.6
    assert calibrated.calibration_profile_id == profile.profile_id
    assert calibrated.calibration_scope == "scope-A"
    assert calibrated.calibration_model_id == "AAOS_BELIEF_V1"

    # Controller-equivalent mismatch handling: a scope/model mismatch must
    # pass None to the calibration layer, never the untrusted loaded profile.
    runtime_scope = "scope-B"
    runtime_model = "AAOS_BELIEF_V1"
    authorized = profile if (runtime_scope == profile.scope and runtime_model == profile.model_id) else None
    blocked = EpistemicCalibrationEngine.calibrate_epistemic_state(
        posterior_probability=0.5,
        sample_size=100,
        calibration_profile=authorized,
        runtime_model_id=runtime_model,
        runtime_scope=runtime_scope,
        runtime_population_scope="population-A",
    )
    assert blocked.calibration_status == "UNAVAILABLE"
    assert blocked.calibrated_probability is None
    assert blocked.probability_semantics == "MODEL_BASED_BELIEF"

    # Model mismatch must behave identically.
    runtime_scope = "scope-A"
    runtime_model = "OTHER_MODEL"
    authorized = profile if (runtime_scope == profile.scope and runtime_model == profile.model_id) else None
    blocked = EpistemicCalibrationEngine.calibrate_epistemic_state(
        posterior_probability=0.5,
        sample_size=100,
        calibration_profile=authorized,
        runtime_model_id=runtime_model,
        runtime_scope=runtime_scope,
        runtime_population_scope="population-A",
    )
    assert blocked.calibration_status == "UNAVAILABLE"
    assert blocked.calibrated_probability is None

    # Population mismatch must also fail closed even when model and scope match.
    blocked_population = EpistemicCalibrationEngine.calibrate_epistemic_state(
        posterior_probability=0.5,
        sample_size=100,
        calibration_profile=profile,
        runtime_model_id="AAOS_BELIEF_V1",
        runtime_scope="scope-A",
        runtime_population_scope="population-B",
    )
    assert blocked_population.calibration_status == "UNAVAILABLE"
    assert blocked_population.calibrated_probability is None

    # Environment-loaded profiles must fail closed when the runtime scope is absent.
    os.environ.pop("AAOS_CALIBRATION_SCOPE", None)
    os.environ["AAOS_CALIBRATION_MODEL"] = "AAOS_BELIEF_V1"
    blocked_env = EpistemicCalibrationEngine.calibrate_epistemic_state(
        posterior_probability=0.5, sample_size=100, calibration_profile=None
    )
    assert blocked_env.calibration_status == "UNAVAILABLE"
    assert blocked_env.calibrated_probability is None

    if previous_scope is None:
        os.environ.pop("AAOS_CALIBRATION_SCOPE", None)
    else:
        os.environ["AAOS_CALIBRATION_SCOPE"] = previous_scope
    if previous_model is None:
        os.environ.pop("AAOS_CALIBRATION_MODEL", None)
    else:
        os.environ["AAOS_CALIBRATION_MODEL"] = previous_model

    print("PASS: calibration runtime authority 4/4")


if __name__ == "__main__":
    main()
