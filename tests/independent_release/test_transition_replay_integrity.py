from packages.analytics_core.src.intelligence.transition import ScientificTransitionService
from packages.schemas.src.analysis import RawObservationRecord


def test_raw_observation_persists_verification_status():
    obs = RawObservationRecord(observation_id="OBS-X", experiment_id="EXP-X", verification_status="FAILED")
    assert obs.verification_status == "FAILED"


def test_replay_code_does_not_force_verified_status():
    src = ScientificTransitionService.apply_post_execution_transition.__code__
    # Structural guard: replay logic must retrieve persisted status, not a
    # hard-coded VERIFIED value. This intentionally complements the direct
    # schema assertion because creating a full state manager is integration-heavy.
    assert src is not None
    import inspect
    text = inspect.getsource(ScientificTransitionService.apply_post_execution_transition)
    assert 'verification_status="VERIFIED"' not in text.split('# Register experiment execution', 1)[0]
    assert 'existing_obs' in text and 'getattr(existing_obs, "verification_status"' in text
