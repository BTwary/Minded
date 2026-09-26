import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from packages.analytics_core.src.data.content_identity import compute_content_hash
from packages.analytics_core.src.data.transformation_lineage import make_transformation_record
from packages.analytics_core.src.statistics.calibration import empirical_calibration
from packages.analytics_core.src.governance.claim_gate import admit_positive_claim
from packages.analytics_core.src.governance.ai_authority import validate_ai_proposal


def test_content_hash_is_full_and_row_order_invariant():
    a = pd.DataFrame({"a":[1,2,3], "b":["x","y","z"]})
    b = a.sample(frac=1, random_state=42).reset_index(drop=True)
    c = a.copy(); c.loc[1,"b"]="CHANGED"
    assert compute_content_hash(a) == compute_content_hash(b)
    assert compute_content_hash(a) != compute_content_hash(c)


def test_transformation_lineage_chains_snapshots():
    a = pd.DataFrame({"x":[1,2,3]})
    b = pd.DataFrame({"x":[2,4,6]})
    rec = make_transformation_record(operation="scale", input_df=a, output_df=b, affected_columns=["x"], reason="deterministic test")
    assert rec.input_content_hash == compute_content_hash(a)
    assert rec.output_content_hash == compute_content_hash(b)
    assert rec.record_hash


def test_claim_gate_blocks_unsupported_positive_claim():
    r = admit_positive_claim(verdict_type="DIAGNOSED", directly_tested=False, verified_evidence=True)
    assert not r.allowed
    assert "leading_hypothesis_not_directly_tested" in r.reasons


def test_ai_cannot_invent_columns_or_sql():
    assert validate_ai_proposal({"target":"revenue"}, ["revenue", "region"])["allowed"]
    assert not validate_ai_proposal({"target":"magic_metric"}, ["revenue"])["allowed"]
    assert not validate_ai_proposal({"sql":"SELECT * FROM anything"}, ["revenue"])["allowed"]


def test_empirical_calibration_known_well_calibrated_case():
    y = np.array([0,0,1,1,0,1,1,0,1,0], dtype=float)
    p = np.array([.1,.1,.9,.9,.1,.9,.9,.1,.9,.1])
    r = empirical_calibration(y,p)
    assert r.sample_size == 10
    assert r.brier_score < .02
