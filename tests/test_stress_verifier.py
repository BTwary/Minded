"""Adversarial tests for scripts/stress_verifier.py.

The verifier is only trustworthy if it FAILS wrong answers.  These tests prove:
  1. generic 'looks substantive' answers with no numbers fail for all 32 specs;
  2. recorded real answers pass only where they are right, and every single-fact
     mutation (wrong number, wrong winner, wrong verdict, wrong aggregation label,
     wrong contract field) turns a CORRECT row into a failure.
"""
import copy, json, os, re, sys
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sns = pytest.importorskip("seaborn")
from scripts import stress_verifier as SV  # noqa: E402

FIX = json.load(open(os.path.join(ROOT, "tests", "fixtures", "stress_answers_v39.json")))


def _frames():
    try:
        from packages.analytics_core.src.data.certification_data import load_all_certification_datasets
        d = load_all_certification_datasets()
    except Exception as e:  # offline
        pytest.skip(f"certification datasets unavailable: {e}")
    for df in d.values():
        for c in df.select_dtypes("category").columns:
            df[c] = df[c].astype(object)
    return d


@pytest.fixture(scope="module")
def frames():
    return _frames()


def _run(row, frames, **over):
    kw = dict(status=row["status"], verdict_type=row["verdict"], answer=row["answer"],
              finding=row["finding"], contract=row["contract"])
    kw.update(over)
    return SV.verify(row["q"], frames[row["ds"]], **kw)


def test_every_question_has_a_spec():
    assert {r["q"] for r in FIX} == set(SV.SPEC_BY_QUESTION)
    assert len(SV.SPECS) == 32


def test_generic_finding_never_passes(frames):
    """The old verifier returned True for any DESC/CAUSAL/TREND/MULTI answer."""
    for r in FIX:
        good_contract = r["contract"]
        v = _run(r, frames, answer="This is a substantive, statistically significant finding "
                 "showing a clear upward trend and a real difference between groups.",
                 finding="verified", contract=good_contract)
        assert not v.ok, f"generic text passed for: {r['q']}"


def test_blank_or_missing_contract_fails(frames):
    for r in FIX:
        assert not _run(r, frames, contract={}).ok
        assert not _run(r, frames, answer="", finding="").ok
        assert not _run(r, frames, status="FAILED").ok


def _bump_numbers(text, factor):
    def rep(m):
        s = m.group(0)
        if "." not in s and len(s.replace(",", "")) < 3:
            return s
        v = float(s.replace(",", "")) * factor
        return f"{v:.{len(s.split('.')[1]) if '.' in s else 0}f}"
    return re.sub(r"(?<![\w.])\d[\d,]*(?:\.\d+)?", rep, text)


def test_numeric_mutation_breaks_correct_rows(frames):
    n = 0
    for r in FIX:
        if not _run(r, frames).ok:
            continue          # only mutate rows the verifier accepts
        n += 1
        v = _run(r, frames, answer=_bump_numbers(r["answer"], 1.13),
                 finding=_bump_numbers(r["finding"], 1.13))
        assert not v.ok, f"13% numeric drift was accepted for: {r['q']}"
    assert n >= 20


def test_contract_mutations_are_caught(frames):
    for r in FIX:
        base = _run(r, frames)
        if not base.ok:
            continue
        for key, bad in (("target", "__wrong__"), ("estimand", "__wrong__"),
                         ("time", ["__wrong__"]), ("requested_aggregation", "SUM_X"),
                         ("claim_ceiling", "CAUSAL")):
            c = copy.deepcopy(r["contract"])
            if key == "requested_aggregation" and c.get("requested_aggregation") == "SUM_X":
                continue
            c[key] = bad
            assert not _run(r, frames, contract=c).ok, f"{key} mutation accepted: {r['q']}"
        c = copy.deepcopy(r["contract"])
        c["grouping"] = sorted(set(c.get("grouping", [])) | {"__spurious__"})
        assert not _run(r, frames, contract=c).ok, f"spurious grouping accepted: {r['q']}"


def test_specific_wrong_answers(frames):
    by = {r["q"]: r for r in FIX}
    # wrong winner
    r = by["Which sex had the highest survival rate?"]
    a = r["answer"].replace("female has the highest", "male has the highest", 1)
    assert not _run(r, frames, answer=a, finding=a).ok
    # flipped Yes/No verdict
    r = by["Do male penguins weigh more than female penguins?"]
    assert not _run(r, frames, answer=re.sub(r"^Yes", "No", r["answer"]), finding="").ok
    # trend direction + aggregation label
    r = by["Has the number of passengers grown over time?"]
    assert not _run(r, frames, answer=r["answer"].replace("trending up", "trending down"),
                    finding="").ok
    assert not _run(r, frames, answer=r["answer"].replace("(total per year)", "(average per year)"),
                    finding="").ok
    # wrong correlation
    r = by["Is mpg correlated with weight?"]
    assert not _run(r, frames, answer=r["answer"].replace("r=-0.83", "r=-0.38"), finding="").ok
    # missing group in a DESC answer
    r = by["What is the average tip by day?"]
    a = r["answer"].replace("Fri: 2.73 (n=19)", "").replace("Fri: 2.735 (n=19)", "")
    assert not _run(r, frames, answer=a, finding="").ok
    # unhedged causal wording
    r = by["Did age affect survival?"]
    assert not _run(r, frames, answer="Age caused lower survival. " + r["answer"]).ok


def test_known_engine_defects_are_reported(frames):
    """Regression pins: these are real defects the verifier must keep surfacing until fixed."""
    by = {r["q"]: r for r in FIX}
    res = {q: _run(r, frames) for q, r in by.items()}
    assert SV.NUMERIC in res["Has the orbital period of discovered planets changed over the years?"].classes
    assert SV.INCOMPLETE in res["What is the average price by clarity?"].classes


def test_pairwise_means_must_be_bound_to_their_labels(frames):
    """Swapping the two means between labels must fail (numbers present, wrong owner)."""
    by = {r["q"]: r for r in FIX}
    r = by["Do male penguins weigh more than female penguins?"]
    a = r["answer"].replace("Male averages 4,546", "Male averages @@").replace("3,862 for Female", "4,546 for Female").replace("@@", "3,862")
    assert not _run(r, frames, answer=a, finding="").ok
    r = by["Do credit card payers tip more than cash payers?"]
    a = r["answer"].replace("credit card averages 2.78 tip vs 0 for cash", "credit card averages 0 tip vs 2.78 for cash")
    assert not _run(r, frames, answer=a, finding="").ok
