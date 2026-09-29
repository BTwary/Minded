"""Session 23: real-dataset stress test through the real InvestigationController.
Each question carries an independently computed pandas ground truth (`truth`) and a
`must` list of substrings/numbers a correct answer should contain."""
import json, os, signal, sys, tempfile, time, traceback
import numpy as np, pandas as pd, seaborn as sns
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from apps.api.src.models.entities import Base, User, Project, Investigation, gen_uuid
from packages.analytics_core.src.runtime.controller import InvestigationController
from packages.analytics_core.src.engines.dataset_provider import InMemoryDatasetProvider
from apps.api.src.models.entities import InvestigationContract
from scripts import stress_verifier as SV

def factory():
    p = os.path.join(tempfile.mkdtemp(), "s.db")
    e = create_engine(f"sqlite:///{p}", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=e)
    return sessionmaker(bind=e)

from packages.analytics_core.src.data.certification_data import load_all_certification_datasets
_dfs = load_all_certification_datasets()
titanic, tips, peng, mpg, dia, planets, taxis, flights = (
    _dfs["titanic"], _dfs["tips"], _dfs["penguins"], _dfs["mpg"],
    _dfs["diamonds"], _dfs["planets"], _dfs["taxis"], _dfs["flights"]
)
import os as _os
variant = _os.environ.get("VARIANT", "raw")
if "--variant" in sys.argv:
    _idx = sys.argv.index("--variant")
    if _idx + 1 < len(sys.argv):
        variant = sys.argv[_idx + 1]

if variant == "clean":
    titanic = titanic.drop(columns=["pclass","alive","adult_male","who","embark_town","alone","deck"], errors="ignore")
    for _d in (titanic, tips, peng, mpg, dia, planets, taxis, flights):
        for _c in _d.select_dtypes("category").columns: _d[_c] = _d[_c].astype(object)
def g(df, by, col, f="mean"): return df.groupby(by, observed=True)[col].agg(f).dropna().round(3).to_dict()
r = lambda a, b: round(float(a.corr(b)), 3)

# (dataset, df, question, truth, category)  category: DESC/CORR/GROUPDIFF/CAUSAL/TREND/RANK/MULTI
Q = [
 ("titanic", titanic, "What is the survival rate by passenger class?", g(titanic,"passenger_class","survived"), "DESC"),
 ("titanic", titanic, "Is survival correlated with fare?", r(titanic.survived, titanic.fare), "CORR"),
 ("titanic", titanic, "Which sex had the highest survival rate?", g(titanic,"sex","survived"), "RANK"),
 ("titanic", titanic, "Did age affect survival?", r(titanic.survived, titanic.age), "CAUSAL"),
 ("titanic", titanic, "Why did third class passengers have lower survival than first class?", g(titanic,"passenger_class","survived"), "CAUSAL"),
 ("titanic", titanic, "Does the effect of class on survival differ between men and women?", titanic.groupby(["sex","passenger_class"],observed=True).survived.mean().round(3).to_dict(), "MULTI"),
 ("tips", tips, "Is tip correlated with total bill?", r(tips.tip, tips.total_bill), "CORR"),
 ("tips", tips, "Which day has the highest average tip?", g(tips,"day","tip"), "RANK"),
 ("tips", tips, "Do smokers tip more than non-smokers?", g(tips,"smoker","tip"), "GROUPDIFF"),
 ("tips", tips, "What is the average tip by day?", g(tips,"day","tip"), "DESC"),
 ("tips", tips, "Does party size drive the tip amount?", r(tips["size"], tips.tip), "CAUSAL"),
 ("penguins", peng, "Is body mass correlated with flipper length?", r(peng.body_mass_g, peng.flipper_length_mm), "CORR"),
 ("penguins", peng, "Which species has the highest average body mass?", g(peng,"species","body_mass_g"), "RANK"),
 ("penguins", peng, "How does bill length differ between species?", g(peng,"species","bill_length_mm"), "GROUPDIFF"),
 ("penguins", peng, "Do male penguins weigh more than female penguins?", g(peng,"sex","body_mass_g"), "GROUPDIFF"),
 ("mpg", mpg, "Is mpg correlated with weight?", r(mpg.mpg, mpg.weight), "CORR"),
 ("mpg", mpg, "Which origin has the highest average mpg?", g(mpg,"origin","mpg"), "RANK"),
 ("mpg", mpg, "Has fuel efficiency improved over model years?", g(mpg,"model_year","mpg"), "TREND"),
 ("mpg", mpg, "Why do cars from the usa have lower mpg?", g(mpg,"origin","mpg"), "CAUSAL"),
 ("mpg", mpg, "Is horsepower or weight more strongly related to mpg?", (r(mpg.mpg,mpg.horsepower), r(mpg.mpg,mpg.weight)), "MULTI"),
 ("diamonds", dia, "Is price correlated with carat?", r(dia.price, dia.carat), "CORR"),
 ("diamonds", dia, "Which cut has the highest average price?", g(dia,"cut","price"), "RANK"),
 ("diamonds", dia, "What is the average price by clarity?", g(dia,"clarity","price"), "DESC"),
 ("diamonds", dia, "Why are Fair cut diamonds priced higher than Ideal cut on average?", g(dia,"cut","price"), "CAUSAL"),
 ("planets", planets, "What is the average mass by discovery method?", g(planets,"method","mass"), "DESC"),
 ("planets", planets, "Which method discovered the most planets?", planets.method.value_counts().head(3).to_dict(), "RANK"),
 ("planets", planets, "Has the orbital period of discovered planets changed over the years?", g(planets,"year","orbital_period"), "TREND"),
 ("taxis", taxis, "Is tip correlated with fare?", r(taxis.tip, taxis.fare), "CORR"),
 ("taxis", taxis, "Which pickup borough has the highest average fare?", g(taxis,"pickup_borough","fare"), "RANK"),
 ("taxis", taxis, "Do credit card payers tip more than cash payers?", g(taxis,"payment","tip"), "GROUPDIFF"),
 ("flights", flights, "Has the number of passengers grown over time?", g(flights,"year","passengers","sum"), "TREND"),
 ("flights", flights, "Which month has the highest average passengers?", g(flights,"month","passengers"), "RANK"),
]
if __name__ == "__main__":
    only = None
    args = sys.argv[1:]
    idx = 0
    while idx < len(args):
        if args[idx] == "--variant":
            idx += 2
        elif args[idx] == "--only":
            only = args[idx + 1] if idx + 1 < len(args) else None
            idx += 2
        else:
            if not args[idx].startswith("-"):
                only = args[idx]
            idx += 1

    class TO(Exception): pass
    def _h(*a): raise TO()
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _h)
    out = []
    for ds, df, q, truth, cat in Q:
        if only and only not in ds and only not in q: continue
        SF = factory(); db = SF()
        u = User(id=f"u-{gen_uuid()[:8]}", email="s@a.ai", hashed_password="x", full_name="S", is_active=True)
        p = Project(id=f"p-{gen_uuid()[:8]}", name="P", owner_id=u.id); db.add_all([u, p]); db.commit()
        inv = Investigation(id=f"I-{gen_uuid()[:8]}", project_id=p.id, user_id=u.id, question=q, status="QUEUED"); db.add(inv); db.commit()
        c = InvestigationController(session_factory=SF, dataset_provider=InMemoryDatasetProvider({ds: df}))
        row = dict(ds=ds, q=q, cat=cat, truth=str(truth)[:400]); t = time.time()
        if hasattr(signal, "alarm"): signal.alarm(180)
        try:
            ok = c.execute_investigation(investigation_id=inv.id, worker_id="w")
            db.expire_all(); x = db.query(Investigation).filter(Investigation.id == inv.id).first()
            ans_str = x.direct_answer or ""
            fnd_str = x.main_finding or ""
            vrd_str = x.verdict_type or ""
            k = (db.query(InvestigationContract).filter_by(investigation_id=inv.id)
                 .order_by(InvestigationContract.version.desc()).first())
            contract = SV.summarize_contract(k, x)
            vd = SV.verify(q, df, status=x.status, verdict_type=x.verdict_type,
                           answer=ans_str, finding=fnd_str, contract=contract)
            row.update(ok=ok, truth_ok=vd.ok, result_class=vd.primary, all_classes=vd.classes,
                       truth_detail="; ".join(vd.details) or "answer + contract match independent ground truth",
                       status=x.status, verdict=x.verdict_type, conf=x.confidence_score,
                       answer=ans_str, finding=fnd_str, contract=contract)
        except TO: row.update(ok=False, truth_ok=False, result_class=SV.UNSUPPORTED, status="TIMEOUT")
        except Exception as e: row.update(ok=False, truth_ok=False, result_class=SV.UNSUPPORTED, status="EXCEPTION", err=f"{type(e).__name__}: {e}", tb=traceback.format_exc(limit=4))
        finally:
            if hasattr(signal, "alarm"): signal.alarm(0)
        row["secs"] = round(time.time() - t, 1); out.append(row); db.close()
        t_label = row.get("result_class", "?")
        print(f"{ds:9} {row['status']:9} {str(row.get('verdict')):26} {row['secs']:5}s [{t_label}] {q}", flush=True)

    from collections import Counter
    total = len(out)
    passed_count = sum(1 for r in out if r.get("truth_ok"))
    print("\nRESULT CLASSES:", dict(Counter(r.get("result_class") for r in out)))
    print("truth_ok:", passed_count, "/", total)
    out_path = _os.environ.get("OUT") or os.path.join(ROOT, "real_stress_results_" + variant + ".json")
    json.dump(out, open(out_path, "w"), indent=1, default=str)
    print("Saved results to", out_path)
    if passed_count != total or total == 0:
        sys.exit(1)
    sys.exit(0)

