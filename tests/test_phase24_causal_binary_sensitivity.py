import numpy as np, pandas as pd
from packages.analytics_core.src.causal.effect_estimation import estimate_binary_treatment_effect
from packages.analytics_core.src.causal.sensitivity import e_value_report

def test_e_value_basic():
    r=e_value_report(treated_risk=.30, control_risk=.15)
    assert r['available'] and r['risk_ratio']==2.0 and r['e_value']>2

def test_binary_aipw_runs_and_reports_risk():
    rng=np.random.default_rng(42); n=160
    x=rng.normal(size=n); p=1/(1+np.exp(-(0.3*x))); a=rng.binomial(1,p)
    pr=1/(1+np.exp(-(-1+0.7*x+0.7*a))); y=rng.binomial(1,pr)
    df=pd.DataFrame({'A':a,'Y':y,'X':x})
    est=estimate_binary_treatment_effect(df,treatment_col='A',outcome_col='Y',adjustment_set=['X'],bootstrap_resamples=25,outcome_type='binary',seed=11)
    assert est.diagnostics['outcome_type']=='binary'
    assert 0 <= est.diagnostics['treated_risk'] <= 1
    assert 0 <= est.diagnostics['control_risk'] <= 1
    assert est.diagnostics['sensitivity']['available']
    assert 'outcome_model' in est.formulas
    assert len(est.confidence_interval)==2
