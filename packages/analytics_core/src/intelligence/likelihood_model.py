"""LikelihoodModel: derives P(evidence | H true) / P(evidence | H false) for a
candidate experiment from an explicit statistical model, instead of a hand-
authored constant.

Scope (read this before extending): this module currently covers two test
families.

1. "Does one category's share of an additive metric exceed a stated
   threshold" (the EXP-CONC / EXP-ISOLATE experiments) --
   `concentration_share_likelihoods`. A normal approximation to a single
   proportion's sampling distribution, treating the top-ranked (or named)
   category's share as if it were one Bernoulli-style proportion estimated
   from `n_rows` draws. That is a simplification, not a full order-statistic
   model of "the max of k category shares" -- the true sampling distribution
   of a multinomial maximum is more concentrated toward 1/k than this treats
   it, so the false-positive branch here is conservative (likely somewhat too
   favorable to "evidence found under H false") rather than anti-conservative.

2. One-way ANOVA eta-squared (the EXP-ANOVA experiment) --
   `anova_eta_squared_likelihoods`. A standard noncentral-F power calculation
   (the same technique G*Power / statsmodels' FTestAnovaPower use for a
   priori power analysis), not an approximation invented for this project.

Both are real, data-dependent numbers that move with n and the test's own
structural parameters (k, df1/df2), which is what EIG needs and a fixed
constant cannot provide.

Every other test family (uniformity/dispersion, churn rate difference,
correlation, forecast, confounding) is NOT covered yet. Callers must leave
likelihood_if_true/likelihood_if_false as None for those -- per EIGOptimizer's
own contract, that correctly makes EIG unavailable (0.0) for those candidates
rather than inventing a number. Extending coverage means adding another
explicit statistical model here, not a shortcut.
"""
import math
from dataclasses import dataclass
from typing import Optional

from scipy import stats


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


@dataclass
class ShareLikelihoodResult:
    likelihood_if_true: float
    likelihood_if_false: float
    basis: str


def concentration_share_likelihoods(
    n_rows: int,
    n_categories: Optional[int],
    share_threshold: float,
    effect_size_if_true: float,
) -> Optional[ShareLikelihoodResult]:
    """
    Models the top/named category's share of an additive metric as a
    proportion estimated from `n_rows` observations, and asks: what is the
    probability that the observed share clears `share_threshold`, under two
    competing assumptions about the true share?

      - H true:  true share = effect_size_if_true (the concentration claim)
      - H false: true share = 1 / n_categories (uniform null -- no category
                 is systematically favored)

    Returns None (never a guessed number) when the inputs needed to build
    the model aren't available -- e.g. n_categories is unknown or n_rows is
    too small for a normal approximation to mean anything.
    """
    if not n_categories or n_categories < 2:
        return None
    if not n_rows or n_rows < 20:
        return None
    if not (0.0 < share_threshold < 1.0) or not (0.0 < effect_size_if_true < 1.0):
        return None

    p_false = 1.0 / float(n_categories)
    p_true = float(effect_size_if_true)

    def p_exceeds_threshold(true_p: float) -> float:
        se = math.sqrt(max(true_p * (1.0 - true_p), 1e-9) / float(n_rows))
        se = max(se, 1e-9)
        z = (share_threshold - true_p) / se
        return float(max(0.0, min(1.0, 1.0 - _normal_cdf(z))))

    l_true = p_exceeds_threshold(p_true)
    l_false = p_exceeds_threshold(p_false)

    basis = (
        f"normal-approx single-proportion power model: "
        f"P(share>={share_threshold:.2f} | true_share={p_true:.2f}, n={n_rows})={l_true:.3f}; "
        f"P(share>={share_threshold:.2f} | true_share=1/{n_categories}={p_false:.3f}, n={n_rows})={l_false:.3f}"
    )
    return ShareLikelihoodResult(likelihood_if_true=l_true, likelihood_if_false=l_false, basis=basis)


def anova_eta_squared_likelihoods(
    n_rows: int,
    n_groups: Optional[int],
    eta_squared_threshold: float,
    effect_size_if_true: float,
) -> Optional[ShareLikelihoodResult]:
    """
    Models a one-way ANOVA F-test as the standard noncentral-F power
    calculation (the same technique used by G*Power / statsmodels'
    FTestAnovaPower for a priori power analysis), instead of a hand-authored
    constant:

      H true:  the group-effect noncentrality parameter corresponds to
               eta^2 = effect_size_if_true (the hypothesis's own claim).
      H false: eta^2 = 0 (pure null -- groups drawn from the same
               distribution; F ~ central F(df1, df2)).

    Converts the declared eta^2 decision threshold into the F statistic an
    observed eta^2 of exactly that value would produce, then asks: what is
    the probability F clears that value under each assumption?

      lambda = N * f^2,  f^2 = eta^2 / (1 - eta^2)   (Cohen's f^2, standard
      one-way-ANOVA noncentrality relation)

      F_threshold = (eta^2_threshold / df1) / ((1 - eta^2_threshold) / df2)

      l_true  = P(F >= F_threshold | noncentral F(df1, df2, lambda_true))
      l_false = P(F >= F_threshold | central F(df1, df2))

    Returns None (never a guessed number) when df1/df2 aren't well-defined
    (fewer than 2 groups, or fewer rows than groups + 1) or the thresholds
    are outside the (0, 1) range eta^2 must occupy.
    """
    if not n_groups or n_groups < 2:
        return None
    if not n_rows or n_rows <= n_groups:
        return None
    if not (0.0 < eta_squared_threshold < 1.0) or not (0.0 < effect_size_if_true < 1.0):
        return None

    df1 = n_groups - 1
    df2 = n_rows - n_groups
    if df1 < 1 or df2 < 1:
        return None

    f_threshold = (eta_squared_threshold / df1) / ((1.0 - eta_squared_threshold) / df2)

    f_squared_true = effect_size_if_true / (1.0 - effect_size_if_true)
    lambda_true = float(n_rows) * f_squared_true

    l_true = float(1.0 - stats.ncf.cdf(f_threshold, df1, df2, lambda_true))
    l_false = float(1.0 - stats.f.cdf(f_threshold, df1, df2))

    l_true = max(0.0, min(1.0, l_true))
    l_false = max(0.0, min(1.0, l_false))

    basis = (
        f"noncentral-F ANOVA power model (df1={df1}, df2={df2}, "
        f"F_threshold={f_threshold:.3f} for eta^2={eta_squared_threshold:.2f}): "
        f"P(F>=F_threshold | eta^2={effect_size_if_true:.2f}, lambda={lambda_true:.2f})={l_true:.3f}; "
        f"P(F>=F_threshold | eta^2=0 (null))={l_false:.3f}"
    )
    return ShareLikelihoodResult(likelihood_if_true=l_true, likelihood_if_false=l_false, basis=basis)
