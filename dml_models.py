"""
Stage 2 — DML Variant
=====================

Double Machine Learning (Chernozhukov, Chetverikov, Demirer, Duflo, Hansen,
Newey, Robins, 2018) for log-log price elasticity in the partial linear model:

    Y = θ(X)·T + g(X) + ε,    E[ε | X, T] = 0

where Y = booked, T = log_price_ratio, X = features.

Procedure
---------
1. Cross-fit nuisance models with K-fold (out-of-fold predictions):
     g̃(X) = E[Y | X]  via LightGBM classifier
     m̃(X) = E[T | X]  via LightGBM regressor
2. Residualize:
     ã = Y − g̃(X)
     p̃ = T − m̃(X)
3. Final stage — heterogeneous via segment interactions, OLS with HC1 robust SE:
     ã = Σⱼ βⱼ (p̃ · Z_j(X)) + u
   where Z(X) is a one-hot of segment combos. The estimated heterogeneous
   elasticity is  η̂(X) = Σⱼ βⱼ · Z_j(X).

Why DML instead of the naive Stage 2 GBM?
-----------------------------------------
• Orthogonalization removes 1st-order bias from observed confounders.
  In hotel pricing, rack rates respond to occupancy, seasonality, lead time —
  all of which also correlate with booking propensity. Naive ML attributes
  some of that demand signal to "price", biasing η̂ toward zero.
• Cross-fitting prevents own-sample overfitting bias in the nuisance models.
• Linear final stage gives analytical confidence intervals (HC1 robust).
• No monotone constraint → can recover positive elasticity (Veblen / status
  segments) when true η > 0. The naive monotone-GBM clamps these at 0.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import clone
from sklearn.model_selection import KFold
from lightgbm import LGBMClassifier, LGBMRegressor

from models import CAT_COLS, FEATURES_X, _prep


class DMLElasticityModel:
    """
    DML-based heterogeneous elasticity estimator, segment-level by default.

    Drop-in replacement for ElasticityModel — exposes the same `.estimate_eta(df)`
    interface, so it plugs into `optimal_price` and `population_policy_eval`
    without any other changes.

    Parameters
    ----------
    n_folds : int, default 5
        K for K-fold cross-fitting of nuisance models.
    segment_cols : tuple
        Columns combined to form the heterogeneity segment.
    nuisance_lr, nuisance_estimators, nuisance_max_depth :
        LightGBM hyperparameters for the nuisance models.
    """

    def __init__(self,
                 n_folds: int = 5,
                 segment_cols: tuple = ('trip_purpose', 'loyalty_tier'),
                 nuisance_lr: float = 0.05,
                 nuisance_estimators: int = 200,
                 nuisance_max_depth: int = 4,
                 nuisance_min_child: int = 500,
                 random_state: int = 42):
        self.n_folds = n_folds
        self.segment_cols = list(segment_cols)
        self.nuisance_lr = nuisance_lr
        self.nuisance_estimators = nuisance_estimators
        self.nuisance_max_depth = nuisance_max_depth
        self.nuisance_min_child = nuisance_min_child
        self.random_state = random_state

    # ------------------------------------------------------------------ utils
    def _segment_label(self, df):
        s = df[self.segment_cols[0]].astype(str)
        for c in self.segment_cols[1:]:
            s = s + '_' + df[c].astype(str)
        return s

    def _build_basis(self, df):
        """Z(X): one-hot matrix of segment combinations."""
        seg = self._segment_label(df)
        Z = pd.get_dummies(seg, prefix='seg').astype(float)
        return Z, seg

    def _cross_fit_predict(self, base_model, X_df, y, classifier: bool):
        """Out-of-fold predictions via K-fold cross-fitting."""
        out = np.zeros(len(y))
        kf = KFold(n_splits=self.n_folds, shuffle=True,
                   random_state=self.random_state)
        for tr, te in kf.split(X_df):
            m = clone(base_model)
            y_tr = y.iloc[tr] if hasattr(y, 'iloc') else y[tr]
            m.fit(X_df.iloc[tr], y_tr, categorical_feature=CAT_COLS)
            if classifier:
                out[te] = m.predict_proba(X_df.iloc[te])[:, 1]
            else:
                out[te] = m.predict(X_df.iloc[te])
        return out

    # ------------------------------------------------------------------- fit
    def fit(self, df):
        df = _prep(df).reset_index(drop=True)

        # ---- Stage A: cross-fit nuisance models -----------------------------
        # Note: nuisance models are intentionally regularized (shallow trees,
        # large min_child_samples). DML theory requires nuisance error to
        # converge faster than n^(-1/4); aggressive regularization helps,
        # AND prevents overfitting the structural part of price so that
        # there's still residual variation left for identification.
        g_model = LGBMClassifier(
            n_estimators=self.nuisance_estimators,
            learning_rate=self.nuisance_lr,
            max_depth=self.nuisance_max_depth,
            min_child_samples=self.nuisance_min_child,
            random_state=self.random_state, verbose=-1)
        m_model = LGBMRegressor(
            n_estimators=self.nuisance_estimators,
            learning_rate=self.nuisance_lr,
            max_depth=self.nuisance_max_depth,
            min_child_samples=self.nuisance_min_child,
            random_state=self.random_state, verbose=-1)

        g_hat = self._cross_fit_predict(g_model, df[FEATURES_X],
                                         df['booked'], classifier=True)
        m_hat = self._cross_fit_predict(m_model, df[FEATURES_X],
                                         df['log_price_ratio'], classifier=False)

        # ---- Stage B: residualize ------------------------------------------
        a_tilde = df['booked'].values - g_hat
        p_tilde = df['log_price_ratio'].values - m_hat

        # ---- Stage C: heterogeneous final stage (probability scale) ---------
        # OLS final stage gives θ such that
        #     P(book) ≈ θ(X) · T + g(X)        (probability-linear model)
        Z, seg_label = self._build_basis(df)
        X_design = p_tilde[:, None] * Z.values
        col_names = list(Z.columns)
        ols = sm.OLS(a_tilde, X_design).fit(cov_type='HC1')

        theta_prob = pd.Series(ols.params, index=col_names)
        theta_se   = pd.Series(ols.bse,    index=col_names)

        # ---- Stage D: convert to log-probability-linear scale --------------
        # The pricing formula p_book = p_ref · m^η uses constant-elasticity:
        #     d log P(book) / d log price = η  (constant)
        # Probability-linear θ converts via   η = θ / P̄_segment
        # (first-order Taylor expansion at the segment's mean booking rate).
        seg_pbar = df.groupby(seg_label)['booked'].mean()
        seg_pbar.index = ['seg_' + s for s in seg_pbar.index]

        eta_const_elas = (theta_prob / seg_pbar).reindex(col_names)
        eta_const_se   = (theta_se   / seg_pbar).reindex(col_names)

        ci_table = pd.DataFrame({
            'theta_prob':  theta_prob.values,        # probability-scale
            'theta_se':    theta_se.values,
            'eta':         eta_const_elas.values,    # log-prob (constant elasticity) scale
            'eta_se':      eta_const_se.values,
            'lo95':        (eta_const_elas - 1.96 * eta_const_se).values,
            'hi95':        (eta_const_elas + 1.96 * eta_const_se).values,
            'p_bar':       seg_pbar.reindex(col_names).values,
        }, index=[c.replace('seg_', '') for c in col_names])

        # ---- global pooled estimate (for diagnostic) -----------------------
        global_num = (p_tilde * a_tilde).sum()
        global_den = (p_tilde ** 2).sum()
        global_theta = global_num / max(global_den, 1e-12)
        u_hat = a_tilde - global_theta * p_tilde
        n = len(a_tilde)
        meat = ((p_tilde * u_hat) ** 2).sum()
        global_theta_se = np.sqrt(n / (n - 1) * meat / (global_den ** 2))
        # convert to constant-elasticity scale at global mean
        y_bar = df['booked'].mean()
        global_eta = global_theta / y_bar
        global_eta_se = global_theta_se / y_bar

        # ---- store ----------------------------------------------------------
        self.eta_by_seg_ = eta_const_elas
        self.theta_by_seg_ = theta_prob
        self.eta_table_ = ci_table.sort_values('eta')
        self.basis_cols_ = col_names
        self.global_eta_ = float(global_eta)
        self.global_eta_se_ = float(global_eta_se)
        self.global_eta_ci_ = (global_eta - 1.96 * global_eta_se,
                                global_eta + 1.96 * global_eta_se)

        # diagnostics
        self.diagnostics_ = {
            'n_train': n,
            'n_segments': len(col_names),
            'g_hat_corr_with_y': float(np.corrcoef(g_hat, df['booked'])[0, 1]),
            'm_hat_corr_with_T': float(np.corrcoef(m_hat, df['log_price_ratio'])[0, 1]),
            'p_tilde_std': float(p_tilde.std()),
            'a_tilde_std': float(a_tilde.std()),
            'mean_y': float(y_bar),
            'global_theta_prob': float(global_theta),
            'global_eta_const_elas': float(global_eta),
            'global_eta_ci': self.global_eta_ci_,
            'ab_test_share': float(df.get('ab_test_arm', pd.Series([0])).mean()),
        }
        return self

    # ------------------------------------------------------------ inference
    def estimate_eta(self, df):
        """Return η̂(x) for each row — same interface as ElasticityModel."""
        df = _prep(df).copy()
        Z, _ = self._build_basis(df)
        for c in self.basis_cols_:
            if c not in Z.columns:
                Z[c] = 0.0
        Z = Z[self.basis_cols_]
        return Z.values @ self.eta_by_seg_.values

    def estimate_eta_with_ci(self, df, alpha=0.05):
        """Per-row η̂ with the 95% CI of its segment."""
        df = _prep(df).copy()
        Z, seg = self._build_basis(df)
        eta = self.estimate_eta(df)
        # map each row to its segment's CI
        seg_to_ci = self.eta_table_[['lo95', 'hi95']]
        ci_lo = seg.map(seg_to_ci['lo95']).values
        ci_hi = seg.map(seg_to_ci['hi95']).values
        return pd.DataFrame({'eta': eta, 'lo95': ci_lo, 'hi95': ci_hi},
                            index=df.index)

    def segment_table(self):
        """Tidy per-segment elasticity table with 95% CIs (constant-elasticity scale)."""
        return self.eta_table_.copy()


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------

def compare_elasticity_models(valid_df, true_eta_col, naive_model, dml_model):
    """
    Side-by-side comparison of true / naive GBM / DML elasticity by segment.

    Returns a DataFrame with mean true η, naive η̂, DML η̂, plus DML 95% CI per segment.
    """
    df = valid_df.copy()
    df['eta_naive'] = naive_model.estimate_eta(df)
    df['eta_dml'] = dml_model.estimate_eta(df)
    df['true_eta'] = df[true_eta_col]

    seg_label = (df['trip_purpose'].astype(str)
                 + '_' + df['loyalty_tier'].astype(str))
    df['segment'] = seg_label

    out = (df.groupby('segment')
             .agg(true_eta=('true_eta', 'mean'),
                  eta_naive=('eta_naive', 'mean'),
                  eta_dml=('eta_dml', 'mean'),
                  n=('booked', 'size'))
             .reset_index())

    # add DML CIs from segment_table
    dml_ci = dml_model.segment_table()[['lo95', 'hi95']].rename_axis('segment').reset_index()
    out = out.merge(dml_ci, on='segment', how='left')

    # bias relative to truth
    out['naive_bias'] = out['eta_naive'] - out['true_eta']
    out['dml_bias'] = out['eta_dml'] - out['true_eta']
    return out


# ---------------------------------------------------------------------------
# Off-policy revenue evaluator
# ---------------------------------------------------------------------------

def offpolicy_revenue_eval(df, prop_model, eta_model,
                            true_eta_col='true_eta',
                            price_floor=0.55, price_ceiling=1.15, n_grid=60):
    """
    Counterfactual revenue evaluation: pick prices using `eta_model`, then
    score the chosen prices using TRUE elasticity to get unbiased revenue.

    This is how you should compare pricing models in a synthetic/simulation
    setting. In production, replace `true_eta_col` with an IPS / DR estimator.

    Returns
    -------
    dict with:
      'mean_actual_rev', 'mean_rack_rev', 'lift_pct',
      'self_estimated_lift_pct',  # what the model thought it would get
      'mean_chosen_mult',
      'per_segment': DataFrame with actual vs self-estimated revenue per segment
    """
    df = df.copy().reset_index(drop=True)
    p_ref = prop_model.predict_p_ref(df)
    eta_model_est = eta_model.estimate_eta(df)
    true_eta = df[true_eta_col].values

    rack = df['rack_rate'].values
    cost = df['variable_cost'].values
    mults = np.linspace(price_floor, price_ceiling, n_grid)

    chosen_mult = np.full(len(df), 1.0)
    self_est_rev = np.full(len(df), -np.inf)

    for m in mults:
        # The model picks price using its OWN eta (this is what gets deployed)
        p_book_est = np.clip(p_ref * (m ** eta_model_est), 0, 1)
        rev_est = p_book_est * (m * rack - cost)
        better = rev_est > self_est_rev
        self_est_rev = np.where(better, rev_est, self_est_rev)
        chosen_mult = np.where(better, m, chosen_mult)

    # Now evaluate ACTUAL revenue at the chosen prices using true eta
    actual_p_book = np.clip(p_ref * (chosen_mult ** true_eta), 0, 1)
    actual_rev = actual_p_book * (chosen_mult * rack - cost)

    # Baseline: charge rack rate (m = 1)
    rack_p_book = np.clip(p_ref, 0, 1)
    rack_rev = rack_p_book * (rack - cost)

    # Per segment breakdown
    seg = (df['trip_purpose'].astype(str) + '_' + df['loyalty_tier'].astype(str))
    per_seg = (pd.DataFrame({
        'segment': seg,
        'self_est_rev': self_est_rev,
        'actual_rev': actual_rev,
        'rack_rev': rack_rev,
        'chosen_mult': chosen_mult,
    }).groupby('segment').agg(
        actual_rev=('actual_rev', 'mean'),
        self_est_rev=('self_est_rev', 'mean'),
        rack_rev=('rack_rev', 'mean'),
        chosen_mult=('chosen_mult', 'mean'),
        n=('actual_rev', 'size'),
    ).reset_index())
    per_seg['actual_lift_%'] = (per_seg['actual_rev'] / per_seg['rack_rev'] - 1) * 100
    per_seg['self_est_lift_%'] = (per_seg['self_est_rev'] / per_seg['rack_rev'] - 1) * 100
    per_seg['overconfidence_%'] = per_seg['self_est_lift_%'] - per_seg['actual_lift_%']

    return {
        'mean_actual_rev': float(actual_rev.mean()),
        'mean_self_est_rev': float(self_est_rev.mean()),
        'mean_rack_rev': float(rack_rev.mean()),
        'lift_pct': float((actual_rev.mean() / rack_rev.mean() - 1) * 100),
        'self_estimated_lift_pct': float((self_est_rev.mean() / rack_rev.mean() - 1) * 100),
        'mean_chosen_mult': float(chosen_mult.mean()),
        'overconfidence_pct': float((self_est_rev.mean() - actual_rev.mean())
                                     / rack_rev.mean() * 100),
        'per_segment': per_seg,
    }
