"""
Approach 2: Two-Stage Decoupled Model
  Stage 1: Propensity P_ref(x) at rack rate
  Stage 2: Elasticity η(x) — how acceptance shifts with log price ratio

Combined: P(book | x, p) = P_ref(x) * (p / p_ref)^η(x)
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

CAT_COLS = ['room_type', 'channel', 'country', 'device',
            'loyalty_tier', 'trip_purpose']
NUM_COLS = ['lead_time_days', 'length_of_stay', 'is_weekend_stay',
            'prior_bookings', 'occupancy_forecast', 'seasonality_index',
            'price_index']
FEATURES_X = NUM_COLS + CAT_COLS


def _prep(df):
    """Cast categoricals so LightGBM uses them natively."""
    df = df.copy()
    for c in CAT_COLS:
        df[c] = df[c].astype('category')
    return df


# ---------------------------------------------------------------------------
# Stage 1: Propensity P_ref(x)
# ---------------------------------------------------------------------------

class PropensityModel:
    """
    Stage 1: P(book | x) at reference price (rack rate, log_price_ratio = 0).

    Trained on full data with log_price_ratio as a feature, then scored at
    log_price_ratio = 0 to extract baseline acceptance.
    """

    def __init__(self):
        self.model = None
        self.calibrator = None
        self.metrics = {}

    def fit(self, train_df, valid_df):
        train = _prep(train_df)
        valid = _prep(valid_df)

        feats = FEATURES_X + ['log_price_ratio']
        self.model = LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            min_child_samples=100,
            random_state=42,
            verbose=-1,
        )
        self.model.fit(train[feats], train['booked'],
                       categorical_feature=CAT_COLS)

        # calibrate raw probabilities on validation set
        raw = self.model.predict_proba(valid[feats])[:, 1]
        self.calibrator = IsotonicRegression(out_of_bounds='clip').fit(
            raw, valid['booked'])

        cal = self.calibrator.transform(raw)
        self.metrics = {
            'auc': roc_auc_score(valid['booked'], cal),
            'log_loss': log_loss(valid['booked'], cal),
            'brier': brier_score_loss(valid['booked'], cal),
            'mean_p': cal.mean(),
            'mean_y': valid['booked'].mean(),
        }
        self.feats = feats
        return self

    def predict_p_ref(self, df):
        """Score at log_price_ratio = 0 → baseline acceptance at rack rate."""
        df = _prep(df).copy()
        df['log_price_ratio'] = 0.0
        raw = self.model.predict_proba(df[self.feats])[:, 1]
        return self.calibrator.transform(raw)

    def predict_at_price(self, df, log_price_ratio):
        """Helper for diagnostics — score at an arbitrary log price ratio."""
        df = _prep(df).copy()
        df['log_price_ratio'] = log_price_ratio
        raw = self.model.predict_proba(df[self.feats])[:, 1]
        return self.calibrator.transform(raw)


# ---------------------------------------------------------------------------
# Stage 2: Elasticity η(x)
# ---------------------------------------------------------------------------

class ElasticityModel:
    """
    Stage 2: heterogeneous elasticity η(x).

    Implementation: LightGBM with monotone constraint on log_price_ratio
    (enforces η < 0). Elasticity at a given x is estimated numerically
    via finite difference on the log-odds.

    This is the GBM variant of Approach 2 — fast, captures interactions.
    For the hierarchical Bayesian variant, swap in a NumPyro model with
    partial pooling across (segment, hotel) — same inference signature.
    """

    def __init__(self):
        self.model = None
        self.feats = None

    def fit(self, train_df, valid_df):
        train = _prep(train_df)
        valid = _prep(valid_df)
        feats = FEATURES_X + ['log_price_ratio']

        # monotone_constraints: -1 on log_price_ratio enforces η < 0
        mono = [0] * len(feats)
        mono[feats.index('log_price_ratio')] = -1

        self.model = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.04,
            max_depth=6,
            num_leaves=31,
            min_child_samples=100,
            monotone_constraints=mono,
            random_state=42,
            verbose=-1,
        )
        self.model.fit(train[feats], train['booked'],
                       categorical_feature=CAT_COLS)
        self.feats = feats
        return self

    def estimate_eta(self, df, eps=0.05):
        """
        Numerical elasticity:
          η(x) ≈ d log P(book) / d log price
          ≈ (log P(p_ref*(1+eps)) - log P(p_ref*(1-eps))) / log((1+eps)/(1-eps))
        """
        df = _prep(df).copy()
        d_up = df.copy(); d_up['log_price_ratio'] = np.log(1 + eps)
        d_dn = df.copy(); d_dn['log_price_ratio'] = np.log(1 - eps)

        p_up = np.clip(self.model.predict_proba(d_up[self.feats])[:, 1], 1e-6, 1-1e-6)
        p_dn = np.clip(self.model.predict_proba(d_dn[self.feats])[:, 1], 1e-6, 1-1e-6)
        return (np.log(p_up) - np.log(p_dn)) / (np.log(1+eps) - np.log(1-eps))


# ---------------------------------------------------------------------------
# Combined inference
# ---------------------------------------------------------------------------

def _row_to_df(x_row):
    """Convert a Series to a 1-row DataFrame preserving dtypes."""
    if isinstance(x_row, pd.DataFrame):
        return x_row
    return pd.DataFrame([x_row.to_dict()])


def acceptance_curve(x_row, prop_model, elas_model,
                     price_floor=0.55, price_ceiling=1.15, n_grid=60):
    """
    For one shopper, compute P(book) and expected revenue across a price grid.

    Returns DataFrame with columns: price_mult, price, p_book, expected_rev.
    """
    df_row = _row_to_df(x_row)
    p_ref = float(prop_model.predict_p_ref(df_row)[0])
    eta = float(elas_model.estimate_eta(df_row)[0])

    rack = float(df_row['rack_rate'].iloc[0])
    cost = float(df_row['variable_cost'].iloc[0])

    mults = np.linspace(price_floor, price_ceiling, n_grid)
    out = []
    for m in mults:
        p = m * rack
        # Approach 2 combination formula
        p_book = p_ref * (m ** eta)
        p_book = float(np.clip(p_book, 0, 1))
        rev = p_book * (p - cost)
        out.append({'price_mult': m, 'price': p,
                    'p_book': p_book, 'expected_rev': rev,
                    'p_ref': p_ref, 'eta': eta})
    return pd.DataFrame(out)


def optimal_price(x_row, prop_model, elas_model, **kwargs):
    """Pick price that maximizes expected revenue per session."""
    curve = acceptance_curve(x_row, prop_model, elas_model, **kwargs)
    best = curve.loc[curve['expected_rev'].idxmax()]
    return best, curve


def population_policy_eval(df, prop_model, elas_model,
                           price_floor=0.55, price_ceiling=1.15, n_grid=40):
    """Evaluate optimal pricing policy across a population vs. baseline (rack)."""
    df = _prep(df).copy()
    df['p_ref'] = prop_model.predict_p_ref(df)
    df['eta_hat'] = elas_model.estimate_eta(df)

    mults = np.linspace(price_floor, price_ceiling, n_grid)

    # for each row, find argmax over the grid (vectorized)
    best_mult = np.full(len(df), np.nan)
    best_rev = np.full(len(df), -np.inf)
    best_p_book = np.full(len(df), np.nan)

    for m in mults:
        p_book = np.clip(df['p_ref'].values * (m ** df['eta_hat'].values), 0, 1)
        rev = p_book * (m * df['rack_rate'].values - df['variable_cost'].values)
        improved = rev > best_rev
        best_rev = np.where(improved, rev, best_rev)
        best_mult = np.where(improved, m, best_mult)
        best_p_book = np.where(improved, p_book, best_p_book)

    df['optimal_mult'] = best_mult
    df['optimal_price'] = best_mult * df['rack_rate']
    df['optimal_p_book'] = best_p_book
    df['optimal_expected_rev'] = best_rev

    # baseline: charge rack rate (m = 1.0)
    df['baseline_p_book'] = np.clip(df['p_ref'].values * (1.0 ** df['eta_hat'].values), 0, 1)
    df['baseline_expected_rev'] = (df['baseline_p_book']
                                   * (df['rack_rate'] - df['variable_cost']))

    return df
