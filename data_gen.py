"""
Synthetic hotel booking data generator.

Encodes a 'true' data generating process where:
- Baseline booking probability depends on shopper features (loyalty, prior history, channel)
- Price elasticity varies by segment (business inelastic, leisure elastic; lead time matters)
- Discount/price variation is wide enough to identify elasticity (mimics promo + dynamic pricing variation)
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

ROOM_TYPES = ['standard', 'deluxe', 'suite']
CHANNELS = ['direct_web', 'direct_app', 'ota', 'meta']
COUNTRIES = ['IN', 'US', 'UK', 'AE', 'SG']
DEVICES = ['ios', 'android', 'desktop']
LOYALTY_TIERS = ['none', 'silver', 'gold', 'platinum']
TRIP_PURPOSE = ['business', 'leisure']


def generate_sessions(n=20000, seed=42, ab_test_share=0.30):
    """Generate synthetic shopper sessions for hotels.

    Parameters
    ----------
    n : int
        Number of sessions.
    seed : int
        Random seed.
    ab_test_share : float, default 0.30
        Fraction of sessions assigned to a price A/B test arm with a uniformly
        random price multiplier in [0.60, 1.10] applied to the rack rate.
        This represents the kind of randomized pricing that mature hotels run
        for elasticity learning. It gives Stage 2 (and especially DML) a clean
        source of identification: for these rows, price is independent of
        features by construction. Set to 0 to disable (DML will then suffer
        from weak identification — try it!).
    """
    rng = np.random.default_rng(seed)

    # ---- Shopper / stay features ----
    df = pd.DataFrame({
        'session_id': [f'S_{i:06d}' for i in range(n)],
        'lead_time_days': rng.choice([1, 3, 7, 14, 30, 60, 90, 120],
                                     size=n, p=[.08,.12,.18,.20,.18,.12,.07,.05]),
        'length_of_stay': rng.choice([1, 2, 3, 4, 5, 7],
                                     size=n, p=[.30,.30,.18,.10,.07,.05]),
        'is_weekend_stay': rng.binomial(1, 0.45, size=n),
        'room_type': rng.choice(ROOM_TYPES, size=n, p=[.55,.35,.10]),
        'channel': rng.choice(CHANNELS, size=n, p=[.30,.20,.40,.10]),
        'country': rng.choice(COUNTRIES, size=n, p=[.45,.20,.15,.10,.10]),
        'device': rng.choice(DEVICES, size=n, p=[.35,.40,.25]),
        'loyalty_tier': rng.choice(LOYALTY_TIERS, size=n, p=[.55,.25,.15,.05]),
        'prior_bookings': rng.poisson(2.0, size=n),
        'trip_purpose': rng.choice(TRIP_PURPOSE, size=n, p=[.35,.65]),
        'occupancy_forecast': np.clip(rng.beta(5, 3, size=n), 0.20, 0.98),
        'seasonality_index': np.clip(rng.normal(1.0, 0.20, size=n), 0.60, 1.50),
    })

    # business trips skew to short lead time, weekday, high loyalty
    biz_mask = df['trip_purpose'] == 'business'
    df.loc[biz_mask, 'lead_time_days'] = rng.choice([1,3,7,14], size=biz_mask.sum(),
                                                    p=[.30,.35,.25,.10])
    df.loc[biz_mask, 'is_weekend_stay'] = rng.binomial(1, 0.15, size=biz_mask.sum())
    df.loc[biz_mask, 'length_of_stay'] = rng.choice([1,2,3], size=biz_mask.sum(),
                                                    p=[.45,.40,.15])

    # ---- Rack rate driven by room type, seasonality, day-of-week, occupancy ----
    base_rate = df['room_type'].map({'standard': 6500, 'deluxe': 9500, 'suite': 16500})
    df['rack_rate'] = (base_rate
                       * df['seasonality_index']
                       * (1 + 0.15 * df['is_weekend_stay'])
                       * (1 + 0.40 * (df['occupancy_forecast'] - 0.5))
                      ).round(0)

    # ---- Offered price: introduce realistic price variation ----
    # Some sessions get rack, others get promo discounts; loyalty members get member rates;
    # late shoppers sometimes see surge pricing. Variation is needed for elasticity ID.
    discount_base = rng.beta(2, 8, size=n)  # most discounts small, some large
    member_extra = df['loyalty_tier'].map({'none':0,'silver':.02,'gold':.05,'platinum':.08})
    promo_eligible = rng.binomial(1, 0.50, size=n)
    surge = ((df['lead_time_days'] <= 3) & (df['occupancy_forecast'] > 0.85))

    discount_pct = (discount_base * promo_eligible + member_extra).clip(0, 0.40)
    surge_factor = np.where(surge, rng.uniform(1.00, 1.15, size=n), 1.0)

    # Standard pricing: rack × (1 − discount_pct) × surge_factor
    standard_price = df['rack_rate'] * (1 - discount_pct) * surge_factor

    # Price A/B test arm: a fraction of sessions get a uniformly random price
    # multiplier in [0.60, 1.10]. This is independent of features by design,
    # which is what makes elasticity *identifiable* for DML.
    df['ab_test_arm'] = rng.binomial(1, ab_test_share, size=n)
    ab_mult = rng.uniform(0.60, 1.10, size=n)
    ab_price = df['rack_rate'] * ab_mult

    df['offered_price'] = np.where(df['ab_test_arm'] == 1, ab_price, standard_price).round(0)
    df['discount_pct'] = 1 - df['offered_price'] / df['rack_rate']
    df['log_price_ratio'] = np.log(df['offered_price'] / df['rack_rate'])

    # competitor price (slight variation around our rack)
    df['competitor_median_price'] = (df['rack_rate']
                                     * rng.normal(0.97, 0.08, size=n)).round(0)
    df['price_index'] = df['offered_price'] / df['competitor_median_price']

    # ---- True latent booking model ----
    # Baseline log-odds from features (this is what Stage 1 learns)
    alpha = (
        -1.50  # base intercept
        + 0.60 * (df['loyalty_tier'] == 'silver')
        + 1.10 * (df['loyalty_tier'] == 'gold')
        + 1.60 * (df['loyalty_tier'] == 'platinum')
        + 0.10 * df['prior_bookings'].clip(0, 10)
        + 0.50 * (df['channel'] == 'direct_app')
        + 0.30 * (df['channel'] == 'direct_web')
        - 0.20 * (df['channel'] == 'meta')
        + 0.40 * (df['trip_purpose'] == 'business')
        - 0.40 * (df['lead_time_days'] > 60)
        + 0.30 * (df['lead_time_days'] <= 3)
        - 0.80 * (df['price_index'] - 1.0)  # priced above comp → fewer bookings
        + 0.25 * (df['country'] == 'IN')
        + 0.30 * (df['device'] == 'ios')
    )

    # True elasticity varies by segment (this is what Stage 2 learns)
    eta = (
        -2.00  # global elasticity
        + 1.20 * (df['trip_purpose'] == 'business')   # business less elastic
        + 0.40 * (df['loyalty_tier'] == 'gold')
        + 0.70 * (df['loyalty_tier'] == 'platinum')
        - 0.50 * (df['lead_time_days'] >= 60)         # far-out shoppers more elastic
        + 0.60 * (df['lead_time_days'] <= 3)          # last-minute less elastic
        + 0.80 * (df['occupancy_forecast'] > 0.85)    # high occupancy → inelastic
        - 0.30 * (df['channel'] == 'meta')            # meta shoppers more elastic
    )
    df['true_eta'] = eta  # keep for evaluation

    # combined log-odds at offered price
    logit_p = alpha + eta * df['log_price_ratio']
    p_book = 1 / (1 + np.exp(-logit_p))
    df['true_p_book'] = p_book
    df['booked'] = rng.binomial(1, p_book)

    # variable cost per night (housekeeping, amenities) ~ 25% of standard rack
    df['variable_cost'] = (0.25 * df['rack_rate']).round(0)

    # timestamp for realism
    df['timestamp'] = pd.to_datetime('2026-01-01') + pd.to_timedelta(
        rng.integers(0, 90, size=n), unit='D')

    # Arrival date = booking timestamp + lead time. The booking consumes
    # inventory across `length_of_stay` consecutive nights starting here.
    # This is what enables the MILP's capacity constraints.
    df['arrival_date'] = (df['timestamp']
                          + pd.to_timedelta(df['lead_time_days'], unit='D')).dt.normalize()

    return df


def make_segment(df):
    """Coarse segment used in elasticity reporting."""
    return (df['trip_purpose'].astype(str) + '_' + df['loyalty_tier'].astype(str))


if __name__ == '__main__':
    df = generate_sessions(n=5000)
    print(df.head())
    print(f"\nBooking rate: {df['booked'].mean():.3f}")
    print(f"Mean discount: {df['discount_pct'].mean():.3f}")
    print(f"Mean true elasticity: {df['true_eta'].mean():.3f}")
