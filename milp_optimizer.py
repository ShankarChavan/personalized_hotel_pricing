"""
Stage 3 — MILP Joint Pricing + Capacity Allocation
==================================================

Per-session greedy pricing (Stage 2 output) ignores capacity constraints.
On peak nights this leads to:
  • overselling — expected demand at chosen prices > capacity
  • or underselling — being too conservative on premium pricing for inelastic
    customers when capacity is actually tight.

The MILP formulation jointly optimizes price assignment for ALL active
sessions in a planning window, subject to per-(date, room_type) capacity:

Decision variables
------------------
  x[i, j] ∈ {0, 1}   = 1 if session i is offered price multiplier mults[j]
  Exactly one j per i:  Σ_j x[i, j] = 1

Pre-computed
------------
  p_book[i, j]  = P_ref(x_i) · mults[j]^η(x_i)   (from Stage 1 + Stage 2)
  rev[i, j]     = p_book[i, j] · (mults[j] · rack_i − cost_i)
  uses[i]       = list of (date, room_type) inventory cells consumed
                  if session i books (depends on arrival_date,
                  length_of_stay, room_type)

Objective
---------
  max  Σ_{i, j} x[i, j] · rev[i, j]

Capacity (expected-demand form)
-------------------------------
  For each inventory cell c = (date, room_type):
    Σ_{i: c ∈ uses[i]}  Σ_j  x[i, j] · p_book[i, j]   ≤   capacity[c]

Notes
-----
• Expected-demand constraints are linear in x (since p_book is a constant
  computed from the model). For tighter risk control add chance constraints
  using Normal approximation: μ + 1.65σ ≤ capacity.
• Solver: CBC via PuLP. Runtime is fine for ~1000 sessions × ~12 mults on
  a laptop. For larger problems use OR-Tools CP-SAT or Gurobi.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pulp


# ---------------------------------------------------------------------------
# Inventory accounting
# ---------------------------------------------------------------------------

def build_inventory_uses(df, capacity_keys):
    """For each session, list which (date, room_type) cells it consumes.

    Parameters
    ----------
    df : DataFrame with columns arrival_date, room_type, length_of_stay
    capacity_keys : list of (pd.Timestamp, str) tuples we have capacity for.
        Sessions whose stay falls outside this set don't consume tracked inventory.

    Returns
    -------
    uses : list of list of int (indices into capacity_keys)
    key_to_idx : dict mapping (date, room_type) → idx into capacity_keys
    """
    key_to_idx = {k: i for i, k in enumerate(capacity_keys)}
    uses = []
    for _, row in df.iterrows():
        nights = []
        a = pd.Timestamp(row['arrival_date']).normalize()
        r = row['room_type']
        k = int(row['length_of_stay'])
        for d in range(k):
            key = (a + pd.Timedelta(days=d), r)
            if key in key_to_idx:
                nights.append(key_to_idx[key])
        uses.append(nights)
    return uses, key_to_idx


def _compute_p_book_and_rev(df, prop_model, eta_model, price_mults):
    """Precompute the (n × J) tables of acceptance probability and revenue."""
    p_ref = prop_model.predict_p_ref(df)
    eta = eta_model.estimate_eta(df)

    rack = df['rack_rate'].values
    cost = df['variable_cost'].values

    n = len(df)
    J = len(price_mults)
    p_book = np.zeros((n, J))
    rev = np.zeros((n, J))
    for j, m in enumerate(price_mults):
        p_book[:, j] = np.clip(p_ref * (m ** eta), 0.0, 1.0)
        rev[:, j] = p_book[:, j] * (m * rack - cost)
    return p_ref, eta, p_book, rev


def _capacity_use_from_decisions(df, chosen_p_book, capacity_keys):
    """Total expected room-nights consumed per inventory cell."""
    use = {k: 0.0 for k in capacity_keys}
    for i, (_, row) in enumerate(df.iterrows()):
        a = pd.Timestamp(row['arrival_date']).normalize()
        r = row['room_type']
        k = int(row['length_of_stay'])
        pb = float(chosen_p_book[i])
        for d in range(k):
            key = (a + pd.Timedelta(days=d), r)
            if key in use:
                use[key] += pb
    return use


# ---------------------------------------------------------------------------
# Policy 1: Rack rate (no optimization)
# ---------------------------------------------------------------------------

def policy_rack_rate(df, prop_model, eta_model):
    """Charge rack rate (mult = 1.0) to every session — the do-nothing baseline."""
    p_ref = prop_model.predict_p_ref(df)
    eta = eta_model.estimate_eta(df)
    p_book = np.clip(p_ref * (1.0 ** eta), 0, 1)  # = p_ref
    rev = p_book * (df['rack_rate'].values - df['variable_cost'].values)

    out = df.reset_index(drop=True).copy()
    out['p_ref'] = p_ref
    out['eta_hat'] = eta
    out['chosen_mult'] = 1.0
    out['chosen_price'] = out['rack_rate']
    out['chosen_p_book'] = p_book
    out['chosen_expected_rev'] = rev
    return out


# ---------------------------------------------------------------------------
# Policy 2: Greedy per-session (Stage 2 output, ignores capacity)
# ---------------------------------------------------------------------------

def policy_greedy_per_session(df, prop_model, eta_model, price_mults):
    """Per-session optimal price ignoring capacity — what Stage 2 produces today."""
    df = df.reset_index(drop=True).copy()
    p_ref, eta, p_book, rev = _compute_p_book_and_rev(df, prop_model, eta_model, price_mults)

    best_j = rev.argmax(axis=1)
    best_mult = np.array([price_mults[j] for j in best_j])
    best_p_book = np.array([p_book[i, best_j[i]] for i in range(len(df))])
    best_rev = np.array([rev[i, best_j[i]] for i in range(len(df))])

    df['p_ref'] = p_ref
    df['eta_hat'] = eta
    df['chosen_mult'] = best_mult
    df['chosen_price'] = best_mult * df['rack_rate']
    df['chosen_p_book'] = best_p_book
    df['chosen_expected_rev'] = best_rev
    return df


# ---------------------------------------------------------------------------
# Policy 3: MILP joint optimization
# ---------------------------------------------------------------------------

def policy_milp_joint(df, prop_model, eta_model, price_mults, capacity,
                      overflow_penalty=None,
                      time_limit_sec=30, verbose=False):
    """Solve the joint pricing + capacity allocation MILP.

    Parameters
    ----------
    df : DataFrame of sessions (must have arrival_date, length_of_stay,
         room_type, rack_rate, variable_cost, plus the model features).
    prop_model, eta_model : Stage 1 + Stage 2.
    price_mults : iterable of price multipliers to consider.
    capacity : dict {(pd.Timestamp, room_type_str) → int} of room-nights available.
        Sessions whose stay falls outside this set are unconstrained.
    overflow_penalty : float per overflow room-night; default = 2.0 × mean rack rate
        (≈ industry walk-fee for overbooking). Soft capacity is always feasible
        and matches the standard revenue-management formulation. Pass None to
        use hard constraints (may report 'Infeasible' if the price grid can't
        ration demand).
    time_limit_sec : CBC time limit. The problem is usually solved well within.
    verbose : print solver progress.

    Returns
    -------
    Dict with:
        sessions : DataFrame of sessions with chosen_mult, chosen_p_book,
                   chosen_expected_rev, chosen_price
        objective : MILP objective value (revenue minus overflow penalty)
        gross_revenue : sum of expected revenue (before overflow penalty)
        overflow_total : sum of overflow room-nights across cells
        status : LP status string ('Optimal', 'Time Limited', etc.)
        capacity_use : dict mapping inventory cell → expected occupancy
        capacity : passed-through capacity dict (for analysis)
        n_priced_out : sessions assigned the maximum price multiplier
                       (effective rejection)
    """
    price_mults = np.asarray(price_mults)
    df = df.reset_index(drop=True).copy()
    n = len(df)
    J = len(price_mults)

    # 1. Pre-compute economics
    p_ref, eta, p_book, rev = _compute_p_book_and_rev(df, prop_model, eta_model, price_mults)

    # 2. Inventory usage map
    capacity_keys = list(capacity.keys())
    uses, _ = build_inventory_uses(df, capacity_keys)

    # 3. Default overflow penalty: 2× mean rack rate (industry walk fee)
    if overflow_penalty is None:
        overflow_penalty = 2.0 * float(df['rack_rate'].mean())
    use_soft = overflow_penalty is not None and overflow_penalty < float('inf')

    # 4. Build MILP
    model = pulp.LpProblem('hotel_pricing_milp', pulp.LpMaximize)

    # Binary decision variables
    x = [[pulp.LpVariable(f'x_{i}_{j}', cat='Binary') for j in range(J)]
         for i in range(n)]

    # Overflow slacks (continuous, ≥ 0) — one per inventory cell
    if use_soft:
        ov = [pulp.LpVariable(f'ov_{k}', lowBound=0, cat='Continuous')
              for k in range(len(capacity_keys))]
    else:
        ov = [None] * len(capacity_keys)

    # Objective
    revenue_term = pulp.lpSum(x[i][j] * float(rev[i, j])
                               for i in range(n) for j in range(J))
    if use_soft:
        penalty_term = pulp.lpSum(overflow_penalty * ov[k]
                                   for k in range(len(capacity_keys)))
        model += revenue_term - penalty_term
    else:
        model += revenue_term

    # One price per session
    for i in range(n):
        model += pulp.lpSum(x[i][j] for j in range(J)) == 1, f'one_price_{i}'

    # Capacity constraints (soft if overflow_penalty set, else hard)
    for k_idx, key in enumerate(capacity_keys):
        cap = capacity[key]
        members = [i for i in range(n) if k_idx in uses[i]]
        if not members:
            continue
        demand_expr = pulp.lpSum(x[i][j] * float(p_book[i, j])
                                  for i in members for j in range(J))
        if use_soft:
            # demand <= capacity + overflow
            model += demand_expr - ov[k_idx] <= cap, f'cap_{k_idx}'
        else:
            model += demand_expr <= cap, f'cap_{k_idx}'

    # Solve
    solver = pulp.PULP_CBC_CMD(msg=verbose, timeLimit=time_limit_sec)
    model.solve(solver)
    status = pulp.LpStatus[model.status]
    objective = pulp.value(model.objective)

    # Extract solution
    chosen_j = np.zeros(n, dtype=int)
    for i in range(n):
        vals = [pulp.value(x[i][j]) or 0.0 for j in range(J)]
        chosen_j[i] = int(np.argmax(vals))

    chosen_mult = price_mults[chosen_j]
    chosen_p_book = np.array([p_book[i, chosen_j[i]] for i in range(n)])
    chosen_rev = np.array([rev[i, chosen_j[i]] for i in range(n)])

    df['p_ref'] = p_ref
    df['eta_hat'] = eta
    df['chosen_mult'] = chosen_mult
    df['chosen_price'] = chosen_mult * df['rack_rate']
    df['chosen_p_book'] = chosen_p_book
    df['chosen_expected_rev'] = chosen_rev

    # Capacity utilization (under model eta)
    cap_use = _capacity_use_from_decisions(df, chosen_p_book, capacity_keys)

    # Overflow per cell
    if use_soft:
        overflow_per_cell = {capacity_keys[k]: float(pulp.value(ov[k]) or 0.0)
                             for k in range(len(capacity_keys))}
    else:
        overflow_per_cell = {k: max(0.0, cap_use[k] - capacity[k])
                             for k in capacity_keys}
    overflow_total = sum(overflow_per_cell.values())

    gross_revenue = float(np.sum(chosen_rev))

    priced_out_threshold = price_mults[-1] - 1e-6
    n_priced_out = int((chosen_mult >= priced_out_threshold).sum())

    return {
        'sessions': df,
        'objective': float(objective) if objective is not None else 0.0,
        'gross_revenue': gross_revenue,
        'overflow_total': overflow_total,
        'overflow_per_cell': overflow_per_cell,
        'overflow_penalty': overflow_penalty if use_soft else None,
        'status': status,
        'capacity_use': cap_use,
        'capacity': capacity,
        'n_priced_out': n_priced_out,
        'price_mults': price_mults,
    }


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------

def compare_policies(df, prop_model, eta_model, price_mults, capacity,
                     true_eta_col='true_eta', milp_time_limit=30,
                     overflow_penalty=None):
    """Run all three policies and produce a comparable summary.

    Returns a DataFrame keyed by policy with self-estimated and actual
    (true-η counterfactual) revenue, plus capacity violation accounting
    measured under TRUE elasticity. The MILP can be run with soft or hard
    capacity (default soft, with penalty=2× mean rack rate).
    """
    out_rack = policy_rack_rate(df, prop_model, eta_model)
    out_grdy = policy_greedy_per_session(df, prop_model, eta_model, price_mults)
    out_milp = policy_milp_joint(df, prop_model, eta_model, price_mults, capacity,
                                  overflow_penalty=overflow_penalty,
                                  time_limit_sec=milp_time_limit)

    capacity_keys = list(capacity.keys())

    rows = []
    for name, sess in [
        ('Rack rate',           out_rack),
        ('Greedy per-session',  out_grdy),
        ('MILP joint',          out_milp['sessions']),
    ]:
        # Self-estimated revenue (uses model's eta to compute p_book)
        self_rev = sess['chosen_expected_rev'].sum()

        # Actual under true elasticity
        true_eta = sess[true_eta_col].values
        rack = sess['rack_rate'].values
        cost = sess['variable_cost'].values
        mult = sess['chosen_mult'].values
        p_ref = sess['p_ref'].values
        actual_p_book = np.clip(p_ref * (mult ** true_eta), 0, 1)
        actual_rev = (actual_p_book * (mult * rack - cost)).sum()

        # Capacity utilization under TRUE elasticity (the honest accounting)
        cap_use_true = _capacity_use_from_decisions(sess, actual_p_book, capacity_keys)
        n_violations = sum(1 for k in capacity_keys
                           if cap_use_true[k] > capacity[k] + 1e-6)
        max_overuse = max(
            (cap_use_true[k] - capacity[k] for k in capacity_keys),
            default=0.0,
        )
        total_overuse = sum(max(0.0, cap_use_true[k] - capacity[k])
                            for k in capacity_keys)

        # Walk-fee penalty: overflow × walk_fee_per_room_night
        walk_fee = 2.0 * float(df['rack_rate'].mean())
        net_actual_rev = actual_rev - walk_fee * total_overuse

        rows.append({
            'policy': name,
            'self_estimated_rev': self_rev,
            'actual_rev_true_eta': actual_rev,
            'walk_fee_penalty': walk_fee * total_overuse,
            'net_actual_rev': net_actual_rev,
            'mean_mult': mult.mean(),
            'expected_bookings_actual': actual_p_book.sum(),
            'cells_oversold_true_eta': n_violations,
            'total_overuse_true_eta': total_overuse,
        })

    summary = pd.DataFrame(rows)
    return summary, {
        'rack': out_rack,
        'greedy': out_grdy,
        'milp': out_milp,
    }
