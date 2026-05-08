# Hotel Pricing — Three-Stage Demo (Approach 2 + DML + MILP)

A self-contained Streamlit demo of personalized + capacity-aware pricing for
hotel bookings. Three layers, each addressing a real production gap:

> **Stage 1 + 2 (Approach 2):** P(book | x, p) = P_ref(x) × (p / rack_rate)^η(x)
> **Stage 3 (MILP):** jointly optimize prices for all active sessions subject to per-(date, room_type) capacity

| Stage | Solves | Output |
|---|---|---|
| 1. **Propensity** | What's the baseline acceptance for this shopper at rack rate? | P_ref(x) — calibrated probability |
| 2. **Elasticity** | How does acceptance shift with log price ratio? | η(x) — heterogeneous elasticity |
| 3. **MILP** | Given P_ref and η for many shoppers, which prices maximize revenue *under capacity*? | Optimal price multiplier per session |

Stage 2 has two implementations:
- **GBM (monotone):** LightGBM with monotonicity constraint enforcing η ≤ 0. Fast, captures interactions, but biased when price is endogenous.
- **DML (causal):** Cross-fit nuisance + segment-heterogeneous OLS final stage with HC1 robust SE. Provides confidence intervals; can recover positive elasticity for Veblen / status segments.

## Files

| File | Purpose |
|---|---|
| `data_gen.py` | Synthetic hotel session generator. Includes a 30% A/B test arm (random price multipliers) — the source of identification for DML. Also computes `arrival_date` for capacity accounting. |
| `models.py` | `PropensityModel` (Stage 1) and `ElasticityModel` (Stage 2 GBM), plus `optimal_price` and `population_policy_eval`. |
| `dml_models.py` | `DMLElasticityModel` (Stage 2 DML), `compare_elasticity_models`, `offpolicy_revenue_eval`. |
| `milp_optimizer.py` | `policy_milp_joint`, `policy_greedy_per_session`, `policy_rack_rate`, and `compare_policies`. CBC solver via PuLP. Soft capacity with overflow walk-fee penalty. |
| `app.py` | Streamlit UI with 7 tabs covering all three stages. |
| `requirements.txt` | Pinned-ish dependencies. |

## Run

```bash
python -m venv pricing_env

.\pricing_env\Scripts\activate

pip install -r requirements.txt

streamlit run app.py
```

The app opens at `http://localhost:8501`.

## What's in the MILP tab (Stage 3)

When a peak weekend hits and inventory is scarce, per-session greedy pricing breaks down because it ignores the *displacement cost*: a discounted booking on Saturday night costs you a higher-margin booking you'd otherwise have taken. The MILP solves the joint problem.

**Demo scenario:** A 3-night peak weekend (Mar 13–15, 2026) with 285 sessions whose stays fall fully within the window. Capacity per night: 40 standard / 28 deluxe / 9 suites. Three policies compared on net revenue (gross minus walk-fee penalty for capacity violations):

| Policy | Net revenue | vs Rack | vs Greedy | Capacity violations (true η) |
|---|---:|---:|---:|---:|
| Rack rate | ₹754,263 | — | — | 13 room-nights oversold |
| Greedy per-session | ₹1,310,529 | +74% | — | 13 room-nights oversold |
| **MILP joint** | **₹1,547,929** | **+105%** | **+18%** | **0 — every constraint satisfied** |

The greedy policy's nominal (gross) revenue looks similar to MILP, but greedy oversells 13 room-nights, so once you subtract walk fees it falls 18% behind. **MILP is the only policy that respects every capacity constraint while maximizing revenue.**

## MILP formulation

Decision variables: `x[i, j] ∈ {0, 1}` — session `i` is offered price multiplier `mults[j]`, exactly one `j` per `i`.

Pre-computed from Stage 1 + Stage 2:
- `p_book[i, j] = P_ref(x_i) · mults[j]^η(x_i)`
- `rev[i, j] = p_book[i, j] · (mults[j] · rack_i − cost_i)`

Soft capacity with overflow slack `o_c ≥ 0` for cell `c = (date, room)`:
```
sum_{i ∈ S(c)} sum_j x[i, j] · p_book[i, j]  −  o_c  ≤  capacity[c]
```

Objective (revenue minus walk-fee penalty `λ` ≈ 2× mean rack rate):
```
maximize  sum_{i, j} x[i, j] · rev[i, j]  −  λ · sum_c o_c
```

Linear in `x` because `p_book` is precomputed from the upstream models. Solved by CBC via PuLP. Runtime is ~30–60s for 300–500 sessions on a laptop. For chance constraints / overbooking risk control, replace `sum p_book` with `μ_c + 1.65·σ_c`.

**Why soft constraints?** With biased η estimates and a finite price grid, hard constraints can render the LP infeasible (e.g., even at the maximum mult, model thinks too many will book). Soft constraints with a high penalty are always feasible AND better match the production reality where overbooking has a finite cost (walk fees).

## DML in 90 seconds

```
Y = θ(X) · T + g(X) + ε       (partial linear model)

Stage A — Cross-fit nuisance (K=5 folds, regularized LightGBM):
   g̃(X) = E[Y | X]   classifier with max_depth=4, min_child_samples=500
   m̃(X) = E[T | X]   regressor   with max_depth=4, min_child_samples=500

Stage B — Residualize:
   ã = Y − g̃(X)
   p̃ = T − m̃(X)

Stage C — Heterogeneous final stage (OLS, HC1 robust SE):
   ã_i = sum_j  β_j · (p̃_i × Z_j(X_i)) + u_i
   where Z(X) is one-hot of trip_purpose × loyalty_tier (8 segments).

Stage D — Convert probability-scale θ to constant-elasticity scale:
   η̂_segment = θ̂_segment / P̄_segment
```

The Stage D conversion is a first-order Taylor expansion at the segment's mean booking rate. Exact at that point; for very elastic segments where the optimal price is far from rack, the linear approximation introduces residual bias.

**Why aggressive nuisance regularization?** DML theory only requires nuisance error to converge faster than n^(-1/4) — overfitting nuisance models doesn't help and actively hurts identification by absorbing too much treatment variance. Shallow trees (depth 4) and a high `min_child_samples` (500) keep nuisances honest.

## Tabs

1. ** Data Overview** — sessions, discount distribution, booking rate by discount bucket.
2. ** Stage 1: Propensity** — AUC, calibration plot, feature importance, distribution of P_ref(x).
3. ** Stage 2: GBM Elasticity** — naive monotone-GBM elasticity baseline.
4. ** Stage 2: DML Variant** — diagnostics, per-segment table with CIs, off-policy revenue evaluation showing the overconfidence gap.
5. ** Single-Shopper Pricing** — interactive sliders. Sidebar selector chooses GBM or DML.
6. ** Population Policy** — apply the chosen Stage 2 policy to the validation set.
7. ** MILP Joint Optimization** — peak weekend scenario; rack vs greedy vs MILP. Capacity heatmaps, per-segment multiplier breakdown, downloadable CSV.

## Three production-grade lessons in this demo

1. **Naive ML is overconfident.** GBM monotone elasticity thinks it gets +20% RevPAR; under true η it actually delivers +7%. DML's headline of +12% is the one you can deploy.
2. **Per-session greedy pricing breaks under capacity.** It oversells nights, generates walk fees, and underprices the marginal premium customer. MILP fixes this with a one-shot optimization across all active sessions.
3. **Better elasticity → better MILP outcomes.** With DML's cleaner η estimates, the MILP solver respects capacity and generates 18% more net revenue than greedy. With GBM's biased η, the MILP can struggle (or get infeasibility on hard constraints).

## Mapping to a real Azure / Databricks deployment

- Sessions log → Delta table (bronze), feature-joined offers (silver), pricing decisions + capacity utilization (gold).
- `PropensityModel`, `ElasticityModel`, `DMLElasticityModel` → MLflow runs, served via Databricks Model Serving.
- Feature Store for user / hotel / occupancy features.
- **Permanent A/B test arm** (5–30% of traffic with randomized prices) to feed the elasticity model. Without it, DML and any other causal estimator loses identification within months of deployment.
- MILP runs on a per-property, per-day batch schedule (or near-real-time for high-velocity inventory) — Spark + PuLP, or Spark + Gurobi for larger instances.
- Drift monitors on Stage 1 calibration, η̂ by segment, RevPAR per session, *and* capacity violation rate.

## Caveats

- **DML still has bias** on this dataset (CI coverage 5/8). Two reasons: (1) `m̃(X) corr with T` ≈ 0.91 — the structural pricing logic is too predictable even with the A/B arm, and (2) the constant-elasticity conversion is only first-order accurate. Push for higher A/B share and consider a logit-link final stage in production.
- **MILP runtime** is fine for ~500 sessions. Above 5,000 you'll want OR-Tools CP-SAT or a commercial solver (Gurobi, CPLEX), or LP relaxation + rounding.
- **The combined formula `p_book = p_ref · m^η` is constant-elasticity**, a simplification of the logit-linear DGP. They agree to first order at the segment mean; for large price moves, model the demand curve directly.
- **Booking outcome here is per-session conversion.** Real systems also need cancellations, no-shows, and length-of-stay extension models.

## Try this

1. **Set `ab_test_share=0` in `data_gen.py`** → DML loses identification (`m̃ corr with T` jumps to ~0.99, `p̃ std` collapses).
2. **In the MILP tab, drop capacity to 20/15/4** → MILP starts pricing nearly everyone out at the max multiplier; net revenue still dominates greedy because it controls overbooking.
3. **In the MILP tab, switch the elasticity model to GBM** → notice the segment-level decisions become noisier (GBM's biased eta shifts who gets premium pricing) and net revenue drops 2–4%.
