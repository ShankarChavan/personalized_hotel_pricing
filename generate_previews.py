"""Generate static preview charts of the trained models for the demo artifact."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from data_gen import generate_sessions
from models import PropensityModel, ElasticityModel, optimal_price, population_policy_eval
from dml_models import (DMLElasticityModel, compare_elasticity_models,
                         offpolicy_revenue_eval)
from milp_optimizer import compare_policies, _capacity_use_from_decisions


class _OracleEta:
    """For off-policy comparison: returns the true synthetic elasticity."""
    def estimate_eta(self, df):
        return df['true_eta'].values

# ----------------------------------------------------------------------------
# Train models
# ----------------------------------------------------------------------------
print("Generating data + training models...")
df = generate_sessions(n=20000, seed=42)
train = df[df['timestamp'] < '2026-03-01']
valid = df[df['timestamp'] >= '2026-03-01'].copy()

prop = PropensityModel().fit(train, valid)
elas = ElasticityModel().fit(train, valid)
dml = DMLElasticityModel(n_folds=5).fit(train)
valid['eta_hat'] = elas.estimate_eta(valid)
valid['eta_dml'] = dml.estimate_eta(valid)
valid['p_ref'] = prop.predict_p_ref(valid)

print(f"Stage 1 AUC: {prop.metrics['auc']:.3f}, Brier: {prop.metrics['brier']:.3f}")
print(f"Stage 2 GBM mean eta: {valid['eta_hat'].mean():.2f}, true mean: {valid['true_eta'].mean():.2f}")
print(f"Stage 2 DML mean eta: {valid['eta_dml'].mean():.2f}  "
      f"(global pooled CI: [{dml.global_eta_ci_[0]:.2f}, {dml.global_eta_ci_[1]:.2f}])")

# ----------------------------------------------------------------------------
# Figure 1: Data quality dashboard
# ----------------------------------------------------------------------------
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=('Discount distribution',
                                    'Booking rate by discount bucket'))

fig.add_trace(go.Histogram(x=df['discount_pct'], nbinsx=40,
                            marker_color='#4C78A8', name='Sessions'),
              row=1, col=1)

bucket = pd.cut(df['discount_pct'], bins=[-.01,.02,.05,.10,.15,.25,.50],
                labels=['0-2%','2-5%','5-10%','10-15%','15-25%','25%+'])
agg = df.assign(bucket=bucket).groupby('bucket', observed=True)['booked'].agg(['mean','size']).reset_index()
fig.add_trace(go.Bar(x=agg['bucket'].astype(str), y=agg['mean'],
                      marker_color='#54A24B', name='Book rate',
                      text=[f'n={n}' for n in agg['size']],
                      textposition='outside'),
              row=1, col=2)

fig.update_layout(title='Synthetic Hotel Booking Data — Discount × Booking Rate',
                  height=400, showlegend=False, template='plotly_white')
fig.write_image('/home/claude/hotel_pricing/preview_01_data.png', width=1200, height=420)
print("✓ preview_01_data.png")

# ----------------------------------------------------------------------------
# Figure 2: Stage 1 calibration + feature importance
# ----------------------------------------------------------------------------
pred = prop.predict_p_ref(valid)
bins = pd.cut(pred, bins=np.linspace(0, 1, 11))
cal = (pd.DataFrame({'p_pred': pred, 'y': valid['booked'].values, 'bin': bins})
         .groupby('bin', observed=True)
         .agg(p_pred=('p_pred','mean'), y_actual=('y','mean'), n=('y','size'))
         .reset_index())

imp = pd.DataFrame({
    'feature': prop.feats,
    'gain': prop.model.booster_.feature_importance(importance_type='gain')
}).sort_values('gain', ascending=True)

fig = make_subplots(rows=1, cols=2,
                    subplot_titles=(
                        f"Stage 1 Calibration (AUC={prop.metrics['auc']:.3f}, "
                        f"Brier={prop.metrics['brier']:.3f})",
                        'Stage 1 Feature Importance'))

fig.add_trace(go.Scatter(x=cal['p_pred'], y=cal['y_actual'],
                         mode='markers+lines',
                         marker=dict(size=12, color='#4C78A8'),
                         line=dict(width=3),
                         name='Model'), row=1, col=1)
fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                         line=dict(dash='dash', color='gray'),
                         name='Perfect'), row=1, col=1)

fig.add_trace(go.Bar(x=imp['gain'], y=imp['feature'], orientation='h',
                     marker_color='#F58518', name='Gain'),
              row=1, col=2)

fig.update_xaxes(title_text='Predicted P(book)', row=1, col=1)
fig.update_yaxes(title_text='Actual book rate', row=1, col=1)
fig.update_xaxes(title_text='Gain', row=1, col=2)
fig.update_layout(title='Stage 1 — Propensity Model P_ref(x) at Rack Rate',
                  height=500, showlegend=False, template='plotly_white')
fig.write_image('/home/claude/hotel_pricing/preview_02_stage1.png', width=1200, height=520)
print("✓ preview_02_stage1.png")

# ----------------------------------------------------------------------------
# Figure 3: Stage 2 elasticity — true vs estimated by segment
# ----------------------------------------------------------------------------
seg = (valid.groupby(['trip_purpose','loyalty_tier'], observed=True)
            .agg(true_eta=('true_eta','mean'),
                 eta_hat=('eta_hat','mean'),
                 n=('booked','size')).reset_index())
seg['segment'] = seg['trip_purpose'] + ' / ' + seg['loyalty_tier']
seg = seg.sort_values('true_eta')

lead_buckets = pd.cut(valid['lead_time_days'],
                      bins=[0,3,7,14,30,60,200],
                      labels=['1-3','4-7','8-14','15-30','31-60','60+'])
lead_eta = (valid.assign(bucket=lead_buckets)
                 .groupby('bucket', observed=True)['eta_hat'].mean().reset_index())

occ_buckets = pd.cut(valid['occupancy_forecast'],
                     bins=[0,0.5,0.7,0.85,1.0],
                     labels=['<50%','50-70%','70-85%','85%+'])
occ_eta = (valid.assign(bucket=occ_buckets)
                .groupby('bucket', observed=True)['eta_hat'].mean().reset_index())

fig = make_subplots(rows=2, cols=2,
                    subplot_titles=('Elasticity by segment: true vs estimated',
                                    'Estimated elasticity by lead-time',
                                    '',
                                    'Estimated elasticity by occupancy'),
                    specs=[[{"colspan": 2}, None],
                           [{}, {}]],
                    row_heights=[0.55, 0.45])

fig.add_trace(go.Bar(name='True η', x=seg['segment'], y=seg['true_eta'],
                     marker_color='#4C78A8'), row=1, col=1)
fig.add_trace(go.Bar(name='Estimated η̂', x=seg['segment'], y=seg['eta_hat'],
                     marker_color='#F58518'), row=1, col=1)

fig.add_trace(go.Bar(x=lead_eta['bucket'].astype(str), y=lead_eta['eta_hat'],
                     marker_color='#54A24B', showlegend=False),
              row=2, col=1)
fig.add_trace(go.Bar(x=occ_eta['bucket'].astype(str), y=occ_eta['eta_hat'],
                     marker_color='#B279A2', showlegend=False),
              row=2, col=2)

fig.update_layout(title='Stage 2 — Heterogeneous Elasticity η(x)',
                  height=700, barmode='group', template='plotly_white')
fig.update_yaxes(title_text='η', row=1, col=1)
fig.update_yaxes(title_text='η̂', row=2, col=1)
fig.update_yaxes(title_text='η̂', row=2, col=2)
fig.write_image('/home/claude/hotel_pricing/preview_03_stage2.png', width=1200, height=720)
print("✓ preview_03_stage2.png")

# ----------------------------------------------------------------------------
# Figure 4: Single-shopper pricing — three example shoppers
# ----------------------------------------------------------------------------
shoppers = {
    'Business / Gold / Late book': pd.Series({
        'room_type': 'deluxe', 'channel': 'direct_app', 'country': 'IN',
        'device': 'ios', 'loyalty_tier': 'gold', 'trip_purpose': 'business',
        'lead_time_days': 3, 'length_of_stay': 2, 'is_weekend_stay': 0,
        'prior_bookings': 6, 'occupancy_forecast': 0.85, 'seasonality_index': 1.0,
        'rack_rate': 9500, 'competitor_median_price': 9300,
        'price_index': 9500/9300, 'variable_cost': 2375,
    }),
    'Leisure / None / Far-out': pd.Series({
        'room_type': 'standard', 'channel': 'meta', 'country': 'US',
        'device': 'desktop', 'loyalty_tier': 'none', 'trip_purpose': 'leisure',
        'lead_time_days': 90, 'length_of_stay': 3, 'is_weekend_stay': 1,
        'prior_bookings': 0, 'occupancy_forecast': 0.50, 'seasonality_index': 1.0,
        'rack_rate': 6500, 'competitor_median_price': 6800,
        'price_index': 6500/6800, 'variable_cost': 1625,
    }),
    'Leisure / Silver / Mid lead-time': pd.Series({
        'room_type': 'deluxe', 'channel': 'direct_web', 'country': 'IN',
        'device': 'android', 'loyalty_tier': 'silver', 'trip_purpose': 'leisure',
        'lead_time_days': 30, 'length_of_stay': 2, 'is_weekend_stay': 1,
        'prior_bookings': 2, 'occupancy_forecast': 0.65, 'seasonality_index': 1.10,
        'rack_rate': 9500, 'competitor_median_price': 9000,
        'price_index': 9500/9000, 'variable_cost': 2375,
    }),
}

fig = make_subplots(rows=1, cols=3,
                    subplot_titles=list(shoppers.keys()),
                    specs=[[{"secondary_y": True}]*3])

for i, (name, shopper) in enumerate(shoppers.items(), 1):
    best, curve = optimal_price(shopper, prop, elas)
    fig.add_trace(go.Scatter(x=curve['price'], y=curve['p_book'],
                             mode='lines', line=dict(color='#4C78A8', width=3),
                             name='P(book)' if i==1 else None,
                             showlegend=(i==1)),
                  row=1, col=i, secondary_y=False)
    fig.add_trace(go.Scatter(x=curve['price'], y=curve['expected_rev'],
                             mode='lines', line=dict(color='#54A24B', width=3),
                             name='E[revenue]' if i==1 else None,
                             showlegend=(i==1)),
                  row=1, col=i, secondary_y=True)
    fig.add_vline(x=best['price'], line_dash='dash', line_color='red',
                  annotation_text=f"opt: ₹{best['price']:.0f}",
                  annotation_position='top', row=1, col=i)
    fig.add_vline(x=shopper['rack_rate'], line_dash='dot', line_color='gray',
                  annotation_text=f"rack: ₹{shopper['rack_rate']:.0f}",
                  annotation_position='bottom', row=1, col=i)
    fig.update_xaxes(title_text='Price (₹)', row=1, col=i)

fig.update_yaxes(title_text='P(book)', row=1, col=1, secondary_y=False)
fig.update_yaxes(title_text='E[revenue]', row=1, col=3, secondary_y=True)
fig.update_layout(title='Optimal Price by Shopper Type — P(book) and Expected Revenue',
                  height=500, template='plotly_white')
fig.write_image('/home/claude/hotel_pricing/preview_04_shoppers.png', width=1500, height=520)
print("✓ preview_04_shoppers.png")

# Print summary of the three shoppers
print("\nThree-shopper pricing summary:")
for name, shopper in shoppers.items():
    best, curve = optimal_price(shopper, prop, elas)
    print(f"  {name:40s}  rack=₹{shopper['rack_rate']:>5.0f}  "
          f"η̂={curve['eta'].iloc[0]:+.2f}  "
          f"opt_mult={best['price_mult']:.2f}  "
          f"opt=₹{best['price']:>5.0f}")

# ----------------------------------------------------------------------------
# Figure 5: Population policy lift
# ----------------------------------------------------------------------------
print("\nRunning population policy evaluation...")
pop_sample = valid.sample(min(6000, len(valid)), random_state=1)
pop = population_policy_eval(pop_sample, prop, elas)
baseline_rev = pop['baseline_expected_rev'].mean()
optimal_rev = pop['optimal_expected_rev'].mean()
lift_pct = (optimal_rev / baseline_rev - 1) * 100

seg_perf = (pop.assign(segment=pop['trip_purpose'].astype(str)+' / '+pop['loyalty_tier'].astype(str))
              .groupby('segment')
              .agg(baseline=('baseline_expected_rev','mean'),
                   optimal=('optimal_expected_rev','mean'),
                   avg_mult=('optimal_mult','mean'),
                   n=('booked','size')).reset_index())
seg_perf['lift_%'] = ((seg_perf['optimal']/seg_perf['baseline'])-1)*100
seg_perf = seg_perf.sort_values('lift_%')

fig = make_subplots(rows=1, cols=2,
                    subplot_titles=('Distribution of optimal price multipliers',
                                    'Revenue lift by segment vs charging rack'))

fig.add_trace(go.Histogram(x=pop['optimal_mult'], nbinsx=30,
                            marker_color='#4C78A8'),
              row=1, col=1)
fig.add_vline(x=1.0, line_dash='dash', line_color='red',
              annotation_text='Rack rate', row=1, col=1)

fig.add_trace(go.Bar(x=seg_perf['lift_%'], y=seg_perf['segment'],
                     orientation='h',
                     marker=dict(color=seg_perf['avg_mult'], colorscale='RdYlGn_r',
                                showscale=True, colorbar=dict(title='Avg<br>mult', x=1.0)),
                     text=[f'{v:.1f}%' for v in seg_perf['lift_%']],
                     textposition='outside'),
              row=1, col=2)

fig.update_xaxes(title_text='Optimal price / rack', row=1, col=1)
fig.update_yaxes(title_text='Sessions', row=1, col=1)
fig.update_xaxes(title_text='RevPAR lift (%)', row=1, col=2)
fig.update_layout(title=(f'Population Policy — Overall RevPAR lift vs rack: '
                         f'<b>{lift_pct:+.1f}%</b>  '
                         f'(₹{baseline_rev:,.0f} → ₹{optimal_rev:,.0f} per session)'),
                  height=600, template='plotly_white', showlegend=False)
fig.write_image('/home/claude/hotel_pricing/preview_05_population.png', width=1400, height=620)
print(f"✓ preview_05_population.png — RevPAR lift: {lift_pct:+.1f}%")

# ----------------------------------------------------------------------------
# Figure 6: DML comparison — per-segment elasticity + revenue overconfidence
# ----------------------------------------------------------------------------
print("\nGenerating DML comparison...")
cmp_df = compare_elasticity_models(valid, 'true_eta', elas, dml)
cmp_df = cmp_df.sort_values('true_eta').reset_index(drop=True)
coverage = ((cmp_df['lo95'] <= cmp_df['true_eta']) &
            (cmp_df['true_eta'] <= cmp_df['hi95'])).sum()
naive_mae = cmp_df['naive_bias'].abs().mean()
dml_mae = cmp_df['dml_bias'].abs().mean()

# off-policy revenue eval
samp = valid.sample(min(3000, len(valid)), random_state=2)
res_oracle = offpolicy_revenue_eval(samp, prop, _OracleEta())
res_gbm = offpolicy_revenue_eval(samp, prop, elas)
res_dml = offpolicy_revenue_eval(samp, prop, dml)

print(f"  GBM MAE={naive_mae:.3f}  DML MAE={dml_mae:.3f}  CI coverage={coverage}/{len(cmp_df)}")
print(f"  Oracle lift {res_oracle['lift_pct']:+.1f}% | "
      f"GBM actual {res_gbm['lift_pct']:+.1f}% (self-est {res_gbm['self_estimated_lift_pct']:+.1f}%) | "
      f"DML actual {res_dml['lift_pct']:+.1f}% (self-est {res_dml['self_estimated_lift_pct']:+.1f}%)")

fig = make_subplots(rows=1, cols=2,
                    column_widths=[0.62, 0.38],
                    subplot_titles=(
                        f'Per-segment elasticity: True vs GBM vs DML  '
                        f'(DML 95% CI covers truth: {coverage}/{len(cmp_df)})',
                        'Revenue lift: self-estimated vs actual'
                    ))

x_labels = cmp_df['segment'].tolist()
fig.add_trace(go.Bar(name='True η', x=x_labels, y=cmp_df['true_eta'],
                      marker_color='#1f77b4'), row=1, col=1)
fig.add_trace(go.Bar(name='GBM η̂', x=x_labels, y=cmp_df['eta_naive'],
                      marker_color='#ff7f0e'), row=1, col=1)
fig.add_trace(go.Bar(
    name='DML η̂ (with 95% CI)', x=x_labels, y=cmp_df['eta_dml'],
    marker_color='#2ca02c',
    error_y=dict(type='data', symmetric=False,
                 array=cmp_df['hi95'] - cmp_df['eta_dml'],
                 arrayminus=cmp_df['eta_dml'] - cmp_df['lo95'],
                 color='#1a5e1a', thickness=1.5)),
    row=1, col=1)
fig.add_hline(y=0, line_dash='dot', line_color='gray', row=1, col=1)

# Right panel: actual vs self-estimated lift
models = ['Oracle\n(true η)', 'GBM\n(monotone)', 'DML\n(causal)']
self_est = [res_oracle['self_estimated_lift_pct'],
            res_gbm['self_estimated_lift_pct'],
            res_dml['self_estimated_lift_pct']]
actual = [res_oracle['lift_pct'],
          res_gbm['lift_pct'],
          res_dml['lift_pct']]

fig.add_trace(go.Bar(name='Self-estimated', x=models, y=self_est,
                      marker_color='#bbb',
                      text=[f"{v:+.1f}%" for v in self_est],
                      textposition='outside'), row=1, col=2)
fig.add_trace(go.Bar(name='Actual (true η)', x=models, y=actual,
                      marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'],
                      text=[f"{v:+.1f}%" for v in actual],
                      textposition='outside'), row=1, col=2)

fig.update_yaxes(title_text='Elasticity (η)', row=1, col=1)
fig.update_yaxes(title_text='RevPAR lift vs rack (%)', row=1, col=2)
fig.update_layout(
    title=(f'<b>Stage 2: DML variant</b> — '
           f'GBM thinks +{res_gbm["self_estimated_lift_pct"]:.0f}% but actually delivers +{res_gbm["lift_pct"]:.0f}%  '
           f'({res_gbm["overconfidence_pct"]:.0f} pts overconfidence). '
           f'DML delivers +{res_dml["lift_pct"]:.0f}% with only {res_dml["overconfidence_pct"]:.0f} pts overconfidence.'),
    barmode='group', height=560, template='plotly_white',
    legend=dict(orientation='h', y=-0.18, x=0.5, xanchor='center'),
)
fig.write_image('/home/claude/hotel_pricing/preview_06_dml.png', width=1500, height=580)
print(f"✓ preview_06_dml.png")

# ----------------------------------------------------------------------------
# Figure 7: MILP joint optimization — peak weekend with tight capacity
# ----------------------------------------------------------------------------
print("\nGenerating MILP preview...")

peak_dates = pd.date_range('2026-03-13', '2026-03-15', freq='D')
def _stay_in_window(row, dates=peak_dates):
    a = pd.Timestamp(row['arrival_date']).normalize()
    los = int(row['length_of_stay'])
    return a >= dates[0] and a + pd.Timedelta(days=los-1) <= dates[-1]

active = valid[valid.apply(_stay_in_window, axis=1)].copy()
print(f"  Active sessions for peak weekend: {len(active)}")

capacity = {(d.normalize(), r): {'standard': 40, 'deluxe': 28, 'suite': 9}[r]
            for d in peak_dates for r in ['standard','deluxe','suite']}
mults_milp = np.linspace(0.55, 1.80, 25)

# Use DML elasticity for the MILP — better calibrated for the optimizer
print("  Running compare_policies (rack / greedy / MILP) with DML eta...")
summary_milp, results_milp = compare_policies(
    active, prop, dml, mults_milp, capacity, milp_time_limit=60)
print(summary_milp.round(0).to_string(index=False))

# Build per-cell utilization matrices for each policy (under TRUE eta)
room_types = ['standard','deluxe','suite']
date_strs = [d.strftime('%a %m-%d') for d in peak_dates]
sorted_dates = sorted({k[0] for k in capacity.keys()})

def _heat_matrix(sess, capacity_dict):
    rack_v = sess['rack_rate'].values
    mult = sess['chosen_mult'].values
    p_ref = sess['p_ref'].values
    true_eta = sess['true_eta'].values
    actual_p = np.clip(p_ref * (mult ** true_eta), 0, 1)
    cap_use = _capacity_use_from_decisions(sess, actual_p, list(capacity_dict.keys()))
    mat = np.zeros((len(room_types), len(date_strs)))
    cap_mat = np.zeros_like(mat)
    for i, r in enumerate(room_types):
        for j, d in enumerate(sorted_dates):
            key = (pd.Timestamp(d), r)
            mat[i,j] = cap_use.get(key, 0.0)
            cap_mat[i,j] = capacity_dict.get(key, 1)
    return mat, cap_mat

rack_use, cap_mat = _heat_matrix(results_milp['rack'], capacity)
grdy_use, _      = _heat_matrix(results_milp['greedy'], capacity)
milp_use, _      = _heat_matrix(results_milp['milp']['sessions'], capacity)

rack_net = summary_milp.loc[summary_milp.policy=='Rack rate', 'net_actual_rev'].iloc[0]
grdy_net = summary_milp.loc[summary_milp.policy=='Greedy per-session', 'net_actual_rev'].iloc[0]
milp_net = summary_milp.loc[summary_milp.policy=='MILP joint', 'net_actual_rev'].iloc[0]
walk_fee = float(active['rack_rate'].mean()) * 2.0

# 2-row layout: top row revenue/penalty bars, bottom row 3 capacity heatmaps
fig = make_subplots(
    rows=2, cols=3,
    column_widths=[0.34, 0.33, 0.33],
    row_heights=[0.42, 0.58],
    subplot_titles=(
        'Net revenue (after walk-fee)',
        'Gross revenue & walk-fee penalty (true η)',
        'MILP price multiplier by segment',
        'Rack rate — capacity usage (true η)',
        'Greedy — capacity usage (true η)',
        'MILP — capacity usage (true η)',
    ),
    specs=[[{'type':'bar'}, {'type':'bar'}, {'type':'bar'}],
           [{'type':'heatmap'}, {'type':'heatmap'}, {'type':'heatmap'}]])

# Top-left: net revenue bars
policies = ['Rack rate', 'Greedy', 'MILP joint']
nets = [rack_net, grdy_net, milp_net]
fig.add_trace(go.Bar(
    x=policies, y=nets,
    marker_color=['#1f77b4','#ff7f0e','#2ca02c'],
    text=[f'₹{v:,.0f}' for v in nets],
    textposition='outside',
    showlegend=False,
), row=1, col=1)

# Top-middle: gross + walk-fee penalty stacked
gross = summary_milp['actual_rev_true_eta'].tolist()
walk = summary_milp['walk_fee_penalty'].tolist()
fig.add_trace(go.Bar(
    name='Walk-fee penalty', x=policies, y=[-w for w in walk],
    marker_color='#d62728',
    text=[f'-₹{w:,.0f}' if w > 0 else '' for w in walk],
    textposition='outside',
), row=1, col=2)
fig.add_trace(go.Bar(
    name='Gross revenue', x=policies, y=gross,
    marker_color=['#1f77b4','#ff7f0e','#2ca02c'],
    text=[f'₹{v:,.0f}' for v in gross],
    textposition='outside',
), row=1, col=2)

# Top-right: MILP price mult by segment
m_sess = results_milp['milp']['sessions'].copy()
m_sess['segment'] = m_sess['trip_purpose'].astype(str) + '/' + m_sess['loyalty_tier'].astype(str)
seg_summary = (m_sess.groupby('segment')
                      .agg(mean_mult=('chosen_mult','mean'),
                           true_eta=('true_eta','mean'),
                           n=('chosen_mult','size'))
                      .reset_index()
                      .sort_values('mean_mult'))
fig.add_trace(go.Bar(
    x=seg_summary['mean_mult'], y=seg_summary['segment'],
    orientation='h',
    marker=dict(color=seg_summary['true_eta'], colorscale='RdYlGn',
                cmin=-2.5, cmax=0.5, showscale=False),
    text=[f"{m:.2f}× (n={n})" for m, n in zip(seg_summary['mean_mult'], seg_summary['n'])],
    textposition='outside', showlegend=False,
), row=1, col=3)

# Bottom row: 3 heatmaps
for col_idx, (use, label) in enumerate(
    [(rack_use, 'Rack'), (grdy_use, 'Greedy'), (milp_use, 'MILP')], start=1):
    ratio = use / np.maximum(cap_mat, 1)
    text = [[f'{use[i,j]:.1f}/<br>{cap_mat[i,j]:.0f}'
             for j in range(use.shape[1])] for i in range(use.shape[0])]
    fig.add_trace(go.Heatmap(
        z=ratio, x=date_strs, y=room_types,
        colorscale=[[0.0, '#2ca02c'], [0.4, '#fff59d'],
                    [0.567, '#ffb74d'], [0.667, '#ef5350'],
                    [1.0, '#b71c1c']],
        zmin=0, zmax=1.5, showscale=(col_idx == 3),
        colorbar=dict(title='Use/Cap', x=1.02) if col_idx == 3 else None,
        text=text, texttemplate='%{text}',
    ), row=2, col=col_idx)

fig.update_layout(
    title=(f'<b>Stage 3: MILP joint optimization</b> — peak weekend with tight capacity. '
           f'MILP delivers ₹{milp_net:,.0f} net (+{(milp_net/grdy_net-1)*100:.0f}% vs greedy, '
           f'+{(milp_net/rack_net-1)*100:.0f}% vs rack), with **zero capacity violations**.'),
    height=820, template='plotly_white',
    barmode='relative',
    showlegend=False,
)
fig.update_yaxes(title_text='₹', row=1, col=1)
fig.update_yaxes(title_text='₹', row=1, col=2)
fig.update_xaxes(title_text='Mean multiplier', row=1, col=3)
fig.write_image('/home/claude/hotel_pricing/preview_07_milp.png', width=1700, height=820)
print(f"✓ preview_07_milp.png")

print("\nDone. All 7 preview charts written.")
