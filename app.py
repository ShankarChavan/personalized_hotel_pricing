"""
Streamlit demo: Approach 2 — Two-Stage Propensity × Elasticity for Hotel Pricing.

Run:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_gen import generate_sessions, make_segment
from models import (PropensityModel, ElasticityModel,
                    optimal_price, population_policy_eval, FEATURES_X)
from dml_models import (DMLElasticityModel, compare_elasticity_models,
                         offpolicy_revenue_eval)
from milp_optimizer import (compare_policies, policy_milp_joint,
                             _capacity_use_from_decisions)

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(page_title='Hotel Pricing — Approach 2', layout='wide')
st.title(' Two-Stage Propensity × Elasticity Pricing')
st.caption('Approach 2: P(book | x, p) = P_ref(x) × (p / rack)^η(x) — synthetic hotel booking data')

# ----------------------------------------------------------------------------
# Sidebar — global controls
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header(' Setup')
    n_sessions = st.slider('Number of sessions', 5000, 40000, 20000, step=5000)
    seed = st.number_input('Random seed', 1, 9999, 42)
    train_frac_label = st.slider('Train cutoff (day of 90-day window)', 30, 80, 60)
    st.markdown('---')
    st.markdown('**Pricing grid bounds**')
    p_floor = st.slider('Min price multiplier', 0.40, 1.00, 0.55, step=0.05)
    p_ceiling = st.slider('Max price multiplier', 1.00, 1.30, 1.15, step=0.05)
    st.markdown('---')
    elasticity_choice = st.radio(
        'Stage 2 model for pricing tabs',
        ['GBM (monotone)', 'DML (causal)'],
        index=0,
        help='Which elasticity model drives the Single-Shopper and Population tabs. '
             'Stage 2 GBM and DML tabs always show their own model.')


# ----------------------------------------------------------------------------
# Cached data + models
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner='Generating synthetic sessions...')
def load_data(n, seed):
    df = generate_sessions(n=n, seed=seed)
    df['segment'] = make_segment(df)
    return df


@st.cache_resource(show_spinner='Training models...')
def train_models(_df_train, _df_valid):
    prop = PropensityModel().fit(_df_train, _df_valid)
    elas = ElasticityModel().fit(_df_train, _df_valid)
    dml = DMLElasticityModel(n_folds=5).fit(_df_train)
    return prop, elas, dml


df = load_data(n_sessions, seed)
cutoff_day = train_frac_label
cutoff_date = pd.Timestamp('2026-01-01') + pd.Timedelta(days=cutoff_day)
train_df = df[df['timestamp'] < cutoff_date].copy()
valid_df = df[df['timestamp'] >= cutoff_date].copy()

prop_model, elas_model, dml_model = train_models(train_df, valid_df)

# Active model for the pricing tabs (Single-Shopper, Population)
active_elas = dml_model if elasticity_choice == 'DML (causal)' else elas_model
active_label = 'DML (causal)' if elasticity_choice == 'DML (causal)' else 'GBM (monotone)'

# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
tab1, tab2, tab3, tab6, tab4, tab5, tab7 = st.tabs([
    ' Data Overview',
    ' Stage 1: Propensity',
    ' Stage 2: GBM Elasticity',
    ' Stage 2: DML Variant',
    ' Single-Shopper Pricing',
    ' Population Policy',
    ' MILP Joint Optimization',
])

# ============================================================================
# TAB 1 — DATA
# ============================================================================
with tab1:
    st.subheader('Synthetic Booking Sessions')

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Sessions', f'{len(df):,}')
    c2.metric('Booking rate', f"{df['booked'].mean():.1%}")
    c3.metric('Mean discount', f"{df['discount_pct'].mean():.1%}")
    c4.metric('Mean rack rate', f"₹{df['rack_rate'].mean():,.0f}")

    st.markdown('**Sample rows**')
    show_cols = ['session_id','room_type','channel','loyalty_tier','trip_purpose',
                 'lead_time_days','rack_rate','offered_price','discount_pct',
                 'occupancy_forecast','booked']
    st.dataframe(df[show_cols].head(15), use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x='discount_pct', nbins=40,
                           title='Discount distribution (key for elasticity ID)',
                           labels={'discount_pct': 'Discount %'})
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        agg = (df.assign(bucket=pd.cut(df['discount_pct'],
                                        bins=[-.01,.02,.05,.10,.15,.25,.50],
                                        labels=['0-2%','2-5%','5-10%','10-15%','15-25%','25%+']))
                 .groupby('bucket', observed=True)['booked']
                 .agg(['mean','size']).reset_index())
        fig = px.bar(agg, x='bucket', y='mean',
                     title='Booking rate by discount bucket',
                     labels={'mean':'Booking rate', 'bucket':'Discount'},
                     text=agg['size'].apply(lambda v: f'n={v}'))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('**Booking rate by segment** (this is what the model has to disentangle)')
    seg_table = (df.groupby(['trip_purpose','loyalty_tier'], observed=True)
                   .agg(sessions=('booked','size'),
                        book_rate=('booked','mean'),
                        avg_discount=('discount_pct','mean'),
                        true_eta=('true_eta','mean'))
                   .round(3).reset_index())
    st.dataframe(seg_table, use_container_width=True, hide_index=True)

# ============================================================================
# TAB 2 — STAGE 1 PROPENSITY
# ============================================================================
with tab2:
    st.subheader('Stage 1: P_ref(x) — Baseline Booking Probability at Rack Rate')

    m = prop_model.metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('AUC', f"{m['auc']:.3f}")
    c2.metric('Log Loss', f"{m['log_loss']:.3f}")
    c3.metric('Brier Score', f"{m['brier']:.3f}")
    c4.metric('Calib (mean p vs y)', f"{m['mean_p']:.3f} / {m['mean_y']:.3f}")

    # Calibration plot
    pred = prop_model.predict_p_ref(valid_df)
    bins = pd.cut(pred, bins=np.linspace(0, 1, 11))
    cal = pd.DataFrame({'p_pred': pred, 'y': valid_df['booked'].values, 'bin': bins})
    cal_agg = (cal.groupby('bin', observed=True)
                  .agg(p_pred=('p_pred','mean'), y_actual=('y','mean'),
                       n=('y','size')).reset_index())

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cal_agg['p_pred'], y=cal_agg['y_actual'],
                                 mode='markers+lines', name='Model',
                                 marker=dict(size=10)))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                                 name='Perfect', line=dict(dash='dash')))
        fig.update_layout(title='Calibration (Stage 1, at rack rate)',
                          xaxis_title='Predicted P(book)', yaxis_title='Actual book rate')
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # feature importance
        imp = pd.DataFrame({
            'feature': prop_model.feats,
            'gain': prop_model.model.booster_.feature_importance(importance_type='gain')
        }).sort_values('gain', ascending=True)
        fig = px.bar(imp, x='gain', y='feature', orientation='h',
                     title='Stage 1 Feature Importance (gain)')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('**Predicted P_ref distribution** — what shoppers would book at rack rate')
    fig = px.histogram(pred, nbins=40,
                       labels={'value': 'P_ref(x)', 'count': 'Sessions'},
                       title='Distribution of baseline acceptance probability')
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 3 — STAGE 2 ELASTICITY
# ============================================================================
with tab3:
    st.subheader('Stage 2: η(x) — Heterogeneous Price Elasticity')

    eta_hat = elas_model.estimate_eta(valid_df)
    valid_df = valid_df.assign(eta_hat=eta_hat)

    c1, c2, c3 = st.columns(3)
    c1.metric('Mean η̂ (estimated)', f"{eta_hat.mean():.2f}")
    c2.metric('Mean η (true)', f"{valid_df['true_eta'].mean():.2f}")
    c3.metric('Range η̂', f"[{eta_hat.min():.2f}, {eta_hat.max():.2f}]")

    st.caption('Negative elasticity (η < 0) means higher price → lower booking. '
               'Monotone constraint enforces η ≤ 0.')

    # Elasticity by segment — true vs estimated
    seg = (valid_df.groupby(['trip_purpose','loyalty_tier'], observed=True)
                   .agg(true_eta=('true_eta','mean'),
                        eta_hat=('eta_hat','mean'),
                        n=('booked','size')).reset_index())
    seg['segment'] = seg['trip_purpose'] + ' / ' + seg['loyalty_tier']

    fig = go.Figure()
    fig.add_trace(go.Bar(name='True η', x=seg['segment'], y=seg['true_eta'],
                         marker_color='#1f77b4'))
    fig.add_trace(go.Bar(name='Estimated η̂', x=seg['segment'], y=seg['eta_hat'],
                         marker_color='#ff7f0e'))
    fig.update_layout(barmode='group',
                      title='Elasticity by segment: true vs estimated',
                      yaxis_title='Elasticity (η)')
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        # Elasticity vs lead time
        lead_buckets = pd.cut(valid_df['lead_time_days'],
                              bins=[0,3,7,14,30,60,200],
                              labels=['1-3','4-7','8-14','15-30','31-60','60+'])
        lead_eta = (valid_df.assign(bucket=lead_buckets)
                            .groupby('bucket', observed=True)['eta_hat'].mean().reset_index())
        fig = px.bar(lead_eta, x='bucket', y='eta_hat',
                     title='Estimated elasticity by lead-time bucket',
                     labels={'eta_hat':'η̂', 'bucket':'Lead time (days)'})
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Elasticity vs occupancy
        occ_buckets = pd.cut(valid_df['occupancy_forecast'],
                             bins=[0,0.5,0.7,0.85,1.0],
                             labels=['<50%','50-70%','70-85%','85%+'])
        occ_eta = (valid_df.assign(bucket=occ_buckets)
                           .groupby('bucket', observed=True)['eta_hat'].mean().reset_index())
        fig = px.bar(occ_eta, x='bucket', y='eta_hat',
                     title='Estimated elasticity by occupancy bucket',
                     labels={'eta_hat':'η̂', 'bucket':'Occupancy forecast'})
        st.plotly_chart(fig, use_container_width=True)

    st.info('💡 Reading the chart: business + high-loyalty + late + high-occupancy shoppers '
            'are less elastic (η closer to 0) — the model can charge them more without losing them. '
            'Leisure + far-out + low-occupancy shoppers are more elastic (η more negative) — '
            'discounts move them.')

# ============================================================================
# TAB 4 — SINGLE SHOPPER PRICING
# ============================================================================
with tab4:
    st.subheader('💰 Pricing Decision for a Single Shopper')
    st.caption('Adjust the shopper attributes; the model picks the price that maximizes expected revenue.')

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('**Stay**')
        room_type = st.selectbox('Room type', ['standard','deluxe','suite'])
        lead_time_days = st.select_slider('Lead time (days)',
                                           options=[1,3,7,14,30,60,90,120], value=14)
        length_of_stay = st.select_slider('Length of stay', options=[1,2,3,4,5,7], value=2)
        is_weekend_stay = st.checkbox('Weekend stay', value=False)

    with c2:
        st.markdown('**Shopper**')
        channel = st.selectbox('Channel', ['direct_web','direct_app','ota','meta'])
        country = st.selectbox('Country', ['IN','US','UK','AE','SG'])
        device = st.selectbox('Device', ['ios','android','desktop'])
        loyalty_tier = st.selectbox('Loyalty tier', ['none','silver','gold','platinum'])
        prior_bookings = st.slider('Prior bookings', 0, 15, 2)
        trip_purpose = st.selectbox('Trip purpose', ['business','leisure'])

    with c3:
        st.markdown('**Demand context**')
        occupancy_forecast = st.slider('Occupancy forecast', 0.20, 0.98, 0.70, step=0.05)
        seasonality_index = st.slider('Seasonality index', 0.60, 1.50, 1.00, step=0.05)
        rack_rate = st.number_input('Rack rate (₹)', 3000, 50000, 8500, step=500)
        comp_price = st.number_input('Competitor median (₹)', 3000, 50000, 8200, step=500)
        variable_cost = st.number_input('Variable cost (₹)', 500, 20000, 2125, step=100)

    # Build the row
    row = pd.Series({
        'room_type': room_type,
        'channel': channel,
        'country': country,
        'device': device,
        'loyalty_tier': loyalty_tier,
        'trip_purpose': trip_purpose,
        'lead_time_days': lead_time_days,
        'length_of_stay': length_of_stay,
        'is_weekend_stay': int(is_weekend_stay),
        'prior_bookings': prior_bookings,
        'occupancy_forecast': occupancy_forecast,
        'seasonality_index': seasonality_index,
        'rack_rate': rack_rate,
        'competitor_median_price': comp_price,
        'price_index': rack_rate / comp_price,
        'variable_cost': variable_cost,
    })

    best, curve = optimal_price(row, prop_model, active_elas,
                                price_floor=p_floor, price_ceiling=p_ceiling)
    st.caption(f'Pricing with **{active_label}** elasticity model (change in sidebar).')

    # Top metric strip
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric('P_ref (at rack)', f"{curve['p_ref'].iloc[0]:.3f}")
    c2.metric('Estimated η̂', f"{curve['eta'].iloc[0]:.2f}")
    c3.metric('Optimal price', f"₹{best['price']:,.0f}",
              f"{(best['price_mult']-1)*100:+.1f}% vs rack")
    c4.metric('P(book) at optimal', f"{best['p_book']:.3f}")
    c5.metric('Expected revenue', f"₹{best['expected_rev']:,.0f}")

    # Curve plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve['price'], y=curve['p_book'],
                             mode='lines', name='P(book)', yaxis='y1',
                             line=dict(color='#1f77b4', width=3)))
    fig.add_trace(go.Scatter(x=curve['price'], y=curve['expected_rev'],
                             mode='lines', name='Expected revenue', yaxis='y2',
                             line=dict(color='#2ca02c', width=3)))
    fig.add_vline(x=best['price'], line_dash='dash', line_color='red',
                  annotation_text=f"Optimal: ₹{best['price']:,.0f}")
    fig.add_vline(x=rack_rate, line_dash='dot', line_color='gray',
                  annotation_text=f'Rack: ₹{rack_rate:,.0f}')
    fig.update_layout(
        title='Acceptance & Expected Revenue across price grid',
        xaxis_title='Offered price (₹)',
        yaxis=dict(title='P(book)', side='left', range=[0,1]),
        yaxis2=dict(title='Expected revenue (₹)', side='right', overlaying='y'),
        legend=dict(x=0.02, y=0.98),
        hovermode='x unified',
    )
    st.plotly_chart(fig, use_container_width=True)

    # vs charging rack rate
    rev_at_rack = curve.loc[(curve['price_mult']-1.0).abs().idxmin(), 'expected_rev']
    lift = (best['expected_rev'] / max(rev_at_rack, 1e-6) - 1) * 100
    st.success(f"📈 Expected revenue lift vs charging rack rate: **{lift:+.1f}%**")

# ============================================================================
# TAB 5 — POPULATION POLICY
# ============================================================================
with tab5:
    st.subheader('🌐 Apply the Pricing Policy Across the Validation Population')
    st.caption(f'Pricing with **{active_label}** elasticity model (change in sidebar).')

    if st.button('Run population evaluation', type='primary'):
        with st.spinner('Pricing 6,000+ sessions...'):
            sample = valid_df.sample(min(6000, len(valid_df)), random_state=1)
            pop = population_policy_eval(sample, prop_model, active_elas,
                                         price_floor=p_floor, price_ceiling=p_ceiling)
            pop['_model_used'] = active_label

        st.session_state['pop_result'] = pop

    if 'pop_result' in st.session_state:
        pop = st.session_state['pop_result']

        c1, c2, c3, c4 = st.columns(4)
        baseline_rev = pop['baseline_expected_rev'].mean()
        optimal_rev = pop['optimal_expected_rev'].mean()
        lift = (optimal_rev / baseline_rev - 1) * 100

        c1.metric('Baseline RevPAR (rack)', f"₹{baseline_rev:,.0f}")
        c2.metric('Optimized RevPAR', f"₹{optimal_rev:,.0f}", f'{lift:+.1f}%')
        c3.metric('Avg optimal mult', f"{pop['optimal_mult'].mean():.3f}")
        c4.metric('Avg P(book) optimal', f"{pop['optimal_p_book'].mean():.3f}")

        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(pop, x='optimal_mult', nbins=30,
                               title='Distribution of optimal price multipliers',
                               labels={'optimal_mult':'Optimal price / rack'})
            fig.add_vline(x=1.0, line_dash='dash', line_color='red',
                         annotation_text='Rack rate')
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            seg_perf = (pop.assign(segment=pop['trip_purpose'].astype(str)+'_'+pop['loyalty_tier'].astype(str))
                          .groupby('segment')
                          .agg(baseline=('baseline_expected_rev','mean'),
                               optimal=('optimal_expected_rev','mean'),
                               avg_mult=('optimal_mult','mean'),
                               n=('booked','size')).reset_index())
            seg_perf['lift_%'] = ((seg_perf['optimal']/seg_perf['baseline'])-1)*100

            fig = px.bar(seg_perf.sort_values('lift_%'),
                         x='lift_%', y='segment', orientation='h',
                         title='Revenue lift by segment',
                         labels={'lift_%':'Lift vs rack (%)','segment':''},
                         color='avg_mult', color_continuous_scale='RdYlGn_r',
                         hover_data=['n','avg_mult'])
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('**Per-session pricing decisions (sample)**')
        show = (pop[['session_id','trip_purpose','loyalty_tier','lead_time_days',
                     'occupancy_forecast','rack_rate','p_ref','eta_hat',
                     'optimal_mult','optimal_price','optimal_p_book',
                     'baseline_expected_rev','optimal_expected_rev']]
                .head(20).round(3))
        st.dataframe(show, use_container_width=True, hide_index=True)

        # download
        csv = pop.to_csv(index=False).encode('utf-8')
        st.download_button('⬇️ Download full pricing decisions (CSV)',
                          csv, 'pricing_decisions.csv', 'text/csv')
    else:
        st.info('Click **Run population evaluation** to score the validation set with the optimal policy.')

# ============================================================================
# TAB 6 — STAGE 2: DML VARIANT
# ============================================================================
with tab6:
    st.subheader('🧪 Stage 2: DML — Causal Elasticity with Cross-Fit Nuisance')
    st.caption('Double Machine Learning (Chernozhukov et al. 2018). Partial linear model: '
               'Y = θ(X)·T + g(X) + ε. Cross-fit nuisance via LightGBM, '
               'segment-heterogeneous final stage via OLS with HC1 robust SE, '
               'then converted to constant-elasticity scale: η = θ / P̄_segment.')

    # ---- Diagnostics row ---------------------------------------------------
    diag = dml_model.diagnostics_
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Cross-fit folds', dml_model.n_folds)
    c2.metric('Segments', diag['n_segments'])
    c3.metric('A/B test arm share', f"{diag['ab_test_share']:.1%}",
              help='Fraction of training rows with random A/B prices — the source of identification.')
    c4.metric('Residualized p̃ std', f"{diag['p_tilde_std']:.3f}",
              help='Std of treatment after residualizing on X. Larger = stronger identification. '
                   'When A/B share is 0 or features explain price too well, this collapses.')

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('ĝ(X) corr with Y', f"{diag['g_hat_corr_with_y']:.3f}",
              help='How well ĝ(X) = E[Y|X] (cross-fit) predicts the outcome.')
    c2.metric('m̂(X) corr with T', f"{diag['m_hat_corr_with_T']:.3f}",
              help='How well m̂(X) = E[T|X] (cross-fit) predicts the treatment. '
                   'Above ~0.95 risks weak identification.')
    c3.metric('Global η (DML)', f"{diag['global_eta_const_elas']:.3f}",
              f"CI [{diag['global_eta_ci'][0]:.2f}, {diag['global_eta_ci'][1]:.2f}]")
    c4.metric('True mean η (valid)', f"{valid_df['true_eta'].mean():.3f}")

    # ---- Per-segment table ------------------------------------------------
    st.markdown('### Per-segment elasticity (DML, with 95% CIs)')
    cmp_df = compare_elasticity_models(valid_df, 'true_eta', elas_model, dml_model)
    cmp_df = cmp_df[['segment', 'n', 'true_eta', 'eta_naive', 'eta_dml',
                     'lo95', 'hi95', 'naive_bias', 'dml_bias']].copy()
    cmp_df['ci_covers_truth'] = ((cmp_df['lo95'] <= cmp_df['true_eta']) &
                                  (cmp_df['true_eta'] <= cmp_df['hi95']))
    cmp_df = cmp_df.sort_values('true_eta')

    coverage = cmp_df['ci_covers_truth'].sum()
    naive_mae = cmp_df['naive_bias'].abs().mean()
    dml_mae = cmp_df['dml_bias'].abs().mean()

    c1, c2, c3 = st.columns(3)
    c1.metric('DML 95% CI coverage', f'{coverage}/{len(cmp_df)} segments')
    c2.metric('GBM η̂ MAE', f'{naive_mae:.3f}')
    c3.metric('DML η̂ MAE', f'{dml_mae:.3f}',
              f'{(dml_mae - naive_mae):+.3f} vs GBM',
              delta_color='inverse')

    st.dataframe(cmp_df.round(3), use_container_width=True, hide_index=True)

    # ---- Side-by-side bar chart with DML CIs ------------------------------
    st.markdown('### True η vs GBM η̂ vs DML η̂ (with DML 95% CI)')
    fig = go.Figure()
    x_labels = cmp_df['segment'].tolist()
    fig.add_trace(go.Bar(name='True η', x=x_labels, y=cmp_df['true_eta'],
                          marker_color='#1f77b4'))
    fig.add_trace(go.Bar(name='GBM η̂', x=x_labels, y=cmp_df['eta_naive'],
                          marker_color='#ff7f0e'))
    fig.add_trace(go.Bar(name='DML η̂', x=x_labels, y=cmp_df['eta_dml'],
                          marker_color='#2ca02c',
                          error_y=dict(type='data', symmetric=False,
                                       array=cmp_df['hi95'] - cmp_df['eta_dml'],
                                       arrayminus=cmp_df['eta_dml'] - cmp_df['lo95'],
                                       color='#1a5e1a', thickness=1.5)))
    fig.add_hline(y=0, line_dash='dot', line_color='gray')
    fig.update_layout(barmode='group',
                      title='Per-segment elasticity: True vs Naive GBM vs DML',
                      yaxis_title='Elasticity (η)',
                      legend=dict(orientation='h', y=1.10, x=0.5, xanchor='center'),
                      height=480)
    st.plotly_chart(fig, use_container_width=True)

    # ---- Off-policy revenue evaluation ------------------------------------
    st.markdown('### 💰 Off-policy revenue evaluation (true-η counterfactual)')
    st.caption('Each model picks prices using its own η̂. We then score those prices '
               'using the **true η** (only available because the data is synthetic) '
               'to get unbiased actual revenue. The gap between self-estimated and '
               'actual lift = **overconfidence**.')

    if st.button('Run counterfactual revenue evaluation', type='primary'):
        with st.spinner('Pricing 3,000 sessions with each model...'):
            sample = valid_df.sample(min(3000, len(valid_df)), random_state=2)

            class OracleEta:
                def estimate_eta(self, df): return df['true_eta'].values

            res_oracle = offpolicy_revenue_eval(sample, prop_model, OracleEta(),
                                                 price_floor=p_floor, price_ceiling=p_ceiling)
            res_gbm = offpolicy_revenue_eval(sample, prop_model, elas_model,
                                              price_floor=p_floor, price_ceiling=p_ceiling)
            res_dml = offpolicy_revenue_eval(sample, prop_model, dml_model,
                                              price_floor=p_floor, price_ceiling=p_ceiling)
            st.session_state['offpolicy'] = (res_oracle, res_gbm, res_dml)

    if 'offpolicy' in st.session_state:
        res_oracle, res_gbm, res_dml = st.session_state['offpolicy']

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('**Oracle** (uses true η)')
            st.metric('Actual lift', f"{res_oracle['lift_pct']:+.1f}%",
                       help='Best achievable — upper bound.')
            st.metric('Self-estimated', f"{res_oracle['self_estimated_lift_pct']:+.1f}%")
            st.metric('Mean chosen mult', f"{res_oracle['mean_chosen_mult']:.3f}")
        with c2:
            st.markdown('**GBM (monotone)**')
            st.metric('Actual lift', f"{res_gbm['lift_pct']:+.1f}%",
                      f"{res_gbm['lift_pct'] - res_oracle['lift_pct']:+.1f} pts vs oracle",
                      delta_color='off')
            st.metric('Self-estimated', f"{res_gbm['self_estimated_lift_pct']:+.1f}%",
                       help='What the model thinks it will achieve.')
            st.metric('Overconfidence', f"+{res_gbm['overconfidence_pct']:.1f} pts",
                       help='Self-estimated minus actual. Higher = more overconfident.',
                       delta_color='inverse')
        with c3:
            st.markdown('**DML (causal)**')
            st.metric('Actual lift', f"{res_dml['lift_pct']:+.1f}%",
                      f"{res_dml['lift_pct'] - res_oracle['lift_pct']:+.1f} pts vs oracle",
                      delta_color='off')
            st.metric('Self-estimated', f"{res_dml['self_estimated_lift_pct']:+.1f}%")
            st.metric('Overconfidence', f"+{res_dml['overconfidence_pct']:.1f} pts",
                       delta_color='inverse')

        # Bar chart comparison
        comp_data = pd.DataFrame([
            {'Model': 'Oracle (true η)',
             'Actual lift': res_oracle['lift_pct'],
             'Self-estimated lift': res_oracle['self_estimated_lift_pct']},
            {'Model': 'GBM (monotone)',
             'Actual lift': res_gbm['lift_pct'],
             'Self-estimated lift': res_gbm['self_estimated_lift_pct']},
            {'Model': 'DML (causal)',
             'Actual lift': res_dml['lift_pct'],
             'Self-estimated lift': res_dml['self_estimated_lift_pct']},
        ])
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Self-estimated', x=comp_data['Model'],
                              y=comp_data['Self-estimated lift'],
                              marker_color='#bbb',
                              text=[f"{v:+.1f}%" for v in comp_data['Self-estimated lift']],
                              textposition='outside'))
        fig.add_trace(go.Bar(name='Actual', x=comp_data['Model'],
                              y=comp_data['Actual lift'],
                              marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'],
                              text=[f"{v:+.1f}%" for v in comp_data['Actual lift']],
                              textposition='outside'))
        fig.update_layout(barmode='group',
                          title='Self-estimated vs actual revenue lift — the overconfidence gap',
                          yaxis_title='RevPAR lift vs rack rate (%)',
                          height=400,
                          legend=dict(orientation='h', y=1.10, x=0.5, xanchor='center'))
        st.plotly_chart(fig, use_container_width=True)

        st.success(
            f'**Bottom line:** GBM thinks it gets +{res_gbm["self_estimated_lift_pct"]:.1f}% '
            f'but actually delivers +{res_gbm["lift_pct"]:.1f}% '
            f'({res_gbm["overconfidence_pct"]:.1f} pts overconfidence). '
            f'DML delivers +{res_dml["lift_pct"]:.1f}% with only {res_dml["overconfidence_pct"]:.1f} pts overconfidence — '
            f'a more honest, deployable estimate.')
    else:
        st.info('Click the button above to run the counterfactual evaluation.')

    # ---- Why DML helps ----------------------------------------------------
    with st.expander('📚 Why DML, and when does it actually help?'):
        st.markdown("""
**DML's three claims:**

1. **Orthogonalization** removes 1st-order bias from observed confounders.
   In hotel pricing, rack rates respond to occupancy, lead time, and seasonality — all of which also affect bookings.
   Naive ML attributes some of that demand signal to "price," biasing η̂ toward zero.
2. **Cross-fitting** (K-fold out-of-fold prediction) prevents own-sample overfitting bias in the nuisance models.
3. **Linear final stage** gives analytical confidence intervals (HC1 robust SE).

**Identification depends on residualized treatment variance.**
DML works when `p̃ = T − m̂(X)` has enough variance left after residualization.
The diagnostics panel shows `m̂ corr with T = {:.2f}` and `p̃ std = {:.3f}`.
If the A/B test share is 0, m̂ would explain almost all price variation and DML loses identification — try it!

**The constant-elasticity scale conversion.**
Partial linear DML estimates θ on the **probability scale**: `P(book) ≈ θ · log_price_ratio + g(X)`.
But the pricing formula `p_book = p_ref · m^η` uses **log-probability** (constant-elasticity) scale.
We convert per-segment via the first-order Taylor expansion: `η = θ / P̄_segment`.
This is exact at the segment's mean booking rate; for very elastic segments where price changes are large,
the linear approximation breaks down and DML estimates carry residual bias.

**When to use which:**
- **GBM (monotone)** is fine when (a) price was randomized in your data and (b) you trust monotonicity.
  Lower variance, no causal guarantees.
- **DML (causal)** is the production choice when (a) price is set by a dynamic pricing engine (selection on X)
  and (b) you need trustworthy elasticities for revenue management. Wider intervals, but honest about uncertainty.
""".format(diag['m_hat_corr_with_T'], diag['p_tilde_std']))

# ============================================================================
# TAB 7 — MILP JOINT OPTIMIZATION (capacity + pricing)
# ============================================================================
with tab7:
    st.subheader('🧮 Joint Pricing + Capacity Allocation MILP')
    st.caption('When inventory is scarce, per-session greedy pricing oversells. '
               'The MILP simultaneously chooses prices for all active sessions '
               'subject to per-(date, room_type) capacity, with soft constraints '
               '(walk-fee penalty on overflow).')

    with st.expander('📚 Formulation'):
        st.markdown(r'''
**Decision variables:** $x_{i,j} \in \{0,1\}$ — session $i$ gets price multiplier $m_j$.
Exactly one $j$ per $i$.

**Pre-computed from Stage 1 + Stage 2:**
- $p^{book}_{i,j} = P_\text{ref}(x_i) \cdot m_j^{\eta(x_i)}$
- $\text{rev}_{i,j} = p^{book}_{i,j} \cdot (m_j \cdot \text{rack}_i - \text{cost}_i)$

**Soft capacity** (with overflow slack $o_c \geq 0$ for cell $c=(\text{date},\text{room})$):
$$\sum_{i \in S(c)} \sum_j x_{i,j}\, p^{book}_{i,j} \;-\; o_c \;\leq\; \text{cap}_c$$

**Objective** (revenue minus walk-fee penalty $\lambda$ ≈ 2× rack):
$$\max \; \sum_{i,j} x_{i,j}\, \text{rev}_{i,j} \;-\; \lambda \sum_c o_c$$

Solved by CBC via PuLP. Linear in $x$ (since $p^{book}$ is precomputed).
For chance constraints / overbooking risk control: replace $\sum p^{book}$ with $\mu_c + 1.65\sigma_c$.
''')

    # ---- Scenario controls ------------------------------------------------
    st.markdown('### 🗓️ Peak weekend scenario')
    c1, c2, c3 = st.columns(3)
    with c1:
        peak_start = st.date_input(
            'Peak weekend start',
            value=pd.Timestamp('2026-03-13').date(),
            min_value=pd.Timestamp('2026-02-01').date(),
            max_value=pd.Timestamp('2026-04-30').date())
    with c2:
        peak_n_nights = st.slider('Number of nights', 1, 5, 3)
    with c3:
        which_eta = st.radio(
            'Elasticity model',
            ['DML (causal)', 'GBM (monotone)'],
            index=0, horizontal=False,
            help='Which Stage 2 model feeds the MILP. DML usually gives better MILP outcomes.')
    chosen_eta_model = dml_model if which_eta == 'DML (causal)' else elas_model

    st.markdown('### 🛏️ Capacity per room type per night')
    c1, c2, c3 = st.columns(3)
    cap_std = c1.slider('Standard rooms / night', 10, 100, 40, step=5)
    cap_dlx = c2.slider('Deluxe rooms / night', 5, 60, 28, step=5)
    cap_ste = c3.slider('Suites / night', 1, 25, 9, step=1)

    st.markdown('### 💸 MILP price grid')
    c1, c2, c3 = st.columns(3)
    p_grid_min = c1.slider('Min multiplier', 0.40, 1.00, 0.55, step=0.05, key='milp_min')
    p_grid_max = c2.slider('Max multiplier', 1.20, 2.50, 1.80, step=0.10, key='milp_max')
    n_grid = c3.slider('Grid points', 10, 40, 25, step=5, key='milp_grid')

    # ---- Filter active sessions -------------------------------------------
    peak_dates = pd.date_range(pd.Timestamp(peak_start),
                               periods=peak_n_nights, freq='D')

    def _stay_fully_in(row, dates=peak_dates):
        a = pd.Timestamp(row['arrival_date']).normalize()
        los = int(row['length_of_stay'])
        return a >= dates[0] and a + pd.Timedelta(days=los-1) <= dates[-1]

    @st.cache_data(show_spinner='Filtering active sessions...')
    def _filter_active(_valid_df, peak_start_str, n_nights):
        dates = pd.date_range(pd.Timestamp(peak_start_str), periods=n_nights, freq='D')
        mask = _valid_df.apply(
            lambda r: (pd.Timestamp(r['arrival_date']).normalize() >= dates[0]
                       and pd.Timestamp(r['arrival_date']).normalize()
                           + pd.Timedelta(days=int(r['length_of_stay'])-1) <= dates[-1]),
            axis=1)
        return _valid_df[mask].copy()

    active = _filter_active(valid_df, str(peak_start), peak_n_nights)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Active sessions', len(active))
    c2.metric('Total room-nights cap',
              cap_std * peak_n_nights + cap_dlx * peak_n_nights + cap_ste * peak_n_nights)
    if len(active) > 0:
        rough_demand = (active['length_of_stay'] * 0.5).sum()  # at ~50% baseline book rate
        c3.metric('Rough greedy demand', f'~{int(rough_demand)} rn')
    c4.metric('Walk fee per overflow rn',
              f"₹{2.0 * float(active['rack_rate'].mean()):,.0f}" if len(active) else '—',
              help='2× mean rack rate. Industry typical for "walking" overbooked guests to nearby hotels.')

    if len(active) == 0:
        st.warning('No active sessions for this window. Pick another peak date or expand the range.')
        st.stop()

    # ---- Run button --------------------------------------------------------
    run_milp = st.button('🚀 Run all 3 policies (rack / greedy / MILP)', type='primary')

    @st.cache_data(show_spinner='Solving MILP — this takes ~30–60s...')
    def _run_compare(_active, _prop_model_id, _eta_model_id,
                     peak_start_str, n_nights,
                     cap_std, cap_dlx, cap_ste,
                     p_min, p_max, n_grid_pts):
        # cache key includes scalar params; models passed by reference
        dates = pd.date_range(pd.Timestamp(peak_start_str), periods=n_nights, freq='D')
        cap = {(d.normalize(), r): {'standard': cap_std,
                                    'deluxe': cap_dlx,
                                    'suite': cap_ste}[r]
               for d in dates for r in ['standard','deluxe','suite']}
        mults = np.linspace(p_min, p_max, n_grid_pts)
        summary, results = compare_policies(
            _active, prop_model, chosen_eta_model, mults, cap, milp_time_limit=45)
        return summary, results, cap, mults

    if run_milp or 'milp_result' in st.session_state:
        if run_milp:
            with st.spinner('Solving... (rack & greedy are instant; MILP ≈30-60s)'):
                summary, results, cap, mults = _run_compare(
                    active, id(prop_model), id(chosen_eta_model),
                    str(peak_start), peak_n_nights,
                    cap_std, cap_dlx, cap_ste,
                    p_grid_min, p_grid_max, n_grid)
            st.session_state['milp_result'] = (summary, results, cap, mults, which_eta)
        else:
            summary, results, cap, mults, _ = st.session_state['milp_result']

        # ---- Headline metrics ----------------------------------------------
        st.markdown('### 💰 Net revenue (after walk-fee for capacity violations)')
        c1, c2, c3 = st.columns(3)
        rack_net = summary.loc[summary.policy == 'Rack rate', 'net_actual_rev'].iloc[0]
        grdy_net = summary.loc[summary.policy == 'Greedy per-session', 'net_actual_rev'].iloc[0]
        milp_net = summary.loc[summary.policy == 'MILP joint', 'net_actual_rev'].iloc[0]

        with c1:
            st.metric('Rack rate', f'₹{rack_net:,.0f}',
                      help='Net = revenue − walk fees on overflow.')
        with c2:
            lift_g_v_r = (grdy_net / rack_net - 1) * 100
            st.metric('Greedy per-session', f'₹{grdy_net:,.0f}',
                      f'{lift_g_v_r:+.1f}% vs rack')
        with c3:
            lift_m_v_r = (milp_net / rack_net - 1) * 100
            lift_m_v_g = (milp_net / grdy_net - 1) * 100
            st.metric('MILP joint', f'₹{milp_net:,.0f}',
                      f'{lift_m_v_g:+.1f}% vs greedy ({lift_m_v_r:+.1f}% vs rack)')

        # ---- Stacked breakdown chart ----------------------------------------
        breakdown = summary[['policy', 'actual_rev_true_eta', 'walk_fee_penalty',
                              'total_overuse_true_eta']].copy()
        breakdown['gross_actual'] = breakdown['actual_rev_true_eta']
        breakdown['net_actual'] = breakdown['gross_actual'] - breakdown['walk_fee_penalty']

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Walk-fee penalty', x=breakdown['policy'],
            y=-breakdown['walk_fee_penalty'],
            marker_color='#d62728',
            text=[f'-₹{v:,.0f}' if v > 0 else '' for v in breakdown['walk_fee_penalty']],
            textposition='outside'))
        fig.add_trace(go.Bar(
            name='Gross actual revenue', x=breakdown['policy'],
            y=breakdown['gross_actual'],
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'],
            text=[f'₹{v:,.0f}' for v in breakdown['gross_actual']],
            textposition='outside'))
        fig.update_layout(
            title='Revenue (gross actual) and walk-fee penalty by policy — under TRUE η',
            yaxis_title='₹', height=420, template='plotly_white',
            barmode='relative',
            legend=dict(orientation='h', y=1.10, x=0.5, xanchor='center'))
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(summary.round(0), use_container_width=True, hide_index=True)

        # ---- Capacity utilization heatmap (per policy) -----------------------
        st.markdown('### 🛏️ Capacity utilization by cell, under TRUE elasticity')

        cap_keys = list(cap.keys())
        room_types = ['standard', 'deluxe', 'suite']
        date_strs = [d.strftime('%a %m-%d') for d in
                     sorted({k[0] for k in cap_keys})]

        def _build_heat(sess, capacity_dict, capacity_keys):
            rack = sess['rack_rate'].values
            mult = sess['chosen_mult'].values
            p_ref = sess['p_ref'].values
            true_eta = sess['true_eta'].values
            actual_p = np.clip(p_ref * (mult ** true_eta), 0, 1)
            cap_use = _capacity_use_from_decisions(sess, actual_p, capacity_keys)
            # Build matrix [room_type x date]
            mat = np.zeros((len(room_types), len(date_strs)))
            cap_mat = np.zeros_like(mat)
            for i, r in enumerate(room_types):
                for j, ds in enumerate(date_strs):
                    d = pd.Timestamp(sorted({k[0] for k in capacity_keys})[j])
                    key = (d, r)
                    mat[i, j] = cap_use.get(key, 0.0)
                    cap_mat[i, j] = capacity_dict.get(key, 1)
            ratio = mat / np.maximum(cap_mat, 1)
            return mat, cap_mat, ratio

        rack_use, _, rack_ratio = _build_heat(results['rack'], cap, cap_keys)
        grdy_use, _, grdy_ratio = _build_heat(results['greedy'], cap, cap_keys)
        milp_use, cap_mat, milp_ratio = _build_heat(results['milp']['sessions'], cap, cap_keys)

        c1, c2, c3 = st.columns(3)
        for col, (title, use, ratio) in zip(
            [c1, c2, c3],
            [('Rack rate', rack_use, rack_ratio),
             ('Greedy', grdy_use, grdy_ratio),
             ('MILP', milp_use, milp_ratio)]):
            text = [[f'{use[i,j]:.1f}/<br>{cap_mat[i,j]:.0f}'
                     for j in range(use.shape[1])] for i in range(use.shape[0])]
            with col:
                fig = go.Figure(data=go.Heatmap(
                    z=ratio, x=date_strs, y=room_types,
                    colorscale=[[0.0, '#2ca02c'], [0.467, '#ffeb3b'],
                                 [0.667, '#ff7f0e'], [1.0, '#d62728']],
                    zmin=0, zmax=1.5, text=text, texttemplate='%{text}',
                    colorbar=dict(title='Use/Cap')))
                fig.update_layout(title=title, height=280, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)

        st.caption('Cells over 1.0× capacity (yellow→red) are oversold under true elasticity → '
                   'walk fees in real life. MILP is the only policy that respects capacity.')

        # ---- Pricing differentiation by segment -----------------------------
        st.markdown('### 🎯 Price multipliers by segment — how does MILP discriminate?')
        m_sess = results['milp']['sessions'].copy()
        m_sess['segment'] = m_sess['trip_purpose'].astype(str) + '_' + m_sess['loyalty_tier'].astype(str)

        seg_summary = (m_sess.groupby('segment')
                              .agg(n=('chosen_mult', 'size'),
                                   mean_mult=('chosen_mult', 'mean'),
                                   median_mult=('chosen_mult', 'median'),
                                   true_eta=('true_eta', 'mean'),
                                   eta_hat=('eta_hat', 'mean'))
                              .reset_index()
                              .sort_values('mean_mult'))

        fig = go.Figure()
        fig.add_trace(go.Bar(x=seg_summary['mean_mult'], y=seg_summary['segment'],
                              orientation='h',
                              marker=dict(color=seg_summary['true_eta'],
                                           colorscale='RdYlGn',
                                           cmin=-2.5, cmax=0.5,
                                           colorbar=dict(title='True η')),
                              text=[f"mult={m:.2f} (n={n})"
                                    for m, n in zip(seg_summary['mean_mult'], seg_summary['n'])],
                              textposition='outside'))
        fig.add_vline(x=1.0, line_dash='dash', line_color='gray',
                       annotation_text='Rack rate')
        fig.update_layout(
            title='Mean MILP price multiplier by segment (color = true elasticity)',
            xaxis_title='Mean chosen multiplier (1.0 = rack)',
            yaxis_title='', height=400, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

        # ---- Pricing decision table ----------------------------------------
        st.markdown('### 📋 Sample of MILP per-session decisions')
        show = (m_sess[['session_id', 'segment', 'room_type', 'arrival_date',
                        'length_of_stay', 'rack_rate', 'p_ref', 'eta_hat',
                        'chosen_mult', 'chosen_price', 'chosen_p_book',
                        'chosen_expected_rev']]
                .head(25).round(3))
        st.dataframe(show, use_container_width=True, hide_index=True)

        # MILP-specific diagnostics
        milp_obj = results['milp']
        st.markdown('### 🔧 MILP diagnostics')
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Solver status', milp_obj['status'])
        c2.metric('Sessions priced out',
                  f"{milp_obj['n_priced_out']}/{len(active)}",
                  help=f'Sessions assigned the maximum mult ({mults[-1]:.2f}×) — effective rationing.')
        c3.metric('Soft overflow (model η)',
                  f"{milp_obj['overflow_total']:.2f} rn",
                  help='Total room-nights of soft constraint slack used.')
        c4.metric('Walk-fee penalty (λ)',
                  f"₹{milp_obj['overflow_penalty']:,.0f}/rn",
                  help='Penalty per overflow room-night in the MILP objective.')

        # CSV download
        csv = m_sess.to_csv(index=False).encode('utf-8')
        st.download_button('⬇️ Download MILP decisions (CSV)', csv,
                          'milp_pricing_decisions.csv', 'text/csv')
    else:
        st.info('Configure the scenario above and click **Run all 3 policies** to compare. '
                'MILP solving takes ~30–60 seconds depending on session count.')

# ----------------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------------
st.markdown('---')
st.caption('**Approach 2 architecture**:  Stage 1 LightGBM (calibrated) for P_ref(x)  ·  '
           'Stage 2 LightGBM (monotone) OR DML (causal) for η(x)  ·  '
           'Stage 3 MILP joint pricing + capacity (CBC via PuLP)  ·  '
           'Combined formula: P(book | x, p) = P_ref(x) × (p/rack)^η(x)')
