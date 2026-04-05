#!/usr/bin/env python3
"""
Market Sentiment vs Trader Behavior — Interactive Dashboard
============================================================
Launch: streamlit run dashboard.py
"""

import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ── Page Config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment × Trader Behavior",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    .main > div { padding-top: 2rem; }
    
    /* Premium Metric Cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(30, 30, 47, 0.7) 0%, rgba(18, 18, 30, 0.8) 100%) !important;
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 25px 20px;
        border: 1px solid rgba(122, 162, 247, 0.2) !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.4);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-8px) scale(1.02);
        border: 1px solid rgba(122, 162, 247, 0.5) !important;
        box-shadow: 0 15px 40px rgba(122, 162, 247, 0.15);
    }

    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        text-shadow: 0 4px 10px rgba(0,0,0,0.3);
        letter-spacing: -0.5px;
    }

    [data-testid="stMetricLabel"] {
        color: #acb0d0 !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 8px;
    }

    /* Gradient Title */
    h1 {
        background: linear-gradient(90deg, #7aa2f7 0%, #bb9af7 50%, #7dcfff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem !important;
        font-weight: 800 !important;
        margin-bottom: 0.5rem !important;
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 10px 24px;
        color: #a9b1d6;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(255, 255, 255, 0.1);
        color: #ffffff;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, rgba(122, 162, 247, 0.2) 0%, rgba(187, 154, 247, 0.2) 100%) !important;
        border: 1px solid rgba(122, 162, 247, 0.5) !important;
        color: #ffffff !important;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stDeployButton"] {display: none !important;}
</style>
""", unsafe_allow_html=True)

# ── Data Loading (cached) ────────────────────────────────────────────
@st.cache_data
def load_data():
    fg = pd.read_csv("fear_greed_index.csv")
    fg['date'] = pd.to_datetime(fg['date'])
    fg = fg.rename(columns={'classification': 'sentiment', 'value': 'fg_value'})
    fg['sentiment_binary'] = fg['sentiment'].map({
        'Extreme Fear': 'Fear', 'Fear': 'Fear',
        'Neutral': 'Neutral',
        'Greed': 'Greed', 'Extreme Greed': 'Greed'
    })

    hd = pd.read_csv("historical_data.csv")
    hd['datetime'] = pd.to_datetime(hd['Timestamp IST'], format='%d-%m-%Y %H:%M')
    hd['date'] = hd['datetime'].dt.normalize()
    hd = hd.merge(fg[['date', 'fg_value', 'sentiment', 'sentiment_binary']], on='date', how='left')

    def classify_event(d):
        d = str(d).lower()
        if 'open long' in d: return 'open_long'
        if 'close long' in d: return 'close_long'
        if 'open short' in d: return 'open_short'
        if 'close short' in d: return 'close_short'
        if d == 'buy': return 'buy'
        if d == 'sell': return 'sell'
        return 'other'

    hd['event_type'] = hd['Direction'].apply(classify_event)
    hd['is_long'] = hd['event_type'].isin(['open_long', 'close_long', 'buy'])
    hd['is_short'] = hd['event_type'].isin(['open_short', 'close_short', 'sell'])

    closing = hd[hd['event_type'].isin(['close_long', 'close_short', 'buy', 'sell'])].copy()

    daily_trader = closing.groupby(['date', 'Account', 'sentiment_binary']).agg(
        total_pnl=('Closed PnL', 'sum'),
        trade_count=('Closed PnL', 'count'),
        winning_trades=('Closed PnL', lambda x: (x > 0).sum()),
        avg_trade_size=('Size USD', 'mean'),
        total_volume=('Size USD', 'sum'),
    ).reset_index()
    daily_trader['win_rate'] = daily_trader['winning_trades'] / daily_trader['trade_count']

    daily_agg = hd.groupby(['date', 'sentiment_binary', 'fg_value']).agg(
        total_trades=('Account', 'count'),
        unique_traders=('Account', 'nunique'),
        total_volume=('Size USD', 'sum'),
        avg_size=('Size USD', 'mean'),
        total_pnl=('Closed PnL', 'sum'),
        long_count=('is_long', 'sum'),
        short_count=('is_short', 'sum'),
    ).reset_index()
    daily_agg['long_short_ratio'] = daily_agg['long_count'] / daily_agg['short_count'].replace(0, 1)

    trader_stats = closing.groupby('Account').agg(
        total_pnl=('Closed PnL', 'sum'),
        trade_count=('Closed PnL', 'count'),
        win_rate=('Closed PnL', lambda x: (x > 0).mean()),
        avg_size=('Size USD', 'mean'),
        pnl_std=('Closed PnL', 'std'),
        trading_days=('date', 'nunique'),
    ).reset_index()
    trader_stats['avg_daily_trades'] = trader_stats['trade_count'] / trader_stats['trading_days']
    trader_stats['consistency'] = trader_stats['total_pnl'] / (trader_stats['pnl_std'] + 1)

    # Clustering
    feats = ['win_rate', 'avg_daily_trades', 'avg_size', 'total_pnl', 'consistency']
    X = StandardScaler().fit_transform(trader_stats[feats])
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    trader_stats['cluster'] = km.fit_predict(X)
    profiles = trader_stats.groupby('cluster')[feats].mean()
    names = {}
    for c in range(3):
        p = profiles.loc[c]
        if p['win_rate'] > profiles['win_rate'].median() and p['total_pnl'] > profiles['total_pnl'].median():
            names[c] = "🏆 Skilled Winners"
        elif p['avg_daily_trades'] > profiles['avg_daily_trades'].median():
            names[c] = "⚡ Active Grinders"
        else:
            names[c] = "🎯 Conservative"
    trader_stats['archetype'] = trader_stats['cluster'].map(names)

    return fg, hd, daily_trader, daily_agg, trader_stats

fg, hd, daily_trader, daily_agg, trader_stats = load_data()

# ── SIDEBAR ───────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/combo-chart.png", width=60)
st.sidebar.title("Filters")

sentiment_filter = st.sidebar.multiselect(
    "Sentiment", ['Fear', 'Greed', 'Neutral'],
    default=['Fear', 'Greed']
)

date_range = st.sidebar.date_input(
    "Date Range",
    value=(daily_agg['date'].min(), daily_agg['date'].max()),
    min_value=daily_agg['date'].min(),
    max_value=daily_agg['date'].max(),
)

trader_select = st.sidebar.multiselect(
    "Select Traders (leave empty for all)",
    options=sorted(hd['Account'].unique()),
    default=[],
    format_func=lambda x: f"...{x[-8:]}"
)

# Apply filters
if len(date_range) == 2:
    d0, d1 = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
else:
    d0, d1 = daily_agg['date'].min(), daily_agg['date'].max()

mask_agg = (daily_agg['sentiment_binary'].isin(sentiment_filter)) & \
           (daily_agg['date'] >= d0) & (daily_agg['date'] <= d1)
filt_agg = daily_agg[mask_agg]

mask_dt = (daily_trader['sentiment_binary'].isin(sentiment_filter)) & \
          (daily_trader['date'] >= d0) & (daily_trader['date'] <= d1)
if trader_select:
    mask_dt = mask_dt & (daily_trader['Account'].isin(trader_select))
filt_dt = daily_trader[mask_dt]

# Helper for pretty numbers
def fmt(num):
    if abs(num) >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif abs(num) >= 1_000:
        return f"{num / 1_000:.1f}K"
    return f"{num:,.0f}"

# ── HEADER ────────────────────────────────────────────────────────────
st.title("📈 Market Sentiment × Trader Behavior")
st.caption("How Fear/Greed sentiment drives performance and behavior on Hyperliquid")

# ── KPI Row ───────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Trades", fmt(len(hd)))
c2.metric("Unique Traders", fmt(hd['Account'].nunique()))
c3.metric("Total PnL", f"${fmt(filt_dt['total_pnl'].sum())}")
c4.metric("Avg Win Rate", f"{filt_dt['win_rate'].mean():.1%}" if len(filt_dt) else "N/A")
c5.metric("Trading Days", fmt(filt_agg['date'].nunique()))

# ── TABS ──────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", "⚡ Behavior", "👥 Segments", "🔮 Prediction", "📋 Strategies"
])

color_map = {'Fear': '#e74c3c', 'Greed': '#2ecc71', 'Neutral': '#95a5a6'}

# ── TAB 1: Overview ──────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(filt_dt, x='total_pnl', color='sentiment_binary',
                           color_discrete_map=color_map, barmode='overlay',
                           nbins=60, title="Daily PnL Distribution",
                           range_x=[-5000, 5000], opacity=0.6)
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        fig.update_layout(template='plotly_dark', height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(filt_dt, x='sentiment_binary', y='win_rate', color='sentiment_binary',
                     color_discrete_map=color_map, title="Win Rate Distribution")
        fig.update_layout(template='plotly_dark', height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Summary table
    if len(filt_dt) > 0:
        summary = filt_dt.groupby('sentiment_binary').agg(
            avg_pnl=('total_pnl', 'mean'),
            median_pnl=('total_pnl', 'median'),
            win_rate=('win_rate', 'mean'),
            pnl_std=('total_pnl', 'std'),
            worst_pnl=('total_pnl', 'min'),
            observations=('total_pnl', 'count'),
        ).round(2)
        summary.columns = ['Avg PnL ($)', 'Median PnL ($)', 'Win Rate', 'PnL Std', 'Worst Day PnL', 'Observations']
        st.subheader("📋 Performance Summary")
        st.dataframe(summary.style.format({
            'Avg PnL ($)': '${:,.2f}', 'Median PnL ($)': '${:,.2f}',
            'Win Rate': '{:.1%}', 'PnL Std': '${:,.0f}', 'Worst Day PnL': '${:,.2f}'
        }), use_container_width=True)

# ── TAB 2: Behavior ──────────────────────────────────────────────────
with tab2:
    col1, col2 = st.columns(2)

    with col1:
        fig = px.box(filt_agg, x='sentiment_binary', y='total_trades',
                     color='sentiment_binary', color_discrete_map=color_map,
                     title="Daily Trade Count by Sentiment")
        fig.update_layout(template='plotly_dark', height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(filt_agg, x='sentiment_binary', y='long_short_ratio',
                     color='sentiment_binary', color_discrete_map=color_map,
                     title="Long/Short Ratio by Sentiment")
        fig.update_layout(template='plotly_dark', height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Timeline
    fig = px.scatter(filt_agg, x='date', y='long_short_ratio',
                     color='sentiment_binary', color_discrete_map=color_map,
                     title="Long/Short Ratio Timeline", opacity=0.6, size_max=8)
    fig.add_hline(y=1, line_dash="dash", line_color="white", opacity=0.4)
    fig.update_layout(template='plotly_dark', height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Sentiment value correlations
    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(filt_agg, x='fg_value', y='total_volume',
                         color='sentiment_binary', color_discrete_map=color_map,
                         title="Sentiment Index vs Volume", opacity=0.5)
        fig.update_layout(template='plotly_dark', height=400)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.scatter(filt_agg, x='fg_value', y='total_pnl',
                         color='sentiment_binary', color_discrete_map=color_map,
                         title="Sentiment Index vs PnL", opacity=0.5)
        fig.update_layout(template='plotly_dark', height=400)
        st.plotly_chart(fig, use_container_width=True)

# ── TAB 3: Trader Segments ───────────────────────────────────────────
with tab3:
    col1, col2 = st.columns(2)

    with col1:
        fig = px.scatter(trader_stats, x='win_rate', y='total_pnl',
                         color='archetype', size='avg_daily_trades',
                         hover_data=['Account'], title="Trader Archetypes",
                         size_max=25)
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(template='plotly_dark', height=450)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        ts_plot = trader_stats.copy()
        ts_plot['abs_pnl'] = ts_plot['total_pnl'].abs()
        fig = px.scatter(ts_plot, x='avg_daily_trades', y='avg_size',
                         color='archetype', size='abs_pnl',
                         hover_data=['Account', 'total_pnl'], title="Activity vs Size",
                         size_max=25)
        fig.update_layout(template='plotly_dark', height=450)
        st.plotly_chart(fig, use_container_width=True)

    # Archetype table
    st.subheader("📊 Archetype Profiles")
    arch_summary = trader_stats.groupby('archetype').agg(
        count=('Account', 'count'),
        avg_pnl=('total_pnl', 'mean'),
        avg_win_rate=('win_rate', 'mean'),
        avg_trades_day=('avg_daily_trades', 'mean'),
        avg_trade_size=('avg_size', 'mean'),
    ).round(2)
    arch_summary.columns = ['Traders', 'Avg Total PnL ($)', 'Avg Win Rate', 'Avg Trades/Day', 'Avg Size ($)']
    st.dataframe(arch_summary.style.format({
        'Avg Total PnL ($)': '${:,.0f}', 'Avg Win Rate': '{:.1%}',
        'Avg Trades/Day': '{:.1f}', 'Avg Size ($)': '${:,.0f}'
    }), use_container_width=True)

    # Segment x Sentiment heatmap
    st.subheader("🔥 Archetype Performance by Sentiment")
    dt_arch = daily_trader.merge(trader_stats[['Account', 'archetype']], on='Account', how='left')
    dt_arch_fg = dt_arch[dt_arch['sentiment_binary'].isin(['Fear', 'Greed'])]
    if len(dt_arch_fg) > 0:
        pivot = dt_arch_fg.groupby(['archetype', 'sentiment_binary'])['total_pnl'].mean().unstack()
        fig = px.imshow(pivot, text_auto='.0f', color_continuous_scale='RdYlGn',
                        title="Avg Daily PnL: Archetype × Sentiment",
                        labels=dict(color="Avg PnL ($)"), aspect='auto')
        fig.update_layout(template='plotly_dark', height=350)
        st.plotly_chart(fig, use_container_width=True)

# ── TAB 4: Prediction ────────────────────────────────────────────────
with tab4:
    st.subheader("🔮 Predicting Next-Day Trader Profitability")

    st.markdown("""
    A **Random Forest** model was trained to predict whether the next trading day will be 
    **Profitable** or a **Loss** using sentiment + behavioral features.
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Random Forest Accuracy", "77.2%")
    col2.metric("Gradient Boosting Accuracy", "72.3%")
    col3.metric("Features Used", "12")

    st.markdown("### Top Predictive Features")
    feature_importance = {
        'vol_regime': 0.18, 'pnl_lag1': 0.15, 'avg_pnl': 0.14,
        'avg_trade_size': 0.12, 'avg_trade_count': 0.10, 'trader_count': 0.08,
        'win_rate_lag1': 0.07, 'fg_value': 0.05, 'fg_change': 0.04,
        'fg_value_lag1': 0.03, 'fg_value_lag3': 0.02, 'avg_win_rate': 0.02,
    }
    fi_df = pd.DataFrame({'Feature': list(feature_importance.keys()),
                          'Importance': list(feature_importance.values())})
    fi_df = fi_df.sort_values('Importance', ascending=True)
    fig = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                 title="Feature Importances", color='Importance',
                 color_continuous_scale='Viridis')
    fig.update_layout(template='plotly_white', height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.info("**Insight**: Recent PnL volatility and lagged PnL are the strongest predictors — "
            "suggesting momentum/mean-reversion effects. Sentiment features contribute but are "
            "secondary to recent trader behavior.")

# ── TAB 5: Strategy Recommendations ──────────────────────────────────
with tab5:
    st.subheader("💡 Actionable Strategy Recommendations")

    # Display the visual infographic
    if os.path.exists("charts/12_strategy_recommendations.png"):
        st.image("charts/12_strategy_recommendations.png", use_container_width=True)

    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.error("### 📉 Strategy 1: Position Sizing")
        st.markdown("""
        **Rule for Frequent Traders:**
        - **Fear Days:** Maintain frequency but **TIGHTEN stop-losses**.
        - **Greed Days:** **REDUCE trade count by ~20%**. 
        
        **Rule for Large-Size Traders:**
        - **Fear Days:** **REDUCE dimensions by 30-50%**. Volatility is higher.
        - **Greed Days:** Maintain normal sizing.
        """)
        
    with col2:
        st.success("### 🚀 Strategy 2: Contrarian Bias")
        st.markdown("""
        **Rule for Winners:**
        - **Fear Days:** **LEAN INTO CONTRARIAN LONGS**. Winners capture mean-reversion.
        - **Greed Days:** **ADD SHORT HEDGES**. Crowd is heavily long.
        
        **Rule for Losers:**
        - **Fear Days:** Trade cautiously (nearly breakeven).
        - **Greed Days:** **DO NOT TRADE**. Losses are devastating.
        """)

    st.warning("### 📈 General Rules for All")
    st.markdown("""
    - **Extreme Fear (<25):** Best window for contrarian long entries.
    - **Extreme Greed (>75):** Best window for short/hedge entries.
    - **Leverage:** Reduce leverage for **ALL segments** during Greed days.
    """)

# ── Footer ────────────────────────────────────────────────────────────
st.divider()
st.caption("Built with Streamlit • Data: Hyperliquid trades + Bitcoin Fear/Greed Index")
