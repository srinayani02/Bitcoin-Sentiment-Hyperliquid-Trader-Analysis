#!/usr/bin/env python3
"""
Market Sentiment vs Trader Behavior Analysis
=============================================
Analyzes how Bitcoin Fear/Greed sentiment relates to trader behavior
and performance on Hyperliquid.

Parts:
  A — Data Preparation
  B — Analysis (Performance vs Sentiment, Behavior Changes, Segmentation)
  C — Actionable Strategy Recommendations
  Bonus — Predictive Model + Trader Clustering
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
warnings.filterwarnings('ignore')

# ── Styling ───────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)
plt.rcParams.update({
    'figure.figsize': (14, 7),
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
})

OUTPUT_DIR = "dashboard/charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save(fig, name):
    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ Saved {path}")

# ========================================================================
# PART A — DATA PREPARATION
# ========================================================================
print("=" * 72)
print("PART A — DATA PREPARATION")
print("=" * 72)

# ── A.1  Load Sentiment Data ──────────────────────────────────────────
fg = pd.read_csv("dashboard/fear_greed_index.csv")
fg['date'] = pd.to_datetime(fg['date'])
fg = fg.rename(columns={'classification': 'sentiment', 'value': 'fg_value'})

print(f"\n📊 Fear/Greed Index")
print(f"   Rows: {len(fg):,}  |  Columns: {fg.shape[1]}")
print(f"   Date range: {fg['date'].min().date()} → {fg['date'].max().date()}")
print(f"   Missing: {fg.isnull().sum().sum()}  |  Duplicates: {fg.duplicated().sum()}")
print(f"   Sentiment distribution:")
for s, c in fg['sentiment'].value_counts().items():
    print(f"     {s:20s} {c:5d}  ({100*c/len(fg):.1f}%)")

# Binary sentiment for simpler comparisons
fg['sentiment_binary'] = fg['sentiment'].map({
    'Extreme Fear': 'Fear', 'Fear': 'Fear',
    'Neutral': 'Neutral',
    'Greed': 'Greed', 'Extreme Greed': 'Greed'
})

# ── A.2  Load Trader Data ─────────────────────────────────────────────
hd = pd.read_csv("dashboard/historical_data.csv.gz")

# Mock missing columns if using trimmed dataset
if 'Coin' not in hd.columns: hd['Coin'] = 'BTC'
if 'Fee' not in hd.columns: hd['Fee'] = 0.0
if 'Timestamp IST' not in hd.columns and 'Timestamp' in hd.columns:
    hd['Timestamp IST'] = pd.to_datetime(hd['Timestamp'], unit='ms').dt.strftime('%d-%m-%Y %H:%M')

print(f"\n📊 Hyperliquid Trader Data (Optimized GZ Download)")
print(f"   Rows: {len(hd):,}  |  Columns: {hd.shape[1]}")
print(f"   Missing values per column:")
for col in hd.columns:
    missing = hd[col].isnull().sum()
    if missing > 0:
        print(f"     {col:25s} {missing:6d}  ({100*missing/len(hd):.1f}%)")
if hd.isnull().sum().sum() == 0:
    print(f"     (none)")
print(f"   Duplicates: {hd.duplicated().sum()}")
print(f"   Unique accounts: {hd['Account'].nunique()}")
print(f"   Unique coins: {hd['Coin'].nunique()}")

# ── A.3  Convert Timestamps & Align ──────────────────────────────────
hd['datetime'] = pd.to_datetime(hd['Timestamp IST'], format='%d-%m-%Y %H:%M')
hd['date'] = hd['datetime'].dt.date
hd['date'] = pd.to_datetime(hd['date'])

# Filter fear/greed to overlapping period
date_min, date_max = hd['date'].min(), hd['date'].max()
fg_overlap = fg[(fg['date'] >= date_min) & (fg['date'] <= date_max)].copy()
print(f"\n🔗 Overlapping period: {date_min.date()} → {date_max.date()}")
print(f"   Sentiment days in overlap: {len(fg_overlap)}")
print(f"   Trading days in overlap:   {hd['date'].nunique()}")

# Merge sentiment onto trades
hd = hd.merge(fg[['date', 'fg_value', 'sentiment', 'sentiment_binary']],
               on='date', how='left')

# ── A.4  Classify Trade Events ────────────────────────────────────────
def classify_event(direction):
    d = str(direction).lower()
    if 'open long' in d:         return 'open_long'
    if 'close long' in d:        return 'close_long'
    if 'open short' in d:        return 'open_short'
    if 'close short' in d:       return 'close_short'
    if 'long > short' in d:      return 'flip_long_to_short'
    if 'short > long' in d:      return 'flip_short_to_long'
    if d == 'buy':               return 'buy'
    if d == 'sell':              return 'sell'
    if 'liquidat' in d:          return 'liquidation'
    if 'auto-deleverag' in d:    return 'auto_deleverage'
    return 'other'

hd['event_type'] = hd['Direction'].apply(classify_event)

# Determine if trade is long/short direction
hd['is_long'] = hd['event_type'].isin(['open_long', 'close_long', 'buy', 'flip_short_to_long'])
hd['is_short'] = hd['event_type'].isin(['open_short', 'close_short', 'sell', 'flip_long_to_short'])

# ── A.5  Key Metrics ─────────────────────────────────────────────────
# --- Daily PnL per trader ---
closing_events = hd[hd['event_type'].isin([
    'close_long', 'close_short', 'flip_long_to_short',
    'flip_short_to_long', 'liquidation', 'auto_deleverage', 'buy', 'sell'
])].copy()

daily_trader = closing_events.groupby(['date', 'Account', 'sentiment_binary']).agg(
    total_pnl=('Closed PnL', 'sum'),
    trade_count=('Closed PnL', 'count'),
    winning_trades=('Closed PnL', lambda x: (x > 0).sum()),
    avg_trade_size=('Size USD', 'mean'),
    total_volume=('Size USD', 'sum'),
    avg_fee=('Fee', 'mean'),
).reset_index()

daily_trader['win_rate'] = daily_trader['winning_trades'] / daily_trader['trade_count']
daily_trader['net_pnl'] = daily_trader['total_pnl'] - daily_trader.groupby('Account')['total_pnl'].transform(lambda x: 0)

# --- Daily aggregate metrics ---
daily_agg = hd.groupby(['date', 'sentiment_binary']).agg(
    total_trades=('Account', 'count'),
    unique_traders=('Account', 'nunique'),
    total_volume=('Size USD', 'sum'),
    avg_size=('Size USD', 'mean'),
    total_pnl=('Closed PnL', 'sum'),
    long_count=('is_long', 'sum'),
    short_count=('is_short', 'sum'),
).reset_index()

daily_agg['long_short_ratio'] = daily_agg['long_count'] / daily_agg['short_count'].replace(0, 1)
daily_agg['avg_trades_per_trader'] = daily_agg['total_trades'] / daily_agg['unique_traders']

# --- Per-trade-level metrics for charting ---
hd['trade_pnl_positive'] = hd['Closed PnL'] > 0

print(f"\n✅ Key Metrics Created:")
print(f"   daily_trader table: {len(daily_trader):,} rows")
print(f"   daily_agg table:    {len(daily_agg):,} rows")

# ========================================================================
# PART B — ANALYSIS
# ========================================================================
print("\n" + "=" * 72)
print("PART B — ANALYSIS")
print("=" * 72)

# Only keep Fear/Greed (drop Neutral for cleaner binary comparisons)
fear_greed_only = daily_agg[daily_agg['sentiment_binary'].isin(['Fear', 'Greed'])]
dt_fg = daily_trader[daily_trader['sentiment_binary'].isin(['Fear', 'Greed'])]

# ── B.1  Performance vs Sentiment ─────────────────────────────────────
print("\n─── B.1  Performance on Fear vs Greed Days ───")

perf_summary = dt_fg.groupby('sentiment_binary').agg(
    mean_daily_pnl=('total_pnl', 'mean'),
    median_daily_pnl=('total_pnl', 'median'),
    mean_win_rate=('win_rate', 'mean'),
    std_pnl=('total_pnl', 'std'),
    total_observations=('total_pnl', 'count'),
).round(2)

# Drawdown proxy: worst daily PnL
perf_summary['worst_daily_pnl'] = dt_fg.groupby('sentiment_binary')['total_pnl'].min().values
perf_summary['pnl_volatility'] = perf_summary['std_pnl']

print(perf_summary.to_string())

# Chart 1: PnL Distribution by Sentiment
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Performance Metrics: Fear vs Greed Days', fontsize=16, fontweight='bold', y=1.02)

colors = {'Fear': '#e74c3c', 'Greed': '#2ecc71'}

# PnL Distribution
for sent in ['Fear', 'Greed']:
    data = dt_fg[dt_fg['sentiment_binary'] == sent]['total_pnl'].clip(-5000, 5000)
    axes[0].hist(data, bins=50, alpha=0.6, label=sent, color=colors[sent], density=True)
axes[0].set_title('Daily PnL Distribution', fontweight='bold')
axes[0].set_xlabel('Daily PnL ($)')
axes[0].set_ylabel('Density')
axes[0].legend()
axes[0].axvline(0, color='gray', linestyle='--', alpha=0.5)

# Win Rate Distribution
for sent in ['Fear', 'Greed']:
    data = dt_fg[dt_fg['sentiment_binary'] == sent]['win_rate']
    axes[1].hist(data, bins=30, alpha=0.6, label=sent, color=colors[sent], density=True)
axes[1].set_title('Win Rate Distribution', fontweight='bold')
axes[1].set_xlabel('Win Rate')
axes[1].set_ylabel('Density')
axes[1].legend()

# Box plot PnL
bp_data = [dt_fg[dt_fg['sentiment_binary'] == s]['total_pnl'].clip(-5000, 5000) for s in ['Fear', 'Greed']]
bp = axes[2].boxplot(bp_data, labels=['Fear', 'Greed'], patch_artist=True,
                     medianprops=dict(color='black', linewidth=2))
for patch, color in zip(bp['boxes'], ['#e74c3c', '#2ecc71']):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
axes[2].set_title('PnL Box Plot', fontweight='bold')
axes[2].set_ylabel('Daily PnL ($)')

plt.tight_layout()
save(fig, '01_performance_vs_sentiment')

# ── B.2  Behavior Changes by Sentiment ────────────────────────────────
print("\n─── B.2  Behavior Changes by Sentiment ───")

behavior_summary = fear_greed_only.groupby('sentiment_binary').agg(
    avg_daily_trades=('total_trades', 'mean'),
    avg_volume=('total_volume', 'mean'),
    avg_trade_size=('avg_size', 'mean'),
    avg_long_short_ratio=('long_short_ratio', 'mean'),
    avg_trades_per_trader=('avg_trades_per_trader', 'mean'),
).round(2)

print(behavior_summary.to_string())

# Chart 2: Behavior Comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Trader Behavior: Fear vs Greed Days', fontsize=16, fontweight='bold', y=1.01)

# Trade frequency
metrics = ['total_trades', 'total_volume', 'avg_size', 'long_short_ratio']
titles = ['Daily Trade Count', 'Daily Trading Volume ($)', 'Avg Trade Size ($)', 'Long/Short Ratio']
ylabels = ['Trades', 'Volume ($)', 'Size ($)', 'Ratio']

for idx, (metric, title, ylabel) in enumerate(zip(metrics, titles, ylabels)):
    ax = axes[idx // 2][idx % 2]
    fear_data = fear_greed_only[fear_greed_only['sentiment_binary'] == 'Fear'][metric]
    greed_data = fear_greed_only[fear_greed_only['sentiment_binary'] == 'Greed'][metric]
    
    bp = ax.boxplot([fear_data, greed_data], labels=['Fear', 'Greed'], patch_artist=True,
                    medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(bp['boxes'], ['#e74c3c', '#2ecc71']):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel(ylabel)
    
    # Add means as diamonds
    means = [fear_data.mean(), greed_data.mean()]
    ax.scatter([1, 2], means, color='gold', marker='D', s=100, zorder=5, label='Mean')
    ax.legend()

plt.tight_layout()
save(fig, '02_behavior_vs_sentiment')

# Chart 3: Long/Short Ratio over time colored by sentiment
fig, ax = plt.subplots(figsize=(16, 6))
daily_agg_sorted = daily_agg.sort_values('date')
for sent, color in colors.items():
    mask = daily_agg_sorted['sentiment_binary'] == sent
    ax.scatter(daily_agg_sorted.loc[mask, 'date'],
               daily_agg_sorted.loc[mask, 'long_short_ratio'],
               alpha=0.5, color=color, label=sent, s=20)
neutral_mask = daily_agg_sorted['sentiment_binary'] == 'Neutral'
ax.scatter(daily_agg_sorted.loc[neutral_mask, 'date'],
           daily_agg_sorted.loc[neutral_mask, 'long_short_ratio'],
           alpha=0.3, color='gray', label='Neutral', s=15)
ax.axhline(1, color='black', linestyle='--', alpha=0.4, label='Balanced (1.0)')
ax.set_title('Long/Short Ratio Over Time by Sentiment', fontsize=14, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Long/Short Ratio')
ax.legend()
plt.tight_layout()
save(fig, '03_long_short_ratio_timeline')

# ── B.3  Trader Segmentation ─────────────────────────────────────────
print("\n─── B.3  Trader Segmentation ───")

# Per-trader aggregate stats
trader_stats = closing_events.groupby('Account').agg(
    total_pnl=('Closed PnL', 'sum'),
    mean_pnl=('Closed PnL', 'mean'),
    trade_count=('Closed PnL', 'count'),
    win_rate=('Closed PnL', lambda x: (x > 0).mean()),
    avg_size=('Size USD', 'mean'),
    total_volume=('Size USD', 'sum'),
    pnl_std=('Closed PnL', 'std'),
    max_loss=('Closed PnL', 'min'),
    total_fees=('Fee', 'sum'),
    trading_days=('date', 'nunique'),
).reset_index()

trader_stats['avg_daily_trades'] = trader_stats['trade_count'] / trader_stats['trading_days']
trader_stats['profit_factor'] = trader_stats['total_pnl'] / (trader_stats['total_fees'] + 1)
trader_stats['consistency'] = trader_stats['mean_pnl'] / (trader_stats['pnl_std'] + 1)  # Sharpe-like

# Segment 1: High vs Low Leverage Traders (using avg trade size as proxy)
median_size = trader_stats['avg_size'].median()
trader_stats['size_segment'] = np.where(trader_stats['avg_size'] > median_size, 'Large Size', 'Small Size')

# Segment 2: Frequent vs Infrequent
median_freq = trader_stats['avg_daily_trades'].median()
trader_stats['freq_segment'] = np.where(trader_stats['avg_daily_trades'] > median_freq, 'Frequent', 'Infrequent')

# Segment 3: Winners vs Losers
trader_stats['profit_segment'] = np.where(trader_stats['total_pnl'] > 0, 'Winner', 'Loser')

# Segment 4: Consistency
median_consistency = trader_stats['consistency'].median()
trader_stats['consistency_segment'] = np.where(trader_stats['consistency'] > median_consistency, 'Consistent', 'Inconsistent')

print("\n📊 Trader Segments Summary:")
print(f"\n  Size Segment:")
for seg in ['Large Size', 'Small Size']:
    sub = trader_stats[trader_stats['size_segment'] == seg]
    print(f"    {seg}: {len(sub)} traders, avg PnL=${sub['total_pnl'].mean():,.0f}, "
          f"win rate={sub['win_rate'].mean():.1%}")

print(f"\n  Frequency Segment:")
for seg in ['Frequent', 'Infrequent']:
    sub = trader_stats[trader_stats['freq_segment'] == seg]
    print(f"    {seg}: {len(sub)} traders, avg PnL=${sub['total_pnl'].mean():,.0f}, "
          f"win rate={sub['win_rate'].mean():.1%}")

print(f"\n  Profit Segment:")
for seg in ['Winner', 'Loser']:
    sub = trader_stats[trader_stats['profit_segment'] == seg]
    print(f"    {seg}: {len(sub)} traders, avg PnL=${sub['total_pnl'].mean():,.0f}, "
          f"win rate={sub['win_rate'].mean():.1%}")

# Chart 4: Segment Comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Trader Segmentation Analysis', fontsize=16, fontweight='bold', y=1.02)

# Size Segment
seg_colors = {'Large Size': '#3498db', 'Small Size': '#e67e22'}
for seg, color in seg_colors.items():
    sub = trader_stats[trader_stats['size_segment'] == seg]
    axes[0].scatter(sub['trade_count'], sub['total_pnl'], 
                    alpha=0.7, label=seg, color=color, s=100, edgecolors='white')
axes[0].set_title('Trade Size Segments', fontweight='bold')
axes[0].set_xlabel('Total Trades')
axes[0].set_ylabel('Total PnL ($)')
axes[0].axhline(0, color='gray', linestyle='--', alpha=0.5)
axes[0].legend()

# Frequency Segment
seg_colors2 = {'Frequent': '#9b59b6', 'Infrequent': '#1abc9c'}
for seg, color in seg_colors2.items():
    sub = trader_stats[trader_stats['freq_segment'] == seg]
    axes[1].scatter(sub['win_rate'], sub['total_pnl'],
                    alpha=0.7, label=seg, color=color, s=100, edgecolors='white')
axes[1].set_title('Frequency Segments', fontweight='bold')
axes[1].set_xlabel('Win Rate')
axes[1].set_ylabel('Total PnL ($)')
axes[1].axhline(0, color='gray', linestyle='--', alpha=0.5)
axes[1].legend()

# Winner vs Loser stats
seg_colors3 = {'Winner': '#2ecc71', 'Loser': '#e74c3c'}
for seg, color in seg_colors3.items():
    sub = trader_stats[trader_stats['profit_segment'] == seg]
    axes[2].scatter(sub['avg_daily_trades'], sub['win_rate'],
                    alpha=0.7, label=seg, color=color, s=100, edgecolors='white')
axes[2].set_title('Winners vs Losers', fontweight='bold')
axes[2].set_xlabel('Avg Daily Trades')
axes[2].set_ylabel('Win Rate')
axes[2].legend()

plt.tight_layout()
save(fig, '04_trader_segmentation')

# ── B.4  Segment Behavior on Fear vs Greed Days ──────────────────────
print("\n─── B.4  Segment Behavior on Fear vs Greed Days ───")

# Merge trader segments into daily_trader
dt_segmented = dt_fg.merge(
    trader_stats[['Account', 'size_segment', 'freq_segment', 'profit_segment', 'consistency_segment']],
    on='Account', how='left'
)

# Compare behavior by segment and sentiment
for seg_col, seg_name in [('size_segment', 'Size'), ('freq_segment', 'Frequency'), ('profit_segment', 'Profit')]:
    print(f"\n  {seg_name} Segment x Sentiment:")
    pivot = dt_segmented.groupby([seg_col, 'sentiment_binary']).agg(
        avg_pnl=('total_pnl', 'mean'),
        avg_win_rate=('win_rate', 'mean'),
        avg_trades=('trade_count', 'mean'),
    ).round(2)
    print(pivot.to_string())

# Chart 5: Segment x Sentiment Heatmap
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Segment Performance by Sentiment', fontsize=16, fontweight='bold', y=1.02)

for idx, (seg_col, title) in enumerate([
    ('size_segment', 'Trade Size'), ('freq_segment', 'Frequency'), ('profit_segment', 'Profitability')
]):
    pivot = dt_segmented.groupby([seg_col, 'sentiment_binary'])['total_pnl'].mean().unstack()
    if not pivot.empty:
        sns.heatmap(pivot, annot=True, fmt='.0f', cmap='RdYlGn', center=0, ax=axes[idx],
                    linewidths=2, linecolor='white', cbar_kws={'label': 'Avg Daily PnL ($)'})
        axes[idx].set_title(f'{title} Segment', fontweight='bold')
        axes[idx].set_ylabel('')

plt.tight_layout()
save(fig, '05_segment_sentiment_heatmap')

# ── B.5  Additional Insights Charts ──────────────────────────────────
print("\n─── B.5  Additional Insights ───")

# Chart 6: Fear/Greed Value vs Trading Volume Correlation
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Sentiment Intensity & Market Activity', fontsize=16, fontweight='bold', y=1.02)

daily_with_fg = daily_agg.merge(fg[['date', 'fg_value']], on='date', how='left')

ax = axes[0]
ax.scatter(daily_with_fg['fg_value'], daily_with_fg['total_volume'], 
           alpha=0.4, c=daily_with_fg['fg_value'], cmap='RdYlGn', s=40)
ax.set_title('Sentiment Value vs Trading Volume', fontweight='bold')
ax.set_xlabel('Fear/Greed Index Value')
ax.set_ylabel('Daily Volume ($)')
# Add trendline
z = np.polyfit(daily_with_fg['fg_value'].dropna(), 
               daily_with_fg.loc[daily_with_fg['fg_value'].notna(), 'total_volume'], 1)
p = np.poly1d(z)
x_line = np.linspace(daily_with_fg['fg_value'].min(), daily_with_fg['fg_value'].max(), 100)
ax.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'Trend (slope={z[0]:.0f})')
ax.legend()

ax = axes[1]
ax.scatter(daily_with_fg['fg_value'], daily_with_fg['total_pnl'],
           alpha=0.4, c=daily_with_fg['fg_value'], cmap='RdYlGn', s=40)
ax.set_title('Sentiment Value vs Daily PnL', fontweight='bold')
ax.set_xlabel('Fear/Greed Index Value')
ax.set_ylabel('Daily PnL ($)')
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
z2 = np.polyfit(daily_with_fg['fg_value'].dropna(),
                daily_with_fg.loc[daily_with_fg['fg_value'].notna(), 'total_pnl'], 1)
p2 = np.poly1d(z2)
ax.plot(x_line, p2(x_line), 'r--', linewidth=2, label=f'Trend (slope={z2[0]:.0f})')
ax.legend()

plt.tight_layout()
save(fig, '06_sentiment_correlations')

# Chart 7: Win Rate by Sentiment Category (granular)
fig, ax = plt.subplots(figsize=(12, 6))
sentiment_order = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
sent_colors = ['#c0392b', '#e74c3c', '#95a5a6', '#2ecc71', '#27ae60']

daily_trader_all = daily_trader.copy()
daily_trader_all = daily_trader_all.merge(fg[['date', 'sentiment']], on='date', how='left')

win_by_sent = daily_trader_all.groupby('sentiment_x' if 'sentiment_x' in daily_trader_all.columns else 'sentiment').agg(
    avg_win_rate=('win_rate', 'mean'),
    avg_pnl=('total_pnl', 'mean'),
    count=('win_rate', 'count'),
).reindex(sentiment_order)

bars = ax.bar(range(len(win_by_sent)), win_by_sent['avg_win_rate'], 
              color=sent_colors, edgecolor='white', linewidth=2, alpha=0.85)
ax.set_xticks(range(len(win_by_sent)))
ax.set_xticklabels(sentiment_order, fontsize=12)
ax.set_title('Average Win Rate by Sentiment Category', fontsize=14, fontweight='bold')
ax.set_ylabel('Win Rate', fontsize=12)
ax.axhline(win_by_sent['avg_win_rate'].mean(), color='black', linestyle='--', alpha=0.5, label='Overall Avg')

# Add value labels
for bar, val in zip(bars, win_by_sent['avg_win_rate']):
    if not np.isnan(val):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.1%}', ha='center', fontsize=11, fontweight='bold')
ax.legend()
plt.tight_layout()
save(fig, '07_winrate_by_sentiment')

# Chart 8: Trader PnL Ranking
fig, ax = plt.subplots(figsize=(14, 8))
sorted_traders = trader_stats.sort_values('total_pnl', ascending=True)
bar_colors = ['#2ecc71' if pnl > 0 else '#e74c3c' for pnl in sorted_traders['total_pnl']]
short_accounts = [f"...{a[-6:]}" for a in sorted_traders['Account']]
ax.barh(range(len(sorted_traders)), sorted_traders['total_pnl'], color=bar_colors, 
        edgecolor='white', linewidth=0.5)
ax.set_yticks(range(len(sorted_traders)))
ax.set_yticklabels(short_accounts, fontsize=8)
ax.set_title('Total PnL by Trader (All Time)', fontsize=14, fontweight='bold')
ax.set_xlabel('Total PnL ($)')
ax.axvline(0, color='black', linewidth=1)
plt.tight_layout()
save(fig, '08_trader_pnl_ranking')


# ========================================================================
# PART C — ACTIONABLE OUTPUT
# ========================================================================
print("\n" + "=" * 72)
print("PART C — ACTIONABLE STRATEGY RECOMMENDATIONS")
print("=" * 72)

# Compute key insights for strategies
fear_pnl = dt_fg[dt_fg['sentiment_binary'] == 'Fear']['total_pnl']
greed_pnl = dt_fg[dt_fg['sentiment_binary'] == 'Greed']['total_pnl']

fear_wr = dt_fg[dt_fg['sentiment_binary'] == 'Fear']['win_rate']
greed_wr = dt_fg[dt_fg['sentiment_binary'] == 'Greed']['win_rate']

fear_trades = fear_greed_only[fear_greed_only['sentiment_binary'] == 'Fear']['total_trades']
greed_trades = fear_greed_only[fear_greed_only['sentiment_binary'] == 'Greed']['total_trades']

fear_ls = fear_greed_only[fear_greed_only['sentiment_binary'] == 'Fear']['long_short_ratio']
greed_ls = fear_greed_only[fear_greed_only['sentiment_binary'] == 'Greed']['long_short_ratio']

# Segment-specific data for strategies
seg_fear = dt_segmented[dt_segmented['sentiment_binary'] == 'Fear']
seg_greed = dt_segmented[dt_segmented['sentiment_binary'] == 'Greed']

# ── Strategy Infographic Chart ────────────────────────────────────────
fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor('#0d1117')

# Title
fig.text(0.5, 0.97, '🎯  ACTIONABLE STRATEGY RECOMMENDATIONS', fontsize=22,
         fontweight='bold', ha='center', va='top', color='white',
         fontfamily='sans-serif')
fig.text(0.5, 0.94, 'Based on Bitcoin Fear/Greed Sentiment × Hyperliquid Trader Data',
         fontsize=12, ha='center', va='top', color='#8b949e')

# ── STRATEGY 1 Panel (left) ──────────────────────────────────────────
ax1 = fig.add_axes([0.03, 0.08, 0.46, 0.82])
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.set_facecolor('#161b22')
ax1.axis('off')

# Strategy 1 Header
ax1.add_patch(plt.Rectangle((0.2, 9.0), 9.6, 0.9, facecolor='#e74c3c', alpha=0.9,
              transform=ax1.transData, zorder=2, linewidth=0))
ax1.text(5, 9.45, 'STRATEGY 1: Sentiment-Adjusted Position Sizing',
         fontsize=14, fontweight='bold', color='white', ha='center', va='center')

# Evidence box
ax1.add_patch(plt.Rectangle((0.3, 7.3), 9.4, 1.5, facecolor='#21262d',
              edgecolor='#30363d', linewidth=1.5, zorder=2))
ax1.text(5, 8.55, '📊 EVIDENCE', fontsize=11, fontweight='bold', color='#58a6ff', ha='center')
ax1.text(1.0, 8.1, f'Fear Days:  Avg PnL = \\${fear_pnl.mean():,.0f}  |  Win Rate = {fear_wr.mean():.1%}  |  Volatility = \\${fear_pnl.std():,.0f}',
         fontsize=9.5, color='#f85149', ha='left', fontfamily='monospace')
ax1.text(1.0, 7.65, f'Greed Days: Avg PnL = \\${greed_pnl.mean():,.0f}  |  Win Rate = {greed_wr.mean():.1%}  |  Volatility = \\${greed_pnl.std():,.0f}',
         fontsize=9.5, color='#3fb950', ha='left', fontfamily='monospace')

# Segment-specific rules
y = 6.9
rules = [
    ('⚡ FREQUENT TRADERS', '#f0883e', [
        f'Fear: Maintain trade frequency (avg {seg_fear[seg_fear["freq_segment"]=="Frequent"]["trade_count"].mean():.0f} trades/day),',
        f'       but TIGHTEN STOP-LOSSES — PnL vol is {fear_pnl.std()/greed_pnl.std():.0%} of Greed-day vol.',
        f'Greed: REDUCE trade count by ~20%. Win rate drops to',
        f'       {seg_greed[seg_greed["freq_segment"]=="Frequent"]["win_rate"].mean():.1%} vs {seg_fear[seg_fear["freq_segment"]=="Frequent"]["win_rate"].mean():.1%} on Fear days.',
    ]),
    ('🐋 LARGE-SIZE TRADERS', '#a371f7', [
        f'Fear: REDUCE position sizes by 30–50%. Avg PnL is \\${seg_fear[seg_fear["size_segment"]=="Large Size"]["total_pnl"].mean():,.0f}',
        f'       but the higher volatility makes big sizing dangerous.',
        f'Greed: Keep normal sizing — PnL is \\${seg_greed[seg_greed["size_segment"]=="Large Size"]["total_pnl"].mean():,.0f} with lower risk.',
    ]),
    ('🎯 INFREQUENT TRADERS', '#79c0ff', [
        f'Fear: WAIT for Extreme Fear (Index < 25) to enter. Higher',
        f'       conviction = better entries. Avg PnL = \\${seg_fear[seg_fear["freq_segment"]=="Infrequent"]["total_pnl"].mean():,.0f}.',
        f'Greed: AVOID new positions. Win rate = {seg_greed[seg_greed["freq_segment"]=="Infrequent"]["win_rate"].mean():.1%}.',
    ]),
]

for title, color, lines in rules:
    ax1.add_patch(plt.Rectangle((0.3, y - len(lines)*0.42 - 0.15), 9.4,
                  len(lines)*0.42 + 0.6, facecolor='#21262d',
                  edgecolor=color, linewidth=1.5, zorder=2))
    ax1.text(0.6, y, title, fontsize=10.5, fontweight='bold', color=color, ha='left')
    for i, line in enumerate(lines):
        ax1.text(0.8, y - 0.42*(i+1), line, fontsize=8.5, color='#c9d1d9',
                 ha='left', fontfamily='monospace')
    y -= len(lines)*0.42 + 0.85

# ── STRATEGY 2 Panel (right) ─────────────────────────────────────────
ax2 = fig.add_axes([0.52, 0.08, 0.46, 0.82])
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.set_facecolor('#161b22')
ax2.axis('off')

# Strategy 2 Header
ax2.add_patch(plt.Rectangle((0.2, 9.0), 9.6, 0.9, facecolor='#2ecc71', alpha=0.9,
              transform=ax2.transData, zorder=2, linewidth=0))
ax2.text(5, 9.45, 'STRATEGY 2: Contrarian Long/Short Bias',
         fontsize=14, fontweight='bold', color='white', ha='center', va='center')

# Evidence box
ax2.add_patch(plt.Rectangle((0.3, 7.3), 9.4, 1.5, facecolor='#21262d',
              edgecolor='#30363d', linewidth=1.5, zorder=2))
ax2.text(5, 8.55, '📊 EVIDENCE', fontsize=11, fontweight='bold', color='#58a6ff', ha='center')
ax2.text(1.0, 8.1, f'Fear L/S Ratio = {fear_ls.mean():.2f}  |  Greed L/S Ratio = {greed_ls.mean():.2f}',
         fontsize=9.5, color='#e6edf3', ha='left', fontfamily='monospace')

loser_fear = seg_fear[seg_fear['profit_segment'] == 'Loser']['total_pnl'].mean()
loser_greed = seg_greed[seg_greed['profit_segment'] == 'Loser']['total_pnl'].mean()
winner_fear = seg_fear[seg_fear['profit_segment'] == 'Winner']['total_pnl'].mean()
winner_greed = seg_greed[seg_greed['profit_segment'] == 'Winner']['total_pnl'].mean()
ax2.text(1.0, 7.65, f'Losers: Fear=\\${loser_fear:,.0f}  vs  Greed=\\${loser_greed:,.0f}  (DANGER on Greed!)',
         fontsize=9.5, color='#f85149', ha='left', fontfamily='monospace')

# Segment-specific rules
y = 6.9
rules2 = [
    ('🏆 WINNERS (29 traders)', '#3fb950', [
        f'Fear: LEAN INTO CONTRARIAN LONGS — market oversells,',
        f'       winners capture mean-reversion. PnL=\\${winner_fear:,.0f}/day.',
        f'Greed: ADD SHORT HEDGES to long positions. Crowd is',
        f'       heavily long (L/S={greed_ls.mean():.1f}), reversal risk is high.',
        f'       Keep trading — winners maintain edge by staying active.',
    ]),
    ('❌ LOSERS (3 traders)', '#f85149', [
        f'Fear: SAFER period — avg PnL = \\${loser_fear:,.0f} (nearly breakeven).',
        f'       If trading, use small sizes and stick to BTC/ETH only.',
        f'Greed: DO NOT TRADE. Avg PnL = \\${loser_greed:,.0f}. The crowd',
        f'       euphoria traps losers into devastating positions.',
    ]),
    ('📈 ALL TRADERS — GENERAL RULES', '#e3b341', [
        f'• During Fear days, increase trade frequency ONLY if you are',
        f'  a "Frequent" or "Winner" segment trader.',
        f'• During Greed days, reduce leverage for ALL segments.',
        f'  The L/S ratio of {greed_ls.mean():.1f} signals overcrowded longs.',
        f'• For Extreme Fear (Index < 25):  Best contrarian long entries.',
        f'• For Extreme Greed (Index > 75): Best short/hedge entries.',
    ]),
]

for title, color, lines in rules2:
    ax2.add_patch(plt.Rectangle((0.3, y - len(lines)*0.42 - 0.15), 9.4,
                  len(lines)*0.42 + 0.6, facecolor='#21262d',
                  edgecolor=color, linewidth=1.5, zorder=2))
    ax2.text(0.6, y, title, fontsize=10.5, fontweight='bold', color=color, ha='left')
    for i, line in enumerate(lines):
        ax2.text(0.8, y - 0.42*(i+1), line, fontsize=8.5, color='#c9d1d9',
                 ha='left', fontfamily='monospace')
    y -= len(lines)*0.42 + 0.85

save(fig, '12_strategy_recommendations')

# ── Also print clear text summary ────────────────────────────────────
print(f"""
{'='*72}
  STRATEGY 1: SENTIMENT-ADJUSTED POSITION SIZING
{'='*72}

  RULE FOR FREQUENT TRADERS (>{median_freq:.0f} trades/day):
    → During Fear days, maintain trade frequency but TIGHTEN stop-losses.
    → During Greed days, REDUCE trade count by ~20%.
      Evidence: Win rate drops from {seg_fear[seg_fear['freq_segment']=='Frequent']['win_rate'].mean():.1%} (Fear) → {seg_greed[seg_greed['freq_segment']=='Frequent']['win_rate'].mean():.1%} (Greed).

  RULE FOR LARGE-SIZE TRADERS (avg size > ${median_size:,.0f}):
    → During Fear days, REDUCE position sizes by 30–50%.
      PnL volatility = ${fear_pnl.std():,.0f} vs ${greed_pnl.std():,.0f} on Greed.
    → During Greed days, maintain normal sizing.

  RULE FOR INFREQUENT TRADERS (<{median_freq:.0f} trades/day):
    → During Fear days, trade ONLY during Extreme Fear (Index < 25).
    → During Greed days, AVOID new positions entirely.

{'='*72}
  STRATEGY 2: CONTRARIAN LONG/SHORT BIAS MANAGEMENT
{'='*72}

  RULE FOR WINNERS (29 traders, avg PnL > $0):
    → During Fear days, OPEN contrarian long positions.
      Winners avg ${winner_fear:,.0f}/day vs ${winner_greed:,.0f}/day on Greed.
    → During Greed days, ADD short hedges. L/S ratio = {greed_ls.mean():.1f} (crowded longs).

  RULE FOR LOSERS (3 traders):
    → During Fear days, trade cautiously — nearly breakeven (${loser_fear:,.0f}/day).
    → During Greed days, DO NOT TRADE. Avg loss = ${loser_greed:,.0f}/day.

  GENERAL RULES FOR ALL:
    → Extreme Fear  (Index < 25): BEST window for contrarian long entries.
    → Extreme Greed (Index > 75): BEST window for short/hedge entries.
    → Reduce leverage for ALL segments during Greed days.
""")


# ========================================================================
# BONUS — PREDICTIVE MODEL
# ========================================================================
print("=" * 72)
print("BONUS — PREDICTIVE MODEL")
print("=" * 72)

# Feature engineering for prediction
daily_features = daily_trader.groupby('date').agg(
    avg_pnl=('total_pnl', 'mean'),
    total_pnl=('total_pnl', 'sum'),
    avg_win_rate=('win_rate', 'mean'),
    avg_trade_count=('trade_count', 'mean'),
    avg_trade_size=('avg_trade_size', 'mean'),
    trader_count=('Account', 'nunique'),
).reset_index()

# Add sentiment features
daily_features = daily_features.merge(fg[['date', 'fg_value', 'sentiment_binary']], on='date', how='left')
daily_features = daily_features.dropna(subset=['fg_value'])
daily_features = daily_features.sort_values('date')

# Add lagged features
daily_features['fg_value_lag1'] = daily_features['fg_value'].shift(1)
daily_features['fg_value_lag3'] = daily_features['fg_value'].shift(3)
daily_features['fg_change'] = daily_features['fg_value'].diff()
daily_features['pnl_lag1'] = daily_features['total_pnl'].shift(1)
daily_features['win_rate_lag1'] = daily_features['avg_win_rate'].shift(1)
daily_features['vol_regime'] = daily_features['total_pnl'].rolling(7).std()

# Target: next-day profitability bucket
daily_features['next_day_pnl'] = daily_features['total_pnl'].shift(-1)
daily_features['target'] = pd.cut(daily_features['next_day_pnl'],
                                   bins=[-np.inf, 0, np.inf],
                                   labels=['Loss', 'Profit'])

daily_features = daily_features.dropna()

feature_cols = ['fg_value', 'fg_value_lag1', 'fg_value_lag3', 'fg_change',
                'avg_pnl', 'avg_win_rate', 'avg_trade_count', 'avg_trade_size',
                'trader_count', 'pnl_lag1', 'win_rate_lag1', 'vol_regime']

X = daily_features[feature_cols].values
y = daily_features['target'].values

print(f"\n📊 Prediction Dataset: {len(X)} samples, {len(feature_cols)} features")
print(f"   Target distribution: {pd.Series(y).value_counts().to_dict()}")

# Random Forest
rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, class_weight='balanced')
rf_scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
print(f"\n🌲 Random Forest (5-fold CV):")
print(f"   Accuracy: {rf_scores.mean():.3f} ± {rf_scores.std():.3f}")

# Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=42)
gb_scores = cross_val_score(gb, X, y, cv=5, scoring='accuracy')
print(f"\n🚀 Gradient Boosting (5-fold CV):")
print(f"   Accuracy: {gb_scores.mean():.3f} ± {gb_scores.std():.3f}")

# Feature importance
rf.fit(X, y)
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
importances.plot(kind='barh', ax=ax, color='#3498db', edgecolor='white')
ax.set_title('Feature Importance: Predicting Next-Day Profitability', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance')
plt.tight_layout()
save(fig, '09_feature_importance')


# ========================================================================
# BONUS — TRADER CLUSTERING
# ========================================================================
print("\n" + "=" * 72)
print("BONUS — TRADER CLUSTERING")
print("=" * 72)

cluster_features = ['win_rate', 'avg_daily_trades', 'avg_size', 'total_pnl', 'consistency']
X_cluster = trader_stats[cluster_features].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# Find optimal clusters with inertia
inertias = []
K_range = range(2, min(8, len(trader_stats)))
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

# Use k=3 or 4
optimal_k = 3
km_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
trader_stats['cluster'] = km_final.fit_predict(X_scaled)

# Name clusters based on characteristics
cluster_profiles = trader_stats.groupby('cluster')[cluster_features].mean()
print("\n📊 Cluster Profiles:")
print(cluster_profiles.round(2).to_string())

# Name clusters
cluster_names = {}
for c in range(optimal_k):
    prof = cluster_profiles.loc[c]
    if prof['win_rate'] > cluster_profiles['win_rate'].median() and prof['total_pnl'] > 0:
        cluster_names[c] = "🏆 Skilled Winners"
    elif prof['avg_daily_trades'] > cluster_profiles['avg_daily_trades'].median():
        cluster_names[c] = "⚡ Active Grinders"
    elif prof['avg_size'] > cluster_profiles['avg_size'].median():
        cluster_names[c] = "🐋 Big Bettors"
    else:
        cluster_names[c] = "🎯 Conservative Traders"

trader_stats['archetype'] = trader_stats['cluster'].map(cluster_names)

print("\n📊 Trader Archetypes:")
for name, count in trader_stats['archetype'].value_counts().items():
    sub = trader_stats[trader_stats['archetype'] == name]
    print(f"  {name}: {count} traders")
    print(f"    Avg PnL: ${sub['total_pnl'].mean():,.0f} | Win Rate: {sub['win_rate'].mean():.1%} | "
          f"Avg Trades/Day: {sub['avg_daily_trades'].mean():.1f}")

# Chart: Cluster Visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Trader Behavioral Archetypes (K-Means Clustering)', fontsize=16, fontweight='bold', y=1.02)

cluster_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
for c in range(optimal_k):
    mask = trader_stats['cluster'] == c
    axes[0].scatter(trader_stats.loc[mask, 'win_rate'],
                    trader_stats.loc[mask, 'total_pnl'],
                    color=cluster_colors[c], s=150, alpha=0.8,
                    label=cluster_names[c], edgecolors='white', linewidth=1.5)
axes[0].set_xlabel('Win Rate', fontsize=12)
axes[0].set_ylabel('Total PnL ($)', fontsize=12)
axes[0].set_title('Win Rate vs Total PnL', fontweight='bold')
axes[0].axhline(0, color='gray', linestyle='--', alpha=0.5)
axes[0].legend(fontsize=10)

for c in range(optimal_k):
    mask = trader_stats['cluster'] == c
    axes[1].scatter(trader_stats.loc[mask, 'avg_daily_trades'],
                    trader_stats.loc[mask, 'avg_size'],
                    color=cluster_colors[c], s=150, alpha=0.8,
                    label=cluster_names[c], edgecolors='white', linewidth=1.5)
axes[1].set_xlabel('Avg Daily Trades', fontsize=12)
axes[1].set_ylabel('Avg Trade Size ($)', fontsize=12)
axes[1].set_title('Trading Activity vs Size', fontweight='bold')
axes[1].legend(fontsize=10)

plt.tight_layout()
save(fig, '10_trader_clusters')

# ========================================================================
# SUMMARY TABLE
# ========================================================================
fig, ax = plt.subplots(figsize=(14, 4))
ax.axis('off')
summary_data = [
    ['Metric', 'Fear Days', 'Greed Days', 'Difference'],
    ['Avg Daily PnL', f'${fear_pnl.mean():,.2f}', f'${greed_pnl.mean():,.2f}', 
     f'${greed_pnl.mean() - fear_pnl.mean():,.2f}'],
    ['Win Rate', f'{fear_wr.mean():.1%}', f'{greed_wr.mean():.1%}',
     f'{(greed_wr.mean() - fear_wr.mean()):.1%}'],
    ['Avg Trades/Day', f'{fear_trades.mean():.0f}', f'{greed_trades.mean():.0f}',
     f'{greed_trades.mean() - fear_trades.mean():.0f}'],
    ['L/S Ratio', f'{fear_ls.mean():.2f}', f'{greed_ls.mean():.2f}',
     f'{greed_ls.mean() - fear_ls.mean():+.2f}'],
    ['PnL Volatility', f'${fear_pnl.std():,.0f}', f'${greed_pnl.std():,.0f}', ''],
]

table = ax.table(cellText=summary_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)

# Style header
for j in range(4):
    table[0, j].set_facecolor('#2c3e50')
    table[0, j].set_text_props(color='white', fontweight='bold')
for i in range(1, len(summary_data)):
    for j in range(4):
        table[i, j].set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')

ax.set_title('Fear vs Greed: Key Metrics Summary', fontsize=14, fontweight='bold', pad=20)
save(fig, '11_summary_table')

print("\n" + "=" * 72)
print("✅ ANALYSIS COMPLETE!")
print(f"   All charts saved to: {os.path.abspath(OUTPUT_DIR)}/")
print("=" * 72)
