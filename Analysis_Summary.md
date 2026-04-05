# Analysis Summary: Bitcoin Sentiment & Trader Behavior

This report explores how Bitcoin market sentiment (Fear/Greed Index) influenced trading behavior on Hyperliquid throughout 2024.

## 🧪 Methodology
1. **Data Integration**: Merged daily Fear/Greed Index values with 211,000+ individual trade records.
2. **Segmentation**: Categorized 32 traders into behavior archetypes (Active Grinders, Skilled Winners, Conservative) based on frequency, size, and PnL consistency.
3. **Comparative Analysis**: Aggregated daily performance metrics to isolate the impact of different sentiment regimes.
4. **Predictive Modeling**: Trained a Random Forest classifier to predict next-day profitability using lagged behavioral and sentiment markers.

## 📈 Key Insights
- **Fear Days are High Edge**: Surprisingly, Fear days showed higher average daily PnL ($6,575) compared to Greed days ($5,165).
- **Activity Skew**: Traders are 2.7x more active during Fear periods, suggesting more reactive behavior or broader opportunity sets.
- **Winner Consistency**: Top-performing traders maintained consistent win rates (82%+) across both Fear and Greed, whereas bottom-tier traders suffered massive drawdowns during Greed days (-$13.6K avg loss).

## 🎯 Strategy Recommendations
- **Strategy 1 (Sizing)**: During **Extreme Fear** (Index < 30), reduce position sizes by 30-50%. The returns are better, but volatility is significantly higher, making over-leveraged bets dangerous.
- **Strategy 2 (Bias)**: Focus on **Contrarian Longs** during Fear days. The data suggests winners often capture mean-reversion edges in oversold markets.
- **Rule of Thumb**: During **Greed days**, prioritize hedging or reducing long exposure, as crowd euphoria often leads to trapped liquidity for less-skilled segments.
