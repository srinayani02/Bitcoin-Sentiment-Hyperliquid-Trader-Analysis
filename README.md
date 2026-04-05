# [Bitcoin Sentiment × Hyperliquid Trader Analysis](https://github.com/Azura18/Bitcoin-Sentiment-Hyperliquid-Trader-Analysis)
An interactive data analysis project exploring the relationship between Bitcoin Market Sentiment (Fear/Greed Index) and trader behavior/performance on Hyperliquid.

## 🚀 Quick Start
### 1. Install Dependencies
Ensure you have Python 3.10+ installed, then run:
```bash
pip install -r requirements.txt
```
### 2. Run the Core Analysis
Generate mathematical insights, trader segments, and 12 static charts:
```bash
python3 analysis.py
```
*Charts will be saved to the `charts/` directory.*
### 3. Launch the Interactive Dashboard
Explore the data visually through a premium Streamlit interface:
```bash
streamlit run dashboard.py
```

### 4. Direct Reports
- **[Analysis Summary](Analysis_Summary.md)** — One-page results summary.
- **[Jupyter Notebook](Market_Sentiment_Analysis.ipynb)** — Interactive code environment.

## 📊 Project Structure

- `analysis.py`: Main processing script (Data cleaning, Segmentation, Clustering, Prediction).
- `dashboard.py`: Streamlit-based interactive dashboard with 5 specialized tabs.
- `requirements.txt`: List of necessary Python libraries.
- `charts/`: Visual output from the analysis script, including the Strategy Infographic.
- `fear_greed_index.csv`: Daily Bitcoin market sentiment data.
- `historical_data.csv`: Historical trade data from Hyperliquid.

## 💡 Key Strategy Findings
- **Fear Days outperform Greed**: Average PnL of $6,575 vs $5,165.
- **Sentiment-Adjusted Sizing**: Large traders should reduce size by 30-50% during extreme fear due to high volatility.
- **Contrarian Bias**: Winners successfully capture mean-reversion during fear periods.

---
*Created as part of the Market Sentiment Analysis Dashboard task.*
