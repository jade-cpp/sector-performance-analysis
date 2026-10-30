# Sector Performance Analysis (2019–2026)

How did different parts of the U.S. stock market behave through 2020 and 2022?

This project uses Python to analyze the 11 S&P 500 sector funds (SPDR Select Sector ETFs) plus the S&P 500 (SPY), and Tableau to present the results.

**Dashboard:** [add your Tableau Public link here]

## What it does
1. Downloads daily prices from Yahoo Finance using `yfinance`.
2. Calculates, for each fund:
   - daily returns and growth of $100
   - 30-day rolling volatility (annualized by multiplying by √252)
   - drawdown, meaning the drop from the highest price reached so far
3. Computes the correlation of daily returns between every pair of funds.
4. Runs a principal component analysis (PCA) using the singular value decomposition (SVD) to see how much of the movement comes from one shared market factor.
5. Saves tidy CSV files that load directly into Tableau.

## Key findings
- Energy had the deepest drawdown (about -63%) and the biggest volatility spike, in early 2020.
- Technology had the best return per unit of risk (about 1.02, vs about 0.90 for the S&P 500).
- Energy moved the least with the rest of the market (correlation about 0.38 with Technology).
- The first principal component explains about [check your latest output]% of sector return variance, so most sector movement comes from one market-wide factor.

## Limitations
This analysis is descriptive. It uses historical price data only and does not predict future performance or give investment advice.

## Notes
Built with Python (pandas, NumPy, yfinance) and Tableau Public. I used an AI coding assistant to help structure the code, then ran, debugged, and checked it myself.
