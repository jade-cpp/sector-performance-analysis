"""
Sector performance project: how did parts of the U.S. stock market behave
through 2020 and 2022?

What this script does
  1. Downloads daily prices for the S&P 500 sector funds (SPDR ETFs) from Yahoo Finance.
  2. Computes daily returns, growth of $100, rolling volatility, and drawdowns.
  3. Builds a correlation table and runs a PCA (using SVD from linear algebra).
  4. Saves tidy CSV files that you can load straight into Tableau Public.

Run it on your own computer (it needs internet access):
    pip install yfinance pandas numpy
    python sector_analysis.py

To test the pipeline without internet, use fake data:
    python sector_analysis.py --demo
"""

import argparse
import numpy as np
import pandas as pd

SECTORS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Health Care",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}
BENCHMARK = "SPY"
START = "2019-01-01"
TRADING_DAYS = 252


def load_prices(demo: bool) -> pd.DataFrame:
    """Return a DataFrame of adjusted closing prices, one column per ticker."""
    tickers = list(SECTORS) + [BENCHMARK]
    if demo:
        # Fake random-walk prices so the pipeline can be tested offline.
        rng = np.random.default_rng(0)
        dates = pd.bdate_range(START, "2025-12-31")
        market = rng.normal(0.0004, 0.01, len(dates))
        data = {
            t: 100 * np.exp(np.cumsum(0.7 * market + rng.normal(0.0001, 0.008, len(dates))))
            for t in tickers
        }
        return pd.DataFrame(data, index=dates)

    import yfinance as yf  # imported here so --demo works without it

    prices = yf.download(tickers, start=START, auto_adjust=True, progress=False)["Close"]
    return prices.dropna(how="all")


def build_tidy_table(prices: pd.DataFrame) -> pd.DataFrame:
    """One row per date and ticker, with the metrics we want to chart."""
    returns = prices.pct_change()
    growth = 100 * prices / prices.bfill().iloc[0]            # growth of $100
    vol_30d = returns.rolling(30).std() * np.sqrt(TRADING_DAYS)  # annualized
    drawdown = prices / prices.cummax() - 1                   # drop from prior peak

    frames = []
    for ticker in prices.columns:
        frames.append(
            pd.DataFrame(
                {
                    "date": prices.index,
                    "ticker": ticker,
                    "sector": SECTORS.get(ticker, "S&P 500 (benchmark)"),
                    "price": prices[ticker].values,
                    "daily_return": returns[ticker].values,
                    "growth_of_100": growth[ticker].values,
                    "volatility_30d": vol_30d[ticker].values,
                    "drawdown": drawdown[ticker].values,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def build_summary(prices: pd.DataFrame) -> pd.DataFrame:
    """One row per ticker: overall return, volatility, and worst drawdown."""
    returns = prices.pct_change().dropna(how="all")
    years = (prices.index[-1] - prices.index[0]).days / 365.25
    total_return = prices.iloc[-1] / prices.bfill().iloc[0] - 1
    summary = pd.DataFrame(
        {
            "sector": [SECTORS.get(t, "S&P 500 (benchmark)") for t in prices.columns],
            "annualized_return": (1 + total_return) ** (1 / years) - 1,
            "annualized_volatility": returns.std() * np.sqrt(TRADING_DAYS),
            "max_drawdown": (prices / prices.cummax() - 1).min(),
        },
        index=prices.columns,
    )
    summary["return_per_unit_risk"] = (
        summary["annualized_return"] / summary["annualized_volatility"]
    )
    return summary.reset_index(names="ticker")


def build_correlations(prices: pd.DataFrame) -> pd.DataFrame:
    """Correlation of daily returns between every pair of tickers (long format)."""
    corr = prices.pct_change().corr()
    corr.index.name = "ticker_a"
    corr.columns.name = "ticker_b"
    return corr.stack().rename("correlation").reset_index()


def run_pca(prices: pd.DataFrame) -> pd.DataFrame:
    """PCA on sector returns via SVD: how much of the movement is one shared 'market' factor?"""
    sector_returns = prices[list(SECTORS)].pct_change().dropna()
    centered = sector_returns - sector_returns.mean()
    _, singular_values, _ = np.linalg.svd(centered.values, full_matrices=False)
    explained = singular_values**2 / np.sum(singular_values**2)
    return pd.DataFrame(
        {"component": range(1, len(explained) + 1), "share_of_variance": explained}
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="use fake data (no internet)")
    args = parser.parse_args()

    prices = load_prices(args.demo)
    print(f"Loaded {prices.shape[0]} days x {prices.shape[1]} tickers "
          f"({prices.index[0].date()} to {prices.index[-1].date()})")

    build_tidy_table(prices).to_csv("sector_daily_metrics.csv", index=False)
    summary = build_summary(prices)
    summary.to_csv("sector_summary.csv", index=False)
    build_correlations(prices).to_csv("sector_correlations.csv", index=False)
    pca = run_pca(prices)
    pca.to_csv("sector_pca.csv", index=False)

    print("\nSummary (sorted by return per unit of risk):")
    print(summary.sort_values("return_per_unit_risk", ascending=False).round(3).to_string(index=False))
    print(f"\nFirst principal component explains {pca.share_of_variance.iloc[0]:.1%} "
          "of sector return variance.")
    print("\nSaved: sector_daily_metrics.csv, sector_summary.csv, "
          "sector_correlations.csv, sector_pca.csv")


if __name__ == "__main__":
    main()
