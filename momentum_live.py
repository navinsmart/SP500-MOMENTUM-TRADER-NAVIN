"""
S&P 500 Momentum Trading Robot
- 6-month momentum
- Top 10 stocks
- Weekly rebalancing
- Volatility targeting (aggressive)
- Fractional shares
- Crash Protection (Hysteresis):
    - Exit to cash when SPY < 100-day MA
    - Reinvest only when SPY > 55-day MA
"""

import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

# ============================================================
# CREDENTIALS
# ============================================================
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")

if not API_KEY or not API_SECRET:
    raise ValueError("Missing ALPACA_API_KEY or ALPACA_SECRET_KEY")

PAPER = False   # False = Live trading

# ============================================================
# STRATEGY SETTINGS
# ============================================================
TOP_N = 10
MOMENTUM_DAYS = 126
VOL_LOOKBACK = 63
TARGET_PORTFOLIO_VOL = 0.20
MAX_POSITION_WEIGHT = 0.20
MIN_PRICE = 5.0
CASH_BUFFER = 0.00

# Crash Protection
USE_MARKET_TREND_FILTER = True
EXIT_MA_DAYS = 100
REENTRY_MA_DAYS = 55

# ============================================================
# GET S&P 500 TICKERS
# ============================================================
def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        import requests
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        tables = pd.read_html(response.text)
        df = tables[0]
        tickers = df["Symbol"].astype(str).str.strip().str.replace(".", "-", regex=False).tolist()
        tickers = [t for t in tickers if t.isalpha() or "-" in t]
        print(f"Loaded {len(tickers)} S&P 500 tickers")
        return tickers
    except Exception as e:
        print(f"Wikipedia error: {e}")
        return ["AAPL","MSFT","NVDA","AMZN","META","GOOGL","BRK-B","JPM","V","UNH",
                "XOM","JNJ","WMT","MA","PG","HD","CVX","MRK","ABBV","KO"]

# ============================================================
# ALPACA HELPERS
# ============================================================
trading_client = TradingClient(API_KEY, API_SECRET, paper=PAPER)

def get_account_equity():
    return float(trading_client.get_account().equity)

def get_current_positions():
    positions = trading_client.get_all_positions()
    return {p.symbol: float(p.qty) for p in positions}

def get_latest_prices(symbols):
    data = yf.download(symbols, period="5d", auto_adjust=True, progress=False)["Close"]
    if isinstance(data, pd.Series):
        return {data.name: float(data.iloc[-1])}
    return data.iloc[-1].to_dict()

def sell_everything():
    positions = get_current_positions()
    if not positions:
        print("Already in cash.")
        return
    print("Selling all positions...")
    for symbol, qty in positions.items():
        if abs(qty) > 0:
            side = OrderSide.SELL if qty > 0 else OrderSide.BUY
            order = MarketOrderRequest(symbol=symbol, qty=abs(qty), side=side, time_in_force=TimeInForce.DAY)
            trading_client.submit_order(order)
            print(f"  Sold {symbol}")
    print("All positions closed.")

# ============================================================
# MARKET FILTER (SPY)
# ============================================================
def is_market_healthy():
    if not USE_MARKET_TREND_FILTER:
        return True

    spy = yf.download("SPY", period="1y", auto_adjust=True, progress=False)["Close"]
    if isinstance(spy, pd.DataFrame):
        spy = spy.iloc[:, 0]

    if len(spy) < EXIT_MA_DAYS:
        return True

    current = float(spy.iloc[-1])
    ma_exit = float(spy.rolling(EXIT_MA_DAYS).mean().iloc[-1])
    ma_reentry = float(spy.rolling(REENTRY_MA_DAYS).mean().iloc[-1])

    print(f"SPY: {current:.2f} | 100-MA: {ma_exit:.2f} | 55-MA: {ma_reentry:.2f}")

    positions = get_current_positions()
    currently_invested = len(positions) > 0

    if currently_invested:
        if current < ma_exit:
            print("→ SPY below 100-day MA → SELL EVERYTHING")
            return False
        else:
            print("→ Healthy (above 100-day MA)")
            return True
    else:
        if current > ma_reentry:
            print("→ SPY above 55-day MA → ALLOW REINVESTING")
            return True
        else:
            print("→ Below 55-day MA → STAY IN CASH")
            return False

# ============================================================
# TARGET WEIGHTS
# ============================================================
def calculate_target_weights():
    print("Getting S&P 500 list...")
    universe = get_sp500_tickers()

    end = datetime.now()
    start = end - timedelta(days=300)

    print("Downloading price data...")
    raw = yf.download(universe, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    raw = raw.dropna(axis=1, how="all")
    print(f"Data ready for {len(raw.columns)} stocks")

    momentum = raw.pct_change(periods=MOMENTUM_DAYS).iloc[-1].dropna()
    last_prices = raw.iloc[-1]
    momentum = momentum[last_prices.reindex(momentum.index) > MIN_PRICE]

    if len(momentum) < TOP_N:
        return {}

    ranked = momentum.sort_values(ascending=False)
    print("\n===== TOP 15 (6-Month Momentum) =====")
    for i, (ticker, score) in enumerate(ranked.head(15).items(), 1):
        print(f"{i:2d}. {ticker:6s}  {score*100:6.1f}%")
    print("=====================================\n")

    top = ranked.head(TOP_N).index.tolist()
    print(f"Selected Top {TOP_N}: {top}")

    daily_rets = raw.pct_change()
    vols = daily_rets[top].iloc[-VOL_LOOKBACK:].std() * np.sqrt(252)
    vols = vols.replace(0, np.nan).dropna()

    if len(vols) < 3:
        weight = (1.0 - CASH_BUFFER) / len(top)
        return {t: weight for t in top}

    inv_vol = 1.0 / vols
    raw_w = inv_vol / inv_vol.sum()
    raw_w = raw_w.clip(upper=MAX_POSITION_WEIGHT)
    raw_w = raw_w / raw_w.sum()

    port_vol = np.sqrt((raw_w**2 * vols**2).sum())
    scale = min(TARGET_PORTFOLIO_VOL / port_vol, 1.5) if port_vol > 0 else 1.0

    final = (raw_w * scale).clip(upper=MAX_POSITION_WEIGHT)
    final = final / final.sum() * (1.0 - CASH_BUFFER)
    return final.to_dict()

# ============================================================
# REBALANCE
# ============================================================
def rebalance():
    print(f"\n[{datetime.now()}] Starting S&P 500 rebalance...")
    equity = get_account_equity()
    print(f"Account equity: ${equity:,.2f}")

    if not is_market_healthy():
        sell_everything()
        print("Market filter → staying in cash.")
        return

    targets = calculate_target_weights()
    if not targets:
        print("No targets generated.")
        return

    print("\nTarget weights:")
    for symbol, weight in sorted(targets.items(), key=lambda x: -x[1]):
        print(f"  {symbol}: {weight:.1%}")

    current_positions = get_current_positions()
    all_symbols = list(set(list(targets.keys()) + list(current_positions.keys())))
    prices = get_latest_prices(all_symbols)
    target_dollars = {s: equity * w for s, w in targets.items()}

    for symbol, qty in current_positions.items():
        if symbol not in target_dollars and abs(qty) > 0:
            side = OrderSide.SELL if qty > 0 else OrderSide.BUY
            order = MarketOrderRequest(symbol=symbol, qty=abs(qty), side=side, time_in_force=TimeInForce.DAY)
            trading_client.submit_order(order)
            print(f"Closing {symbol}")

    time.sleep(2)
    current_positions = get_current_positions()

    for symbol, target_value in target_dollars.items():
        current_qty = current_positions.get(symbol, 0)
        current_price = prices.get(symbol, 0)
        if current_price <= 0:
            continue
        current_value = current_qty * current_price
        diff = target_value - current_value
        if abs(diff) < 5:
            continue
        if diff > 0:
            order = MarketOrderRequest(symbol=symbol, notional=round(diff, 2), side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
            trading_client.submit_order(order)
            print(f"BUY ${diff:.2f} of {symbol}")
        else:
            qty = abs(diff) / current_price
            order = MarketOrderRequest(symbol=symbol, qty=round(qty, 4), side=OrderSide.SELL, time_in_force=TimeInForce.DAY)
            trading_client.submit_order(order)
            print(f"SELL {qty:.4f} of {symbol}")

    print("\nRebalance completed.")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("S&P 500 Momentum Robot started")
    print(f"Paper: {PAPER}")
    rebalance()
    print("Done.")

