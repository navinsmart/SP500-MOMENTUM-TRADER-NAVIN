import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import plotly.express as px

st.set_page_config(page_title="S&P 500 Momentum", page_icon="📊", layout="centered")

st.title("📊 S&P 500 Momentum")
st.caption("6-month momentum + Crash Protection")

# Market Filter
st.subheader("Market Filter Status")

@st.cache_data(ttl=300)
def get_market_status():
    data = yf.download("SPY", period="1y", auto_adjust=True, progress=False)
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    current = float(close.iloc[-1])
    ma100 = float(close.rolling(100).mean().iloc[-1])
    ma55 = float(close.rolling(55).mean().iloc[-1])
    return current, ma100, ma55

try:
    price, ma100, ma55 = get_market_status()
    c1, c2, c3 = st.columns(3)
    c1.metric("SPY Price", f"${price:.2f}")
    c2.metric("100-day MA", f"${ma100:.2f}")
    c3.metric("55-day MA", f"${ma55:.2f}")

    if price > ma100:
        st.success("✅ HEALTHY – Invested Mode")
    else:
        st.error("🛡️ CASH MODE – Protection Active")
except:
    st.warning("Could not load market data")

st.markdown("---")

# Portfolio
st.subheader("Your Portfolio")
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
if API_KEY and API_SECRET:
    try:
        from alpaca.trading.client import TradingClient
        client = TradingClient(API_KEY, API_SECRET, paper=False)
        acc = client.get_account()
        c1, c2 = st.columns(2)
        c1.metric("Equity", f"${float(acc.equity):,.2f}")
        c2.metric("Cash", f"${float(acc.cash):,.2f}")
    except:
        st.info("Could not load Alpaca account")
else:
    st.info("Add Alpaca keys in Streamlit Secrets")

st.markdown("---")

top_n = st.selectbox("Show Top", [5, 10, 15, 20], index=1)

@st.cache_data(ttl=1800)
def get_sp500():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        import requests
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        df = pd.read_html(r.text)[0]
        return df["Symbol"].str.replace(".", "-", regex=False).tolist()
    except:
        return ["AAPL","MSFT","NVDA","AMZN","META","GOOGL","JPM","V","UNH","XOM"]

@st.cache_data(ttl=1800)
def get_ranking(tickers, top_n):
    data = yf.download(tickers, period="1y", auto_adjust=True, progress=False)["Close"]
    data = data.dropna(axis=1, how="all")
    mom = data.pct_change(126).iloc[-1].dropna()
    mom = mom[data.iloc[-1].reindex(mom.index) > 5]
    ranked = mom.sort_values(ascending=False)
    df = pd.DataFrame({
        "Rank": range(1, len(ranked)+1),
        "Symbol": ranked.index,
        "Momentum": (ranked.values*100).round(1),
        "Price": data.iloc[-1].reindex(ranked.index).round(2).values
    })
    df["Momentum"] = df["Momentum"].astype(str) + "%"
    return df

with st.spinner("Loading..."):
    tickers = get_sp500()
    ranking = get_ranking(tickers, top_n)

st.subheader(f"Top {top_n} Stocks")
st.dataframe(ranking.head(top_n), use_container_width=True, hide_index=True)
st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
