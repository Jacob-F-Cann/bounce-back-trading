"""
scan.py - data and analysis for the daily watchlist scan.

Everything here is plain Python with no Streamlit code, so the same functions
can later run on a schedule (e.g. a morning email) as well as in the dashboard.
To add a new insight: write a function here, then show its output in app.py.
"""

import os
import re
from datetime import timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

BENCHMARK = "SPY"
LOOKBACK = 250      # trading days used for beta and "normal" volatility
SHOCK_Z = -2.0      # abnormal return at or below -2 std devs = shock (same rule as the research)


# --------------------------------------------------------------------------- prices

def load_watchlist(path="watchlist.csv"):
    if os.path.exists(path):
        tickers = pd.read_csv(path)["ticker"].dropna().astype(str).str.upper().str.strip()
        return [t for t in tickers if t]
    return ["AAPL", "MSFT", "NVDA"]


def save_watchlist(tickers, path="watchlist.csv"):
    clean = sorted({str(t).upper().strip() for t in tickers if str(t).strip()})
    pd.DataFrame({"ticker": clean}).to_csv(path, index=False)
    return clean


def load_prices(tickers, period="2y"):
    """Adjusted daily close and volume for the tickers plus the benchmark and VIX."""
    symbols = sorted(set(tickers) | {BENCHMARK, "^VIX"})
    data = yf.download(symbols, period=period, auto_adjust=True, progress=False)
    return data["Close"], data["Volume"]


def rsi(close, n=14):
    """Relative Strength Index (Wilder). Above 70 = 'overbought', below 30 = 'oversold'."""
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return float(100 - 100 / (1 + up.iloc[-1] / down.iloc[-1]))


def scan_metrics(close, volume, tickers):
    """One row per ticker: returns, beta, abnormal return and shock flag, volume, 52w high, RSI."""
    rets = close.pct_change(fill_method=None)
    rows = []
    for t in tickers:
        if t not in close.columns or close[t].dropna().shape[0] < 60:
            rows.append({"ticker": t, "note": "not enough price data"})
            continue

        c = close[t].dropna()
        v = volume[t].dropna()
        pair = pd.concat([rets[t], rets[BENCHMARK]], axis=1, keys=["s", "m"]).dropna()
        hist = pair.iloc[-LOOKBACK - 1:-1]                       # history before the latest day
        beta = hist["s"].cov(hist["m"]) / hist["m"].var()
        abn = pair["s"] - beta * pair["m"]                        # abnormal (market-adjusted) returns
        sigma = abn.iloc[-LOOKBACK - 1:-1].std()
        z = abn.iloc[-1] / sigma

        rows.append({
            "ticker": t,
            "date": c.index[-1].date(),
            "close": c.iloc[-1],
            "ret_1d": c.iloc[-1] / c.iloc[-2] - 1,
            "ret_5d": c.iloc[-1] / c.iloc[-6] - 1,
            "ret_1m": c.iloc[-1] / c.iloc[-22] - 1,
            "abn_1d": abn.iloc[-1],
            "abn_z": z,
            "shock": bool(z <= SHOCK_Z),
            "beta": beta,
            "vol_ratio": v.iloc[-1] / v.iloc[-21:-1].mean(),
            "off_52w_high": c.iloc[-1] / c.iloc[-252:].max() - 1,
            "rsi_14": rsi(c),
            "note": "",
        })
    return pd.DataFrame(rows)


def market_context(close):
    spy = close[BENCHMARK].dropna()
    vix = close["^VIX"].dropna() if "^VIX" in close.columns else pd.Series(dtype=float)
    return {
        "spy_1d": spy.iloc[-1] / spy.iloc[-2] - 1,
        "spy_1m": spy.iloc[-1] / spy.iloc[-22] - 1,
        "vix": vix.iloc[-1] if len(vix) else np.nan,
        "vix_chg": vix.iloc[-1] - vix.iloc[-2] if len(vix) > 1 else np.nan,
        "as_of": spy.index[-1].date(),
    }


# --------------------------------------------------------------------------- news

def news_client():
    key, secret = os.getenv("ALPACA_KEY"), os.getenv("ALPACA_SECRET")
    if not key or not secret:
        return None
    from alpaca.data.historical.news import NewsClient
    return NewsClient(key, secret, raw_data=True)


def fetch_news(client, tickers, hours=48, per_ticker=50):
    """Latest headlines per ticker from Alpaca, kept if published in the last `hours`.

    We ask for the latest articles and filter by time ourselves, because date-range
    queries returned nothing on this account's news feed.
    """
    from alpaca.data.requests import NewsRequest
    cols = ["ticker", "time", "headline", "summary", "source", "url"]
    if client is None:
        return pd.DataFrame(columns=cols)

    cutoff = pd.Timestamp.now(tz="UTC") - timedelta(hours=hours)
    rows = []
    for t in tickers:
        try:
            resp = client.get_news(NewsRequest(symbols=t, limit=per_ticker))
        except Exception:
            continue
        for a in resp.get("news", []):
            when = pd.to_datetime(a.get("created_at"), utc=True)
            if when >= cutoff:
                rows.append({"ticker": t, "time": when, "headline": a.get("headline", ""),
                             "summary": a.get("summary", ""), "source": a.get("source", ""),
                             "url": a.get("url", "")})
    df = pd.DataFrame(rows, columns=cols)
    return df.sort_values("time", ascending=False).reset_index(drop=True)


# Keyword rules from the research notebook (generic/noise checked first, then causes)
RULES = [
    ("generic",    r"stocks to|\d+ (top |best |dividend |biotech |tech )?stocks|should you|better buy|is it time|top pick|\betf\b|watchlist"),
    ("macro",      r"^us stocks|wall st|\bs&p 500\b|\bdow\b|\bfed\b|inflation|\bcpi\b|interest rate|tariff|recession|jobs report|treasury yields"),
    ("analyst",    r"downgrad|upgrad|price target|initiat|maintains|underweight|overweight|outperform|underperform|equal-weight"),
    ("guidance",   r"guidance|outlook|forecast|warns|cuts? (its )?view"),
    ("earnings",   r"earnings|\beps\b|quarter|\bq[1-4]\b|results|revenue|profit|beats|misses|\bmiss\b"),
    ("legal_reg",  r"lawsuit|\bsues?\b|sued|probe|investigat|\bsec\b|\bdoj\b|\bfda\b|recall|regulat|antitrust|\bfined?\b|settle"),
    ("deal",       r"acqui|merger|\bdeal\b|buyout|takeover|\bstake\b|rumor|\b(to buy|buys)\b.*\$[\d.]+ ?(bln|billion|mln|million|b|m)\b"),
    ("management", r"\bceo\b|\bcfo\b|resign|steps down|chairman|executive"),
    ("capital",    r"buyback|repurchase|dividend"),
    ("price_move", r"shares (are )?(down|up|fall|drop|slide|sink|tumble|plunge|rise|gain|trading)|stock (is )?(down|falls|drops|slides|sinks)"
                   r"|why .* stock|crash|soar|plunge|tumble|oversold|overbought|52-week|trading (lower|higher)|biggest (losers|decliners)|movers|mid-?day|pre-?market"),
]
CAUSES = {"macro", "analyst", "guidance", "earnings", "legal_reg", "deal", "management", "capital"}


def news_type(headline):
    h = str(headline).lower()
    for name, pattern in RULES:
        if re.search(pattern, h):
            return name
    return "other"


def news_since_last_close(news, close_index):
    """Headlines since 4 PM ET on the trading day before the latest close (the research's day -1/0 window)."""
    if news.empty or len(close_index) < 2:
        return news.iloc[0:0]
    prev = pd.Timestamp(close_index[-2]).tz_localize("America/New_York") + pd.Timedelta(hours=16)
    return news[news["time"] >= prev.tz_convert("UTC")]


def score_sentiment(pipe, headlines):
    """FinBERT score per headline: positive prob minus negative prob, from -1 to +1."""
    if not headlines:
        return []
    out = pipe(list(headlines), batch_size=16)
    return [{d["label"]: d["score"] for d in res}["positive"] - {d["label"]: d["score"] for d in res}["negative"]
            for res in out]


# --------------------------------------------------------------------------- insights

def insights_for(row, ticker_news):
    """Short, plain-language notes for one ticker. Add new rules here as the project grows."""
    notes = []
    if row.get("note"):
        return [row["note"]]

    causes = ticker_news[ticker_news["type"].isin(CAUSES)] if not ticker_news.empty else ticker_news
    if row["shock"]:
        if ticker_news.empty:
            notes.append("🔻 Shock with **no news** found. In the 2017–2023 study, no-news shocks recovered "
                         "~0.8pp more over days 2–10 than news-driven ones (not yet confirmed on recent data).")
        elif causes.empty:
            notes.append("🔻 Shock; headlines found but none give a clear cause.")
        else:
            top = causes["type"].value_counts().idxmax()
            notes.append(f"🔻 Shock with **{top}** news. In the study, news-driven shocks mostly did not recover.")
    if row["vol_ratio"] >= 2:
        notes.append(f"📈 Volume {row['vol_ratio']:.1f}× its 20-day average.")
    if row["rsi_14"] <= 30:
        notes.append(f"RSI {row['rsi_14']:.0f}: technically oversold.")
    elif row["rsi_14"] >= 70:
        notes.append(f"RSI {row['rsi_14']:.0f}: technically overbought.")
    if row["off_52w_high"] >= -0.01:
        notes.append("At or near a 52-week high.")
    elif row["off_52w_high"] <= -0.30:
        notes.append(f"{abs(row['off_52w_high']) * 100:.0f}% below its 52-week high.")
    return notes
