"""
app.py - the dashboard. Run with:   streamlit run app.py

All data work lives in scan.py; this file only arranges it on screen.
"""

import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import scan

st.set_page_config(page_title="Watchlist Scan", page_icon="📊", layout="wide")

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8984"   # series colors (validated categorical slots 1-2 + neutral)


# --------------------------------------------------------------------------- cached data

@st.cache_data(ttl=3600, show_spinner="Downloading prices…")
def get_prices(tickers):
    return scan.load_prices(list(tickers))


@st.cache_data(ttl=900, show_spinner="Fetching news…")
def get_news(tickers, hours):
    df = scan.fetch_news(scan.news_client(), list(tickers), hours=hours)
    df["type"] = [scan.news_type(h) for h in df["headline"]]
    return df


@st.cache_resource(show_spinner="Loading FinBERT (about a minute the first time)…")
def get_finbert():
    from transformers import pipeline
    return pipeline("text-classification", model="ProsusAI/finbert", top_k=None, truncation=True)


@st.cache_data(show_spinner="Scoring headlines…")
def get_sentiment(headlines):
    return scan.score_sentiment(get_finbert(), list(headlines))


# --------------------------------------------------------------------------- sidebar

watchlist = scan.load_watchlist()

with st.sidebar:
    st.header("Watchlist")
    edited = st.data_editor(pd.DataFrame({"ticker": watchlist}), num_rows="dynamic",
                            hide_index=True, key="wl_editor")
    if st.button("Save watchlist", width="stretch"):
        scan.save_watchlist(edited["ticker"].tolist())
        st.rerun()
    st.caption("Add or remove rows, then **Save**.")

    st.header("Settings")
    hours = st.slider("News lookback (hours)", 12, 168, 48, step=12)
    use_finbert = st.checkbox("Score headlines with FinBERT", value=False)
    if st.button("Refresh data", width="stretch"):
        st.cache_data.clear()
        st.rerun()
    if scan.news_client() is None:
        st.warning("No Alpaca keys found in `.env`, so news is off.")

if not watchlist:
    st.info("Add some tickers in the sidebar to get started.")
    st.stop()


# --------------------------------------------------------------------------- data

close, volume = get_prices(tuple(watchlist))
metrics = scan.scan_metrics(close, volume, watchlist)
ctx = scan.market_context(close)
news = get_news(tuple(watchlist), hours).copy()
if use_finbert and not news.empty:
    news["sentiment"] = get_sentiment(tuple(news["headline"]))


# --------------------------------------------------------------------------- header + market context

st.title("Daily watchlist scan")
st.caption(f"Prices as of {ctx['as_of']} · During market hours the latest day is still in progress.")

valid = metrics[metrics["note"] == ""] if "note" in metrics else metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("S&P 500 (SPY) today", f"{ctx['spy_1d'] * 100:+.2f}%")
c2.metric("S&P 500 past month", f"{ctx['spy_1m'] * 100:+.1f}%")
c3.metric("VIX (market fear gauge)", f"{ctx['vix']:.1f}", f"{ctx['vix_chg']:+.1f}", delta_color="inverse")
c4.metric("Shocks in watchlist today", int(valid["shock"].sum()) if "shock" in valid else 0)


# --------------------------------------------------------------------------- today's notes

st.subheader("Today's notes")
any_notes = False
order = metrics.sort_values("shock", ascending=False) if "shock" in metrics else metrics
for _, row in order.iterrows():
    t = row["ticker"]
    recent = scan.news_since_last_close(news[news["ticker"] == t], close.index)
    notes = scan.insights_for(row, recent)
    if notes:
        any_notes = True
        st.markdown(f"**{t}** · " + " · ".join(notes))
if not any_notes:
    st.write("Nothing unusual today.")


tab_scan, tab_news, tab_chart, tab_research = st.tabs(["Scan", "News", "Chart", "Research"])

# --------------------------------------------------------------------------- scan table

with tab_scan:
    cols = ["ticker", "close", "ret_1d", "ret_5d", "ret_1m", "abn_1d", "abn_z", "shock",
            "vol_ratio", "off_52w_high", "rsi_14", "beta"]
    disp = metrics.reindex(columns=cols).copy()
    for c in ["ret_1d", "ret_5d", "ret_1m", "abn_1d", "off_52w_high"]:
        disp[c] = disp[c] * 100
    disp["news"] = disp["ticker"].map(news.groupby("ticker").size()).fillna(0).astype(int)

    st.dataframe(disp, hide_index=True, column_config={
        "ticker": "Ticker",
        "close": st.column_config.NumberColumn("Close", format="$%.2f"),
        "ret_1d": st.column_config.NumberColumn("1 day", format="%.1f%%"),
        "ret_5d": st.column_config.NumberColumn("5 days", format="%.1f%%"),
        "ret_1m": st.column_config.NumberColumn("1 month", format="%.1f%%"),
        "abn_1d": st.column_config.NumberColumn("Abnormal 1d", format="%.1f%%",
                                                help="Today's move minus what the market move predicts (beta × SPY)"),
        "abn_z": st.column_config.NumberColumn("Abnormal z", format="%.1f",
                                               help="Abnormal move in units of its usual daily size. ≤ -2 = shock"),
        "shock": st.column_config.CheckboxColumn("Shock?"),
        "vol_ratio": st.column_config.NumberColumn("Volume vs 20d", format="%.1fx"),
        "off_52w_high": st.column_config.NumberColumn("From 52w high", format="%.0f%%"),
        "rsi_14": st.column_config.NumberColumn("RSI 14", format="%.0f"),
        "beta": st.column_config.NumberColumn("Beta", format="%.2f"),
        "news": st.column_config.NumberColumn(f"Headlines ({hours}h)"),
    })
    st.caption("Shock = abnormal move at or below −2 standard deviations, the same rule as the research notebook. "
               "Beta and volatility use the previous 250 trading days.")

# --------------------------------------------------------------------------- news

with tab_news:
    if news.empty:
        st.info("No headlines found in this window.")
    for t in watchlist:
        tn = news[news["ticker"] == t]
        if tn.empty:
            continue
        with st.expander(f"{t} · {len(tn)} headlines", expanded=False):
            for _, a in tn.iterrows():
                when = a["time"].tz_convert("America/New_York").strftime("%b %d %H:%M ET")
                sent = f" · sentiment {a['sentiment']:+.2f}" if "sentiment" in tn and pd.notna(a.get("sentiment")) else ""
                title = f"[{a['headline']}]({a['url']})" if a["url"] else a["headline"]
                st.markdown(f"`{a['type']}` {title}  \n<small>{when} · {a['source']}{sent}</small>",
                            unsafe_allow_html=True)
    st.caption("Types come from the keyword rules in the research notebook. "
               "'price_move' and 'generic' headlines describe the move or are noise, not a cause.")

# --------------------------------------------------------------------------- chart

with tab_chart:
    left, right = st.columns([1, 3])
    pick = left.selectbox("Ticker", [t for t in watchlist if t in close.columns])
    span = left.radio("Period", ["3M", "6M", "1Y", "2Y"], index=2)
    days = {"3M": 63, "6M": 126, "1Y": 252, "2Y": 504}[span]

    r = close[pick].pct_change(fill_method=None)
    m = close[scan.BENCHMARK].pct_change(fill_method=None)
    beta = (r.rolling(scan.LOOKBACK).cov(m) / m.rolling(scan.LOOKBACK).var()).shift(1)
    abn = r - beta * m
    shock_days = abn[abn <= scan.SHOCK_Z * abn.rolling(scan.LOOKBACK).std().shift(1)].index

    window = close[[pick, scan.BENCHMARK]].dropna().iloc[-days:]
    idx = window / window.iloc[0] * 100
    shocks_in = idx.index.intersection(shock_days)

    fig = go.Figure()
    fig.add_scatter(x=idx.index, y=idx[scan.BENCHMARK], name="SPY", line=dict(color=GRAY, width=2))
    fig.add_scatter(x=idx.index, y=idx[pick], name=pick, line=dict(color=BLUE, width=2))
    fig.add_scatter(x=shocks_in, y=idx.loc[shocks_in, pick], name="Shock day", mode="markers",
                    marker=dict(color=ORANGE, size=10, line=dict(color="white", width=2)))
    fig.update_layout(template="plotly_white", hovermode="x unified", height=460,
                      margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Indexed to 100 at start",
                      legend=dict(orientation="h", y=1.08, x=0))
    right.plotly_chart(fig, width="stretch")
    right.caption("Shock markers need 250 days of history first, so the earliest year of the 2Y view has none.")

# --------------------------------------------------------------------------- research

with tab_research:
    st.markdown("""
**Summary of findings so far (2017–2023, 58 stocks, date-clustered tests)**
- All shocks together: no reliable recovery.
- **No-news shocks recovered more than news-driven ones** in all 5 checks; significant for days 2–10
  (p = 0.047, or 0.017 excluding 2020). *Not yet confirmed on 2024–2026 data.*
- FinBERT headline tone did not predict recovery. The "sector move" label was an artifact of 2019–2020 headline style.
""")
    if os.path.exists("events_covered.csv"):
        ev = pd.read_csv("events_covered.csv")
        ev["news"] = ev["has_news"].map({True: "With news", False: "No news"})
        tbl = (ev.groupby(["group", "news"])[["after5_pct", "after10_pct", "after20_pct"]]
                 .mean().round(2).assign(events=ev.groupby(["group", "news"]).size()))
        st.dataframe(tbl.rename(columns={"after5_pct": "Days 2–5 %", "after10_pct": "Days 2–10 %",
                                         "after20_pct": "Days 2–20 %"}))

        bars = ev.groupby(["group", "news"])["after10_pct"].mean().unstack()
        fig2 = go.Figure()
        for label, color in [("With news", BLUE), ("No news", ORANGE)]:
            if label in bars:
                fig2.add_bar(x=bars.index, y=bars[label], name=label, marker_color=color,
                             hovertemplate="%{x}<br>" + label + ": %{y:.2f}%<extra></extra>")
        fig2.update_layout(template="plotly_white", barmode="group", bargap=0.35, bargroupgap=0.08,
                           height=380, margin=dict(l=10, r=10, t=40, b=10),
                           title="Average abnormal return, days 2–10 after a shock",
                           yaxis_title="%", legend=dict(orientation="h", y=1.12, x=0))
        st.plotly_chart(fig2, width="stretch")
        st.caption("Groups: how many of the 58 stocks shocked on the same day (alone / 2–4 / 5+).")
    else:
        st.info("Put `events_covered.csv` in this folder to see the research charts.")

st.divider()
st.caption("Educational project. Not investment advice.")
