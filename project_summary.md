# Stock News & Overreaction Project: Summary So Far

*Last updated: November 21, 2026*

## What we're building

1. **A daily watchlist dashboard** that scans our stocks, pulls recent news, and flags anything unusual.
2. **A data science study** asking: *when a stock drops sharply on a given day, does it tend to recover (overreaction) or keep falling (drift)? And does the news behind the drop predict which?* The longer-term idea is to find possible swing-trade setups where the market has overreacted.

This is mainly a learning project, so the focus has been on doing the analysis correctly rather than finding a quick strategy.

---

## Setup

- **Python 3** with **VS Code**, in a project folder (`C:\Projects\stock-news`) with a virtual environment (`.venv`).
- **Jupyter notebooks** for research; a **Streamlit** app for the dashboard.
- Main libraries: `yfinance` (prices), `pandas`/`numpy`, `scipy` and `statsmodels` (statistics), `transformers` + `torch` (FinBERT), `alpaca-py` (live news), `streamlit` + `plotly` (dashboard).
- API keys are stored in a `.env` file in the project folder, **never in code**. Don't share or commit that file.

---

## Research method: event study

This is the standard finance approach for measuring how prices react to events.

1. **Abnormal return (AR):** a stock's daily return minus what the market's move predicts (beta × S&P 500 return). Beta and "normal" volatility are estimated from the **previous 250 trading days only**, so no future information leaks in.
2. **Shock:** a day when the abnormal return is **at or below −2 standard deviations**, an unusually large stock-specific drop.
3. **Recovery:** the cumulative abnormal return over **days 2–5, 2–10 and 2–20** after the shock. We start on day 2 because you'd realistically learn of the drop on day 0 and act on day 1. These three windows were chosen up front and are always reported together, to avoid picking whichever looks best.
4. **Clustering:** when many stocks shock on the same day, they tend to recover together, so they aren't independent observations. All significance tests are **clustered by date**. This turned out to matter a lot (see below).

**Universe:** 58 large US stocks across sectors (tech, banks, healthcare, consumer, industrials, energy, telecom/utilities), 2015–2026 prices, about 3,600 shock events.

---

## Data sources (and problems we ran into)

| Need | Source | Notes |
|---|---|---|
| Prices | yfinance | Split- and dividend-adjusted. GE and 3M excluded because spin-offs can show up as fake crashes. |
| Historical news | **FNSPID dataset** (Hugging Face, free) | Headlines through 2023. **Dates only, no times**, so a news item counts for an event if it's dated on the shock day or the trading day before. |
| Live news | Alpaca News API (free paper account) | Works for recent news only. Our free account gets a feed (fiscal.ai) with essentially no history, so it couldn't be used for research. |
| Recent history (2024–26) | Massive (formerly Polygon) or Finnhub free tiers | **Not done yet.** Planned for out-of-sample testing. |

**Data lessons:**
- Facebook appears as **FB** and Alphabet partly as **GOOG** in FNSPID, so ticker renames had to be mapped (FB→META, GOOG→GOOGL).
- News coverage is uneven by stock and year, so we only kept events where the stock had at least **100 articles that year**. Otherwise "no news" might just mean "not covered."
- Remaining biases: **survivorship** (all 58 stocks still exist today, which flatters "buy the dip") and a hand-picked universe of large caps.

---

## Results

| # | Question | Result |
|---|---|---|
| 1 | Do shock drops recover on average? | Small bounce (+0.27% over days 2–10). p = 0.002 when each event is treated as independent, but **p = 0.71 after clustering by date**. **No reliable effect.** |
| 2 | Do bigger drops bounce more? | Slightly, but not reliably. |
| 3 | Market-wide shocks (5+ stocks on the same day) vs. stocks falling alone | Market-wide shocks bounced (+0.54%, p = 0.025); stocks falling alone kept slipping slightly. **But the bounce was driven largely by 2020** (COVID crash and rebound). |
| 4 | **No-news vs. news-driven shocks** (2017–2023, 2,001 events) | **No-news shocks recovered more in all 5 checks** (+0.21 to +1.16 percentage points). Significant for days 2–10 (p = 0.047, and **p = 0.017 excluding 2020**). Not significant for days 2–5 or 2–20. The effect is concentrated in stocks that fell alongside others without news of their own ("collateral damage"). |
| 5 | Does FinBERT headline tone predict recovery? | **No.** Most-negative, middle and least-negative tone groups recovered about the same, and all p-values were above 0.6. FinBERT scores tone, not importance, and headlines like "Shares fall…" are negative *because* of the drop. |
| 6 | Does news **type** matter? (keyword classifier: earnings, analyst, guidance, legal, deal, macro…) | Exploratory only. Earnings-driven drops don't recover (consistent with the known post-earnings drift effect). A promising "sector move" category turned out to be an **artifact**: 113 of 147 events were from 2020, because of a headline style Benzinga used in 2019–2020. |

### Main takeaway
> Drops driven by a stock's own news tend to stick. Drops where the stock seems caught up in a broader move, with no news of its own, tend to partly recover over the next 1–2 weeks. **This is a hypothesis, not a confirmed result.**

### Pre-registered hypothesis (to test once on 2024–2026 data)
> Shocks in the 2–4 or 5+ group with no stock-specific news (day 0 or day −1) have positive cumulative abnormal return over days 2–10, about +1 percentage point above news-driven shocks.

### Methodology lessons
- **Cluster by date.** It turned a "significant" result (p = 0.002) into nothing (p = 0.71).
- **Check whether one period drives a result.** 2020 drove two apparent findings.
- **Check when a text label occurs before trusting it.** A category that only appears in some years usually reflects a change in writing style, not the market.
- **Run many tests and some will look significant by chance.** That's why the windows were fixed up front, and why the main finding still needs a clean out-of-sample test.
- **Read the raw data.** Reading sample headlines exposed classifier mistakes that summary numbers hid.

---

## Dashboard (v1)

Run with `streamlit run app.py` from the project folder, then open **http://localhost:8501**.

- **Market context:** S&P 500 today and past month, VIX, number of shocks in the watchlist today.
- **Today's notes:** plain-language flags per stock: shock (with whether news explains it), unusual volume, RSI extremes, near or far from the 52-week high.
- **Scan tab:** returns (1 day, 5 days, 1 month), abnormal return and z-score, shock flag, volume vs. 20-day average, distance from 52-week high, RSI, beta, headline count.
- **News tab:** recent Alpaca headlines per stock, labeled by type; optional FinBERT sentiment.
- **Chart tab:** a stock vs. the S&P 500 with past shock days marked.
- **Research tab:** findings summary and a no-news vs. news chart.
- **Sidebar:** an editable watchlist (saved to `watchlist.csv`).

The code is split so new insights are easy to add: **`scan.py`** holds all data and analysis functions (it can later run on a schedule without the dashboard), and **`app.py`** handles only the layout.

---

## Project files

| File | What it is |
|---|---|
| `first_look.py` | First script: price download and chart |
| `first_events.ipynb` | Event study, clustering tests, news matching |
| `finbert.ipynb` | FinBERT scoring and news-type classification |
| `fnspid_combined.csv` | Filtered FNSPID headlines for our tickers (with FB/GOOG fixes) |
| `events.csv`, `events_covered.csv` | All shock events, and those with news coverage, with recovery windows |
| `headline_sentiment.csv`, `news_events_scored.csv` | FinBERT scores per headline and per event |
| `app.py`, `scan.py`, `watchlist.csv` | The dashboard |
| `.env` | API keys (**private**) |

---

## Next steps (candidates)

1. **Out-of-sample test:** pull 2024–2026 news (Massive or Finnhub) and test the pre-registered no-news hypothesis once.
2. **Grow the daily scan:** earnings dates, sector context, LLM summaries of key articles and why they matter, and a scheduled morning email or alert.
3. **Better news labels:** use an LLM to judge whether a headline actually explains the drop, instead of keyword rules.
4. **A machine-learning model** combining all features with walk-forward testing. Expect results close to chance; mainly useful for learning about overfitting.
5. **Paper-trade** any rule that survives step 1 before considering real money.

*Educational project. Nothing here is investment advice.*
