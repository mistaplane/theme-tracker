import os
import sys
import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

import pandas as pd
import yfinance as yf
import plotly.express as px

from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import Qt

# =========================
# Logging
# =========================
logging.basicConfig(
    filename='tracker.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# =========================
# Config
# =========================
DEFAULT_DAYS = 4           # “N-day average % change” for themes
CACHE_TTL_SEC = 240        # 4-minute cache to avoid repeated downloads
UNIVERSE_CSV = "/Users/mistaplane/Documents/scripts/theme_tracker/universe.csv"

# =========================
# Helpers
# =========================
def _candidate_theme_paths() -> List[Path]:
    """Candidate locations for themes.json (env var, MEIPASS, exe dir, CWD)."""
    candidates: List[Path] = []
    env_path = os.getenv("THEMES_JSON")
    if env_path:
        candidates.append(Path(env_path))

    # PyInstaller bundle
    base_meipass = getattr(sys, "_MEIPASS", None)
    if base_meipass:
        candidates.append(Path(base_meipass) / "themes.json")

    # Executable dir (frozen) or script dir
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).parent / "themes.json")
    else:
        candidates.append(Path(__file__).resolve().parent / "themes.json")

    # Current working directory as last resort
    candidates.append(Path.cwd() / "themes.json")
    return candidates


def get_themes_path() -> Path:
    """Return the path to themes.json, checking multiple possible locations."""
    for p in _candidate_theme_paths():
        if p.exists():
            logging.info(f"Found themes.json at {p}")
            return p
    logging.error("themes.json not found in any candidate paths")
    print("Error: themes.json not found. Set THEMES_JSON env var or place it beside the script/exe.")
    return Path()  # empty


def load_themes() -> Dict[str, List[str]]:
    """Load JSON mapping: {theme: [tickers...]}."""
    path = get_themes_path()
    if not path:
        return {}
    try:
        with path.open("r") as f:
            themes = json.load(f)
            # Normalize/clean tickers
            themes = {
                theme: sorted({t.strip().upper() for t in tickers if t and str(t).strip()})
                for theme, tickers in themes.items()
            }
            return themes
    except Exception as e:
        logging.exception(f"Failed to load themes.json at {path}: {e}")
        print(f"Error: failed to load themes.json at {path}")
        return {}


def _cache_file() -> Path:
    return Path(tempfile.gettempdir()) / "cc_themes_cache.parquet"


def _is_cache_fresh(p: Path) -> bool:
    if not p.exists():
        return False
    age = datetime.now() - datetime.fromtimestamp(p.stat().st_mtime)
    return age.total_seconds() <= CACHE_TTL_SEC


def fetch_prices_all(themes: Dict[str, List[str]], days: int = DEFAULT_DAYS) -> pd.DataFrame:
    """
    Fetch Adj Close for all unique tickers across themes in ONE call.
    Returns a DataFrame with index=dates, columns=tickers (Adj Close/Close).
    """
    if not themes:
        return pd.DataFrame()

    # Deduplicate tickers across themes
    all_tickers = sorted({t for arr in themes.values() for t in arr})
    if not all_tickers:
        return pd.DataFrame()

    cache_path = _cache_file()
    if _is_cache_fresh(cache_path):
        try:
            df_cached = pd.read_parquet(cache_path)
            # Only reuse if cache has all tickers we need
            if set(all_tickers).issubset(set(df_cached.columns)):
                logging.info("Using cached quotes")
                return df_cached.tail(days)
        except Exception:
            pass  # fall through to fresh download

    logging.info(f"Downloading quotes for {len(all_tickers)} tickers in one batch")
    df = yf.download(
        tickers=all_tickers,
        period=f"{days}d",
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False,
    )

    if df.empty:
        logging.error("yfinance returned empty DataFrame")
        return pd.DataFrame()

    # Extract Adj Close for both single and multi-ticker shapes
    if isinstance(df.columns, pd.MultiIndex):
        cols: List[Tuple[str, str]] = []
        for t in all_tickers:
            if (t, "Adj Close") in df.columns:
                cols.append((t, "Adj Close"))
            elif (t, "Close") in df.columns:
                cols.append((t, "Close"))
        adj = df[cols].copy()
        adj.columns = [c[0] for c in adj.columns]
    else:
        if "Adj Close" in df.columns:
            adj = df[["Adj Close"]].copy()
        else:
            adj = df[["Close"]].copy()
        tick = all_tickers[0] if len(all_tickers) == 1 else "TICKER"
        adj.columns = [tick]

    adj = adj.tail(days)

    # Save/merge cache
    try:
        if cache_path.exists():
            existing = pd.read_parquet(cache_path)
            union = existing.join(adj, how="outer")
            union.to_parquet(cache_path, index=True)
        else:
            adj.to_parquet(cache_path, index=True)
    except Exception as e:
        logging.warning(f"Failed to write cache: {e}")

    return adj


def compute_theme_stats(themes: Dict[str, List[str]], prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-theme average % change across the window.
    - Daily pct_change across the window, mean per ticker, then mean across tickers.
    Returns a DataFrame indexed by theme with columns: avg_return, top_stocks, hot_rank
    """
    if prices.empty or not themes:
        return pd.DataFrame(columns=["avg_return", "top_stocks", "hot_rank"])

    # Use fill_method=None to avoid FutureWarning and preserve NAs
    daily_pct = prices.pct_change(fill_method=None) * 100.0
    last_day_change = daily_pct.iloc[-1:].T  # shape (tickers x 1)

    rows = []
    for theme, tickers in themes.items():
        tickers_in_data = [t for t in tickers if t in prices.columns]
        if not tickers_in_data:
            rows.append((theme, 0.0, []))
            continue

        theme_daily = daily_pct[tickers_in_data]
        avg_return = theme_daily.mean(axis=0, skipna=True).mean(skipna=True)

        # Top 3 by last-day change (kept for compatibility)
        if not last_day_change.empty:
            ls = (
                last_day_change.loc[tickers_in_data]
                .iloc[:, 0]
                .dropna()
                .sort_values(ascending=False)
                .head(3)
                .index.tolist()
            )
        else:
            ls = tickers_in_data[:3]

        rows.append((theme, float(avg_return) if pd.notna(avg_return) else 0.0, ls))

    df = pd.DataFrame(rows, columns=["theme", "avg_return", "top_stocks"]).set_index("theme")
    df["hot_rank"] = df["avg_return"].rank(ascending=False, method="min").astype(int)
    df.sort_values(["avg_return", "theme"], ascending=[False, True], inplace=True)

    logging.info("Computed theme stats:\n%s", df.to_string())
    return df

# =========================
# Auto-discovery helpers
# =========================
def _load_universe_csv(universe_path: str) -> pd.DataFrame:
    """Load a CSV with at least a 'symbol' column, optional 'name' column."""
    p = Path(universe_path)
    if not p.exists():
        logging.warning(f"Universe CSV not found at {universe_path}")
        return pd.DataFrame(columns=["symbol", "name"])
    try:
        dfu = pd.read_csv(p)
        dfu.columns = [c.strip().lower() for c in dfu.columns]
        if "symbol" not in dfu.columns:
            raise ValueError("universe.csv must have a 'symbol' column")
        if "name" not in dfu.columns:
            dfu["name"] = dfu["symbol"]
        dfu["symbol"] = dfu["symbol"].astype(str).str.upper().str.strip()
        dfu["name"] = dfu["name"].astype(str).str.strip()
        dfu = dfu.drop_duplicates(subset=["symbol"]).reset_index(drop=True)
        return dfu
    except Exception as e:
        logging.exception(f"Failed to load universe CSV: {e}")
        return pd.DataFrame(columns=["symbol", "name"])


def _keyword_label_for_group(names: List[str]) -> str:
    """Tiny heuristic labeler based on company names."""
    KEYWORDS = {
        "AI": ["ai", "machine learning"],
        "Data Centers": ["data center", "colocation", "hyperscale", "dc"],
        "Semis": ["semi", "semiconductor", "chip", "gpu", "cpu", "foundry"],
        "Quantum": ["quantum"],
        "BTC Miners/ Crypto": ["bitcoin", "btc", "crypto", "blockchain", "miner"],
        "Nuclear": ["nuclear", "uranium"],
        "Solar": ["solar", "photovoltaic", "pv"],
        "EV - Cars": ["ev", "electric vehicle"],
        "Lithium": ["lithium"],
        "Lidar": ["lidar"],
        "Rockets/Space": ["space", "rocket", "launch", "satellite"],
        "Cyber Sec": ["cyber", "security", "siem", "endpoint"],
        "Software": ["software", "saas", "cloud"],
        "Drones": ["drone", "uav"],
        "Finance/Banking": ["bank", "broker", "fintech", "payments", "visa", "mastercard"],
        "1D Gainers": ["holdings", "inc"],  # generic fallback for new names
    }
    lower_blob = " ".join(n.lower() for n in names)
    scores = {theme: sum(lower_blob.count(k) for k in kws) for theme, kws in KEYWORDS.items()}
    best = max(scores.items(), key=lambda kv: kv[1])
    return best[0] if best[1] > 0 else ""


def discover_new_themes(
    universe_csv: str,
    days: int = 20,
    corr_threshold: float = 0.6,
    min_cluster_size: int = 4,
    hot_threshold: float = 1.0,  # today’s avg % change across cluster
    top_n: int = 5,
) -> Dict[str, List[str]]:
    """
    Find co-moving clusters from a broad universe and keep ones that are 'hot' today.
    Returns {theme_name: [tickers...]} to merge into your themes.
    """
    dfu = _load_universe_csv(universe_csv)
    if dfu.empty:
        return {}

    tickers = dfu["symbol"].tolist()
    logging.info(f"Discover scan: fetching {len(tickers)} tickers for {days}d")
    prices = yf.download(
        tickers=tickers,
        period=f"{days}d",
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    if prices.empty:
        logging.warning("Discover scan: empty price frame")
        return {}

    # Normalize to Adj Close (fallback Close); columns=plain tickers
    if isinstance(prices.columns, pd.MultiIndex):
        cols = []
        for t in tickers:
            if (t, "Adj Close") in prices.columns:
                cols.append((t, "Adj Close"))
            elif (t, "Close") in prices.columns:
                cols.append((t, "Close"))
        adj = prices[cols].copy()
        adj.columns = [c[0] for c in adj.columns]
    else:
        adj = prices[["Adj Close"]] if "Adj Close" in prices.columns else prices[["Close"]]
        adj.columns = [tickers[0]]

    # Clean up empty columns before computing returns
    adj = adj.dropna(axis=1, how="all")
    if len(adj) < 3:
        return {}

    # Use fill_method=None to avoid FutureWarning and preserve NAs
    rets = adj.pct_change(fill_method=None).iloc[1:] * 100.0
    rets = rets.dropna(axis=1, how="all")
    if rets.empty:
        return {}

    today = rets.iloc[-1].dropna()
    corr = rets.corr(min_periods=3)

    visited = set()
    clusters: List[List[str]] = []

    def dfs(node, comp):
        visited.add(node)
        comp.append(node)
        row = corr.loc[node]
        neighbors = row.index[(row >= corr_threshold) & (row.index != node)]
        for nb in neighbors:
            if nb not in visited:
                dfs(nb, comp)

    for t in corr.columns:
        if t not in visited and t in today.index:
            comp = []
            dfs(t, comp)
            if len(comp) >= min_cluster_size:
                clusters.append(sorted(set(comp)))

    new_themes: Dict[str, List[str]] = {}
    auto_id = 1
    for comp in clusters:
        comp_today = today.loc[[t for t in comp if t in today.index]]
        if comp_today.empty:
            continue
        if comp_today.mean() >= hot_threshold:
            # Build a robust list of names with fallback to ticker symbol
            name_series = dfu.set_index("symbol").reindex(comp)["name"]
            names = [
                n if (pd.notna(n) and str(n).strip()) else t
                for n, t in zip(name_series.tolist(), comp)
            ]
            label = _keyword_label_for_group(names) or f"Cluster {auto_id}"
            theme_name = f"🔶 NEW: {label}"
            auto_id += 1
            top = comp_today.sort_values(ascending=False).head(top_n).index.tolist()
            new_themes[theme_name] = top

    logging.info(f"Discover scan found {len(new_themes)} candidate themes")
    return new_themes


# =========================
# Theme rank-delta helpers (arrows)
# =========================
def _theme_strength(rets_window: pd.DataFrame, themes: Dict[str, List[str]]) -> pd.Series:
    """Mean per ticker across days, then mean across tickers within each theme."""
    per_ticker = rets_window.mean(axis=0, skipna=True)  # Series by ticker
    out = {}
    for theme, tickers in themes.items():
        tickers_in = [t for t in tickers if t in per_ticker.index]
        out[theme] = float(per_ticker.loc[tickers_in].mean()) if tickers_in else float("nan")
    return pd.Series(out, name="strength")


def _theme_rank_arrows(prices_last_2w: pd.DataFrame, themes: Dict[str, List[str]], days: int) -> pd.Series:
    """
    From ~2x 'days' worth of prices, compute rank delta per theme and map to ▲/▼/→ with magnitude.
    Returns a Series indexed by theme containing strings like '▲2', '▼1', '→0'.
    """
    if prices_last_2w.empty or len(prices_last_2w) < max(days*2 - 1, 7):
        return pd.Series(dtype="object")

    rets = prices_last_2w.pct_change(fill_method=None) * 100.0
    curr_rets = rets.iloc[-days:]           # current window
    prev_rets = rets.iloc[-(2*days): -days] # previous window

    curr_strength = _theme_strength(curr_rets, themes)
    prev_strength = _theme_strength(prev_rets, themes)

    curr_rank = curr_strength.rank(ascending=False, method="min")
    prev_rank = prev_strength.rank(ascending=False, method="min")

    delta = (prev_rank - curr_rank).reindex(curr_rank.index)  # positive = moved up
    def arrow_mag(d):
        if pd.isna(d): return "→0"
        d = int(d)
        if d > 0:  return f"▲{d}"
        if d < 0:  return f"▼{abs(d)}"
        return "→0"
    return delta.apply(arrow_mag)


# =========================
# Public API
# =========================
def get_hot_themes(days: int = DEFAULT_DAYS, discover: bool = True) -> pd.DataFrame:
    """
    Get ranked theme DataFrame; optionally include auto-discovered clusters,
    add theme arrows, and prepare hover strings for top 4 movers (today).
    """
    logging.info("Starting get_hot_themes")
    themes = load_themes()

    # Auto-discover co-moving, hot clusters from the larger universe
    if discover:
        auto_themes = discover_new_themes(
            universe_csv=UNIVERSE_CSV,
            days=max(days, 20),
            corr_threshold=0.6,
            min_cluster_size=4,
            hot_threshold=1.0,
            top_n=5,
        )
        for k, v in auto_themes.items():
            k2 = k if k not in themes else f"{k} (Auto)"
            themes[k2] = v

    if not themes:
        return pd.DataFrame()

    # 1) Fetch enough history for both current stats and arrows
    prices_2w = fetch_prices_all(themes, days=max(days*2, 8))
    if prices_2w.empty:
        print("No market data fetched.")
        logging.error("No market data fetched.")
        return pd.DataFrame()

    # 2) Current window stats (preserve your original metric over 'days')
    prices_curr = prices_2w.tail(days)
    df = compute_theme_stats(themes, prices_curr)
    if df.empty:
        print("No data to display after computation.")
        return df

    # 3) Theme rank change arrows from prior vs current windows
    arrows = _theme_rank_arrows(prices_2w, themes, days=days)

    # 4) Build per-theme hover strings with top 4 tickers by today's % change (1 decimal)
    daily_pct_curr = prices_curr.pct_change(fill_method=None) * 100.0
    today_pct = daily_pct_curr.iloc[-1] if len(daily_pct_curr) else pd.Series(dtype=float)

    def hover_for(theme: str) -> str:
        if today_pct.empty or theme not in themes:
            return "Top movers (today): n/a"
        tickers_in = [t for t in themes[theme] if t in today_pct.index]
        if not tickers_in:
            return "Top movers (today): n/a"
        s = today_pct.loc[tickers_in].dropna().sort_values(ascending=False).head(4)
        if s.empty:
            return "Top movers (today): n/a"
        parts = [f"{t} {v:+.1f}%" for t, v in s.items()]
        return "Top movers (today): " + ", ".join(parts)

    # 5) Decorate with labels for plotting + hover strings
    def label_for(theme: str) -> str:
        a = arrows.get(theme, "→0")
        return f"{theme}  {a}"

    df = df.copy()
    df["label"] = [label_for(t) for t in df.index]
    df["hover"] = [hover_for(t) for t in df.index]
    return df

# =========================
# GUI
# =========================
class ThemeTrackerApp(QMainWindow):
    """Main application window displaying the themes chart."""
    def __init__(self, df_themes: pd.DataFrame, days: int = DEFAULT_DAYS):
        super().__init__()
        self.setWindowTitle("CC Themes")
        self.setGeometry(100, 100, 900, 600)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        if df_themes.empty:
            msg = QLabel("No data to display")
            msg.setAlignment(Qt.AlignCenter)
            layout.addWidget(msg)
            return

        # Build Plotly figure
        fig = px.bar(
            df_themes,
            y=df_themes.index,          # keep real index for ordering
            x="avg_return",
            title=f"Themes ({days}-Day Avg % Change)",
            labels={"y": "Theme", "avg_return": f"{days}-Day Avg %"},
            color="avg_return",
            color_continuous_scale="Viridis",
            orientation="h"
        )
        fig.update_layout(title={'text': f'Themes ({days}-Day Avg % Change)', 'x': 0.5, 'xanchor': 'center'},
            showlegend=False,
            margin=dict(l=60, r=30, t=60, b=40),
            xaxis_title="Avg % Change",
            yaxis_title="Theme"
        )        
    
        # Keep sort & place hottest at the TOP
        fig.update_yaxes(categoryorder="array", categoryarray=df_themes.index.tolist())
        fig.update_yaxes(autorange="reversed")  # first row at TOP
        # Swap visible tick text to include ▲/▼/→ labels (and 🔶 NEW: badges when present)
        fig.update_yaxes(
            tickmode="array",
            tickvals=df_themes.index.tolist(),
            ticktext=df_themes["label"].tolist()
        )
        # Visual split at 0%
        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)

        # Enhanced hover: include top 4 movers (today) with one-decimal %
        fig.update_traces(
            customdata=df_themes["hover"],
            hovertemplate="<b>%{y}</b><br>Avg: %{x:.1f}%<br>%{customdata}<extra></extra>"
        )

        try:
            html_content = fig.to_html(full_html=False, include_plotlyjs="cdn")
            self.web_view = QWebEngineView()
            self.web_view.setHtml(html_content)
            layout.addWidget(self.web_view)
            logging.info("Rendered Plotly chart in QWebEngineView")
        except Exception as e:
            logging.error(f"Failed to render QWebEngineView: {e}")
            label = QLabel("Failed to render chart")
            label.setAlignment(Qt.AlignCenter)
            layout.addWidget(label)

    def closeEvent(self, event):
        event.accept()
        QApplication.quit()

# =========================
# Entrypoint
# =========================
def main():
    # Usage:
    #   python tracker.py              -> discovery ON
    #   python tracker.py 7            -> 7-day window, discovery ON
    #   python tracker.py --no-discover
    #   python tracker.py 7 --no-discover
    days = DEFAULT_DAYS
    discover_flag = True

    for arg in sys.argv[1:]:
        if arg.isdigit():
            days = int(arg)
        elif arg == "--no-discover":
            discover_flag = False

    print(f"Running main once (days={days}, discover={'ON' if discover_flag else 'OFF'})")
    df = get_hot_themes(days=days, discover=discover_flag)
    if df.empty:
        sys.exit(1)

    app = QApplication(sys.argv)
    window = ThemeTrackerApp(df, days=days)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
