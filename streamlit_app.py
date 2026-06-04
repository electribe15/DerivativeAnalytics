"""
DEX / GEX Options Exposure Dashboard — Streamlit version
Deploy su https://share.streamlit.io puntando a questo file.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import time

import streamlit as st
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DEX / GEX Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Share Tech Mono', monospace;
    background-color: #0d0f14;
    color: #c9d1e0;
}
.stApp { background-color: #0d0f14; }

section[data-testid="stSidebar"] {
    background-color: #13161e;
    border-right: 1px solid #1e2130;
}

/* Metric cards — both old and new Streamlit testid names */
div[data-testid="metric-container"],
div[data-testid="stMetric"] {
    background: #13161e;
    border: 1px solid #1e2130;
    border-top: 3px solid #00e5a0;
    border-radius: 4px;
    padding: 12px 16px;
}
div[data-testid="metric-container"] label,
div[data-testid="stMetric"] label {
    color: #7a8399 !important;
    font-size: 11px !important;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"],
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #00e5a0 !important;
    font-size: 20px !important;
    font-family: 'Share Tech Mono', monospace !important;
}

.stButton > button {
    background: transparent;
    border: 1px solid #00e5a0;
    color: #00e5a0;
    font-family: 'Share Tech Mono', monospace;
    letter-spacing: 1px;
    border-radius: 4px;
    width: 100%;
}
.stButton > button:hover {
    background: #00e5a020;
    border-color: #00e5a0;
    color: #00e5a0;
}

h1 {
    font-family: 'Rajdhani', sans-serif !important;
    color: #00e5a0 !important;
    letter-spacing: 6px;
    font-size: 2.2rem !important;
}
h2, h3 {
    font-family: 'Rajdhani', sans-serif !important;
    color: #c9d1e0 !important;
    letter-spacing: 2px;
}

.status-bar {
    background: #13161e;
    border: 1px solid #1e2130;
    border-left: 3px solid #4db8ff;
    border-radius: 4px;
    padding: 8px 14px;
    font-size: 12px;
    color: #7a8399;
    margin-bottom: 16px;
    letter-spacing: 0.5px;
}

div[data-baseweb="tab-list"] {
    background: #13161e;
    border-bottom: 1px solid #1e2130;
}
button[data-baseweb="tab"] {
    color: #7a8399 !important;
    font-family: 'Share Tech Mono', monospace !important;
    letter-spacing: 1px;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #00e5a0 !important;
    border-bottom: 2px solid #00e5a0 !important;
}

.stSpinner > div { border-top-color: #00e5a0 !important; }
div[data-testid="stSlider"] label { color: #7a8399 !important; font-size: 12px; }
input[type="text"] {
    background: #1a1d27 !important;
    border: 1px solid #1e2130 !important;
    color: #c9d1e0 !important;
    font-family: 'Share Tech Mono', monospace !important;
}
</style>
""", unsafe_allow_html=True)

# ── Imports from backend ──────────────────────────────────────────────────────
try:
    from dex_gex_dashboard import (
        fetch_options_data,
        aggregate_by_strike,
        aggregate_by_expiry,
        apply_dashboard_filters,
        _csv_paths,
        gex_bar_chart,
        dex_bar_chart,
        gex_expiry_chart,
        oi_heatmap,
        vol_smile_chart,
        iv_surface_chart,
        term_structure_chart,
        skew_chart,
        DEFAULT_TICKER,
        OI_THRESHOLD,
        MAX_EXPIRY_DAYS,
        FETCH_EXPIRY_DAYS,
        ACCENT_GRN,
        ACCENT_RED,
        ACCENT_BLU,
        ACCENT_YLW,
    )
except ImportError as e:
    st.error(f"⚠️ Errore importando dex_gex_dashboard.py: {e}")
    st.stop()

# ── Helpers ───────────────────────────────────────────────────────────────────
def fmt_billions(v):
    sign = "+" if v >= 0 else ""
    return f"{sign}${v/1e9:.2f}B"

def fmt_millions(v):
    sign = "+" if v >= 0 else ""
    return f"{sign}${v/1e6:.0f}M"

def build_stat_metrics(raw_df, by_strike_df, spot):
    total_gex   = raw_df["gex"].sum()
    total_dex   = raw_df["dex"].sum()
    peak_strike = by_strike_df.loc[by_strike_df["net_gex"].abs().idxmax(), "strike"]
    call_oi     = int(raw_df[raw_df["flag"] == "c"]["openInterest"].sum())
    put_oi      = int(raw_df[raw_df["flag"] == "p"]["openInterest"].sum())
    pcr         = put_oi / call_oi if call_oi else 0
    regime      = "🟢 POSITIVE" if total_gex > 0 else "🔴 NEGATIVE"
    return {
        "Spot Price":      (f"${spot:.2f}",          ACCENT_YLW),
        "Total GEX":       (fmt_billions(total_gex),  ACCENT_GRN if total_gex > 0 else ACCENT_RED),
        "Total DEX":       (fmt_millions(total_dex),  ACCENT_BLU),
        "GEX Regime":      (regime,                   ACCENT_GRN if total_gex > 0 else ACCENT_RED),
        "Peak GEX Strike": (str(peak_strike),         ACCENT_GRN),
        "Put/Call OI":     (f"{pcr:.2f}",             ACCENT_YLW),
        "Contracts":       (f"{len(raw_df):,}",       ACCENT_BLU),
        "Expiries":        (str(raw_df["expiry"].nunique()), ACCENT_YLW),
    }

def empty_fig(msg="No data"):
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor="#13161e", plot_bgcolor="#13161e",
        font=dict(color="#c9d1e0"),
        annotations=[dict(text=msg, xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False,
                          font=dict(size=14, color="#7a8399"))],
        height=380,
    )
    return fig

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ DEX / GEX")
    st.markdown("---")

    ticker     = st.text_input("Ticker", value=DEFAULT_TICKER).strip().upper()
    window_pct = st.slider("Finestra strike (%)", min_value=5, max_value=25,
                           value=10, step=5) / 100
    max_days   = st.slider("Max giorni scadenza", min_value=14,
                           max_value=FETCH_EXPIRY_DAYS,   # up to the full download horizon
                           value=MAX_EXPIRY_DAYS, step=7)
    oi_thresh  = st.number_input("Min Open Interest", min_value=10,
                                 max_value=500, value=OI_THRESHOLD, step=10)

    st.markdown("---")
    fetch_btn = st.button("↻ CARICA DATI", use_container_width=True)

    st.markdown("---")
    st.markdown(
        "<div style='font-size:11px;color:#7a8399;'>"
        "Dati: Barchart<br>"
        "Modello: Black-Scholes<br>"
        "DEX = Δ × OI × 100<br>"
        "GEX = γ × OI × 100 × Spot"
        "</div>",
        unsafe_allow_html=True,
    )

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# ⚡ DEX / GEX")
st.markdown(
    "<p style='color:#7a8399;font-size:13px;letter-spacing:3px;"
    "margin-top:-12px;margin-bottom:20px;'>OPTIONS EXPOSURE DASHBOARD</p>",
    unsafe_allow_html=True,
)

# ── Fetch ─────────────────────────────────────────────────────────────────────
# Only fetch on explicit button press — never auto-load on page open.
# force_refresh=True bypasses the CSV cache so the button always hits Barchart.
if fetch_btn:
    progress_bar = st.progress(0, text="⏳ Avvio…")
    csv_slot     = st.empty()
    last_msg     = {"text": ""}

    def _report(pct, msg):
        last_msg["text"] = msg
        try:
            progress_bar.progress(min(max(int(pct), 0), 100) / 100.0, text=msg)
        except Exception:
            pass

    try:
        raw_full, spot = fetch_options_data(
            ticker,
            progress_callback=_report,
            force_refresh=True,
        )

        # Apply sidebar filters (DTE cap + min OI) independently of the download
        raw_df = apply_dashboard_filters(raw_full, dte_max=max_days, oi_min=oi_thresh)
        if raw_df.empty:
            progress_bar.empty()
            st.warning(
                f"Nessun contratto per {ticker} con DTE ≤ {max_days} e OI ≥ {oi_thresh}. "
                "Allenta i filtri nella sidebar."
            )
            st.stop()

        by_strike = aggregate_by_strike(raw_df)
        by_expiry = aggregate_by_expiry(raw_df)

        st.session_state["data"] = {
            "raw":       raw_df,
            "by_strike": by_strike,
            "by_expiry": by_expiry,
            "spot":      spot,
            "ticker":    ticker,
        }

        # CSV cache status — background thread writes it after fetch returns
        try:
            csv_path, _ = _csv_paths(ticker)
            waited = 0.0
            while waited < 3.0 and not (
                os.path.exists(csv_path)
                and (time.time() - os.path.getmtime(csv_path)) < 120
            ):
                time.sleep(0.25)
                waited += 0.25
            if os.path.exists(csv_path) and (time.time() - os.path.getmtime(csv_path)) < 120:
                size_kb = os.path.getsize(csv_path) / 1024
                csv_slot.caption(f"💾 CSV cache salvata → {csv_path}  ({size_kb:,.0f} KB)")
            else:
                csv_slot.caption(f"💾 {last_msg['text']}")
        except Exception:
            pass

        progress_bar.progress(1.0, text="✅ Completato")
        progress_bar.empty()

    except Exception as err:
        progress_bar.empty()
        st.error(f"⚠️ Errore: {err}")
        st.stop()

# ── Idle state ────────────────────────────────────────────────────────────────
if "data" not in st.session_state:
    st.info("👈 Inserisci un ticker nella sidebar e premi **↻ CARICA DATI** per iniziare.")
    st.stop()

# ── Read session data ─────────────────────────────────────────────────────────
d         = st.session_state["data"]
raw_df    = d["raw"]
by_strike = d["by_strike"]
by_expiry = d["by_expiry"]
spot      = d["spot"]
cur_tick  = d["ticker"]

# ── Status bar ────────────────────────────────────────────────────────────────
from datetime import datetime
ts = datetime.now().strftime("%H:%M:%S")
st.markdown(
    f"<div class='status-bar'>Last updated: {ts} &nbsp;|&nbsp; "
    f"{len(raw_df):,} contracts &nbsp;|&nbsp; "
    f"{raw_df['expiry'].nunique()} expiries &nbsp;|&nbsp; "
    f"Spot <b style='color:#ffd166'>${spot:.2f}</b></div>",
    unsafe_allow_html=True,
)

# ── Metric cards ──────────────────────────────────────────────────────────────
stats = build_stat_metrics(raw_df, by_strike, spot)
cols  = st.columns(len(stats))
for col, (label, (value, _color)) in zip(cols, stats.items()):
    col.metric(label=label, value=value)

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 GEX / DEX", "📈 Volatilità", "🔥 Open Interest"])

# ── Tab 1: GEX / DEX ─────────────────────────────────────────────────────────
with tab1:
    col_l, col_r = st.columns(2)
    with col_l:
        try:
            st.plotly_chart(gex_bar_chart(by_strike, spot, cur_tick, window_pct),
                            use_container_width=True)
        except Exception as e:
            st.plotly_chart(empty_fig(str(e)), use_container_width=True)

    with col_r:
        try:
            st.plotly_chart(dex_bar_chart(by_strike, spot, cur_tick, window_pct),
                            use_container_width=True)
        except Exception as e:
            st.plotly_chart(empty_fig(str(e)), use_container_width=True)

    try:
        st.plotly_chart(gex_expiry_chart(by_expiry, cur_tick), use_container_width=True)
    except Exception as e:
        st.plotly_chart(empty_fig(str(e)), use_container_width=True)

# ── Tab 2: Volatilità ─────────────────────────────────────────────────────────
with tab2:
    col_a, col_b = st.columns(2)
    with col_a:
        try:
            st.plotly_chart(vol_smile_chart(raw_df, spot, cur_tick),
                            use_container_width=True)
        except Exception as e:
            st.plotly_chart(empty_fig(str(e)), use_container_width=True)

    with col_b:
        try:
            st.plotly_chart(term_structure_chart(raw_df, spot, cur_tick),
                            use_container_width=True)
        except Exception as e:
            st.plotly_chart(empty_fig(str(e)), use_container_width=True)

    try:
        st.plotly_chart(skew_chart(raw_df, spot, cur_tick), use_container_width=True)
    except Exception as e:
        st.plotly_chart(empty_fig(str(e)), use_container_width=True)

    st.markdown("#### 🌐 IV Surface (3D)")
    expiry_options   = sorted(raw_df["expiry"].unique())
    selected_expiries = st.multiselect(
        "Seleziona scadenze per la superficie IV",
        options=expiry_options,
        default=expiry_options[:6],
    )
    try:
        st.plotly_chart(
            iv_surface_chart(raw_df, spot, cur_tick, expiries_list=selected_expiries),
            use_container_width=True,
        )
    except Exception as e:
        st.plotly_chart(empty_fig(str(e)), use_container_width=True)

# ── Tab 3: Open Interest ──────────────────────────────────────────────────────
with tab3:
    try:
        st.plotly_chart(oi_heatmap(raw_df, spot, cur_tick, window_pct),
                        use_container_width=True)
    except Exception as e:
        st.plotly_chart(empty_fig(str(e)), use_container_width=True)

    with st.expander("📋 Dati per strike (tabella)"):
        cols_to_show = ["strike", "net_gex", "net_dex", "call_gex", "put_gex",
                        "call_dex", "put_dex", "total_oi"]
        available = [c for c in cols_to_show if c in by_strike.columns]
        st.dataframe(
            by_strike[available]
            .sort_values("net_gex", ascending=False)
            .reset_index(drop=True),
            use_container_width=True,
        )
