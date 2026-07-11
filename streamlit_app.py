"""
DEX / GEX Options Exposure Dashboard — Streamlit version
"""

import warnings
warnings.filterwarnings("ignore")

import os
import time
from datetime import datetime

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

APP_VERSION = "2.2.0"
APP_BUILD   = "2026-06-16"

st.set_page_config(
    page_title="DEX / GEX Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── Base ───────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    background-color: #F8F9FD; color: #374151;
}
.stApp { background-color: #F8F9FD; }

/* ── Sidebar ─────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1E1B4B 0%, #312E81 100%);
    border-right: none;
    box-shadow: 4px 0 20px rgba(0,0,0,0.15);
}
section[data-testid="stSidebar"] * { color: #E0E7FF !important; }
section[data-testid="stSidebar"] h2 { color: #FFFFFF !important; font-size: 1.1rem !important; letter-spacing: 0.5px; }
section[data-testid="stSidebar"] label { color: #A5B4FC !important; font-size: 12px !important; font-weight: 500; letter-spacing: 0.3px; text-transform: uppercase; }
section[data-testid="stSidebar"] .stMarkdown p { color: #818CF8 !important; font-size: 11px; }
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.1) !important; }

/* ── Buttons ─────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #6C63FF 0%, #8B5CF6 100%);
    color: #FFFFFF !important; border: none; border-radius: 8px;
    font-weight: 600; font-size: 13px; width: 100%;
    box-shadow: 0 4px 12px rgba(108,99,255,0.35);
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #5B54EE 0%, #7C3AED 100%);
    box-shadow: 0 6px 16px rgba(108,99,255,0.45); transform: translateY(-1px);
}

/* ── Metric cards ────────────────────────────────────── */
div[data-testid="metric-container"], div[data-testid="stMetric"] {
    background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 12px; border-top: none;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06), 0 4px 12px rgba(0,0,0,0.04); padding: 14px 18px;
}
div[data-testid="metric-container"] label, div[data-testid="stMetric"] label {
    color: #9CA3AF !important; font-size: 11px !important; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.8px;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"],
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #111827 !important; font-size: 20px !important; font-weight: 700;
    font-family: 'Inter', sans-serif !important;
    white-space: normal !important; overflow: visible !important;
    text-overflow: clip !important;
}
div[data-testid="metric-container"] [data-testid="stMetricLabel"],
div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
    white-space: normal !important; overflow: visible !important;
}

/* ── Tabs ────────────────────────────────────────────── */
div[data-baseweb="tab-list"] {
    background: #FFFFFF; border-radius: 10px; padding: 4px;
    border-bottom: none; gap: 2px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
/* Unselected tabs: grey text. Broad selectors so the rule survives Streamlit
   DOM changes — target the tab button and every descendant text node. */
div[data-baseweb="tab-list"] button[data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important; font-weight: 500; border-radius: 7px; padding: 6px 14px;
}
div[data-baseweb="tab-list"] button[data-baseweb="tab"],
div[data-baseweb="tab-list"] button[data-baseweb="tab"] *,
div[data-baseweb="tab-list"] button[aria-selected="false"],
div[data-baseweb="tab-list"] button[aria-selected="false"] * {
    color: #374151 !important; -webkit-text-fill-color: #374151 !important;
}
/* Selected tab: white text on the purple gradient. */
div[data-baseweb="tab-list"] button[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #6C63FF 0%, #8B5CF6 100%) !important;
    border: none !important;
    box-shadow: 0 2px 8px rgba(108,99,255,0.4);
}
div[data-baseweb="tab-list"] button[aria-selected="true"],
div[data-baseweb="tab-list"] button[aria-selected="true"] * {
    color: #FFFFFF !important; -webkit-text-fill-color: #FFFFFF !important;
}

/* ── Typography ──────────────────────────────────────── */
h1 { font-family: 'Inter', sans-serif !important; color: #111827 !important;
     font-size: 1.9rem !important; font-weight: 700 !important; letter-spacing: -0.5px; }
h2, h3, h4, h5 { font-family: 'Inter', sans-serif !important; color: #1F2937 !important; font-weight: 600 !important; }

/* ── Status bar ──────────────────────────────────────── */
.status-bar {
    background: #FFFFFF; border: 1px solid #E5E7EB; border-left: 4px solid #6C63FF;
    border-radius: 10px; padding: 10px 16px; font-size: 12px; color: #6B7280;
    margin-bottom: 16px; box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}

/* ── Plotly charts — card wrapper ────────────────────── */
div[data-testid="stPlotlyChart"] {
    background: #FFFFFF; border-radius: 12px; border: 1px solid #E5E7EB;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06); padding: 4px; overflow: hidden;
}

/* ── Misc ────────────────────────────────────────────── */
.stSpinner > div { border-top-color: #6C63FF !important; }
div[data-testid="stSlider"] label { color: #6B7280 !important; font-size: 12px; font-weight: 500; }
input[type="text"], input[type="number"] {
    background: #F9FAFB !important; border: 1.5px solid #E5E7EB !important;
    border-radius: 8px !important; color: #374151 !important;
    font-family: 'Inter', sans-serif !important;
}
details { border-radius: 10px !important; border: 1px solid #E5E7EB !important; }
div[data-testid="stProgress"] > div > div { background: #6C63FF !important; border-radius: 8px !important; }
hr { border-color: #E5E7EB !important; }
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
        SNAPSHOT_DIR,
        delta_strike_bounds,
        compute_0dte_metrics,
        compute_max_pain,
        compute_pc_ratio,
        compute_har_rv,
        detect_vol_regime,
        build_alert_flags,
        backtest_har_oos,
        backtest_vol_premium,
        compute_gex_analytics,
        compute_gex_profile,
        compute_voldex,
        save_voldex_snapshot,
        load_voldex_history,
        voldex_history_chart,
        save_gex_metrics_snapshot,
        load_gex_metrics_history,
        gex_metrics_history_chart,
        cumulative_gex_chart,
        compute_key_levels,
        gex_regime_narrative,
        signal_health_check,
        compute_0dte_gamma_schedule,
        save_daily_snapshot,
        compute_dod_changes,
        compute_gex_percentile,
        compute_flip_by_expiry,
        generate_morning_brief,
        fetch_price_history,
        fetch_intraday_history,
        compute_value_area,
        value_area_position,
        value_area_chart,
        fetch_ohlc_history,
        fetch_vix_history,
        compute_rvol_all,
        compute_rvol_cones,
        compute_intraday_rvol,
        rvol_models_chart,
        rvol_cones_chart,
        get_put_monitor,
        price_vs_dex_chart,
        gex_dex_0dte_chart,
        oi_0dte_chart,
        smile_0dte_chart,
        gex_bar_chart,
        dex_bar_chart,
        gex_expiry_chart,
        oi_heatmap,
        vol_smile_chart,
        iv_surface_chart,
        iv_skew_overlay_chart,
        daily_range_chart,
        put_monitor_chart,
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
    return f"{'+'if v>=0 else ''}${v/1e9:.2f}B"

def fmt_millions(v):
    return f"{'+'if v>=0 else ''}${v/1e6:.0f}M"

def build_stat_metrics(raw_df, by_strike_df, spot):
    total_gex   = raw_df["gex"].sum()
    total_dex   = raw_df["dex"].sum()
    peak_strike = by_strike_df.loc[by_strike_df["net_gex"].abs().idxmax(), "strike"]
    call_oi     = int(raw_df[raw_df["flag"] == "c"]["openInterest"].sum())
    put_oi      = int(raw_df[raw_df["flag"] == "p"]["openInterest"].sum())
    pcr         = put_oi / call_oi if call_oi else 0
    # Flip-based regime (standard of market): set by spot vs the gamma flip
    # strike, not by the raw sign of total GEX. Consistent with the 0DTE tab.
    _ga = compute_gex_analytics(raw_df, by_strike_df, spot) or {}
    _flip = _ga.get('gamma_flip')
    _reg  = _ga.get('regime')
    if _reg and 'LONG' in _reg:
        regime, regime_pos = "🟢 LONG γ", True
    elif _reg and 'SHORT' in _reg:
        regime, regime_pos = "🔴 SHORT γ", False
    else:
        regime_pos = total_gex > 0
        regime = "🟢 LONG γ" if regime_pos else "🔴 SHORT γ"
    flip_str = f"${_flip:,.0f}" if _flip else "—"
    return {
        "Spot Price":      (f"${spot:.2f}",          ACCENT_YLW),
        "Total GEX":       (fmt_billions(total_gex),  ACCENT_GRN if total_gex > 0 else ACCENT_RED),
        "Total DEX":       (fmt_millions(total_dex),  ACCENT_BLU),
        "GEX Regime (full chain)": (regime,           ACCENT_GRN if regime_pos else ACCENT_RED),
        "Gamma Flip":      (flip_str,                 ACCENT_YLW),
        "Peak GEX Strike": (str(peak_strike),         ACCENT_GRN),
        "Put/Call OI":     (f"{pcr:.2f}",             ACCENT_YLW),
        "Contracts":       (f"{len(raw_df):,}",       ACCENT_BLU),
        "Expiries":        (str(raw_df["expiry"].nunique()), ACCENT_YLW),
    }

def empty_fig(msg="No data"):
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor="#13161e", plot_bgcolor="#13161e", font=dict(color="#c9d1e0"),
        annotations=[dict(text=msg, xref="paper", yref="paper", x=0.5, y=0.5,
                          showarrow=False, font=dict(size=14, color="#7a8399"))],
        height=380,
    )
    return fig

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## DEX / GEX")
    st.markdown("<p style='color:#818CF8;font-size:11px;margin-top:-8px;'>Options Analytics</p>",
                unsafe_allow_html=True)
    st.markdown("---")

    ticker      = st.text_input("Ticker", value=DEFAULT_TICKER).strip().upper()

    # Delta range slider — maps to OTM strike range on the charts.
    # (-0.20, +0.20) = show from 20-delta put to 20-delta call (~±2-3% of spot).
    # Smaller absolute value → wider range (more OTM).
    # Larger absolute value → narrower range (near-ATM only).
    delta_range = st.slider(
        "Delta range (strike filter)",
        min_value=-1.0, max_value=1.0,
        value=(-0.20, 0.20), step=0.05,
        help="Mostra i strike tra la put con Δ≈min e la call con Δ≈max "
             "(calcolato dalla scadenza più vicina). "
             "Valore assoluto più basso = range più ampio (opzioni OTM).",
    )

    # DTE slider — min starts from the shortest available expiry in the loaded chain
    _min_dte = 0
    if "data" in st.session_state:
        _rf = st.session_state["data"].get("raw_full")
        if _rf is not None and not _rf.empty:
            _min_dte = max(0, int(_rf["T_days"].min()))

    max_days = st.slider(
        "Max giorni scadenza",
        min_value=_min_dte,
        max_value=FETCH_EXPIRY_DAYS,
        value=max(_min_dte, min(MAX_EXPIRY_DAYS, FETCH_EXPIRY_DAYS)),
        step=1,
    )
    oi_thresh   = st.number_input("Min Open Interest", min_value=10,
                                  max_value=500, value=OI_THRESHOLD, step=10)

    st.markdown("---")
    col_f, col_a = st.columns(2)
    with col_f:
        fetch_btn = st.button("⬇ CARICA FULL CHAIN", use_container_width=True)
    with col_a:
        apply_btn = st.button(
            "⊞ APPLICA FILTRI",
            use_container_width=True,
            disabled="data" not in st.session_state,
            help="Ricalcola i grafici applicando i filtri DTE e OI senza ri-scaricare i dati.",
        )

    st.markdown("---")
    st.markdown(
        "<div style='font-size:11px;color:#818CF8;line-height:1.8;'>"
        "📡 Dati: Barchart<br>🧮 Modello: Black-Scholes<br>"
        "DEX = Δ × OI × 100<br>GEX = γ × OI × 100 × Spot"
        "</div>", unsafe_allow_html=True,
    )

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# DEX / GEX")
st.markdown(
    "<p style='color:#9CA3AF;font-size:13px;font-weight:500;letter-spacing:0.5px;"
    "margin-top:-10px;margin-bottom:20px;'>Options Exposure Dashboard</p>",
    unsafe_allow_html=True,
)

# ── Fetch ─────────────────────────────────────────────────────────────────────
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
        raw_full, spot = fetch_options_data(ticker, progress_callback=_report, force_refresh=True)
        raw_df = apply_dashboard_filters(raw_full, dte_max=max_days, oi_min=oi_thresh)
        if raw_df.empty:
            progress_bar.empty()
            st.warning(f"Nessun contratto per {ticker} con DTE ≤ {max_days} e OI ≥ {oi_thresh}.")
            st.stop()

        by_strike = aggregate_by_strike(raw_df)
        by_expiry = aggregate_by_expiry(raw_df)

        # Fetch today's intraday bars — lightweight, non-fatal if unavailable
        with st.spinner("⏳ Dati intraday…"):
            intraday = fetch_intraday_history(ticker, interval_min=5)

        # Fetch 6-month daily OHLC (SPX) + VIX for Realized Vol tab
        with st.spinner("⏳ Storico SPX + VIX (6 mesi)…"):
            ohlc_spx  = fetch_ohlc_history('$SPX', n_calendar_days=550)  # 18 mo: 6mo window + 12mo display
            vix_hist  = fetch_vix_history(n_calendar_days=420)            # 14 mo: margin for 1-year display

        st.session_state["data"] = dict(raw=raw_df, by_strike=by_strike,
                                         by_expiry=by_expiry, spot=spot,
                                         ticker=ticker, raw_full=raw_full,
                                         intraday=intraday,
                                         ohlc_spx=ohlc_spx,
                                         vix_hist=vix_hist,
                                         fetched_at=datetime.now())

        # Persist a compact daily snapshot (enables DoD deltas + GEX percentile)
        try:
            save_daily_snapshot(raw_full, spot, ticker)
        except Exception:
            pass

        # CSV cache confirmation
        try:
            csv_path, _ = _csv_paths(ticker)
            waited = 0.0
            while waited < 3.0 and not (
                os.path.exists(csv_path) and (time.time() - os.path.getmtime(csv_path)) < 120
            ):
                time.sleep(0.25); waited += 0.25
            if os.path.exists(csv_path) and (time.time() - os.path.getmtime(csv_path)) < 120:
                csv_slot.caption(f"💾 CSV → {csv_path}  ({os.path.getsize(csv_path)/1024:,.0f} KB)")
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

# ── Apply filters (no Barchart call — re-slices the already-loaded full chain) ──
elif apply_btn and "data" in st.session_state:
    raw_full = st.session_state["data"].get("raw_full")
    if raw_full is None:
        st.warning("Chain non disponibile: premi prima **⬇ CARICA FULL CHAIN**.")
    else:
        with st.spinner("⏳ Applicazione filtri…"):
            raw_df = apply_dashboard_filters(raw_full, dte_max=max_days, oi_min=oi_thresh)
        if raw_df.empty:
            st.warning(
                f"Nessun contratto con DTE ≤ {max_days} e OI ≥ {oi_thresh}. "
                "Allenta i filtri."
            )
        else:
            by_strike = aggregate_by_strike(raw_df)
            by_expiry = aggregate_by_expiry(raw_df)
            st.session_state["data"]["raw"]       = raw_df
            st.session_state["data"]["by_strike"] = by_strike
            st.session_state["data"]["by_expiry"] = by_expiry
            st.success(
                f"Filtri applicati: {len(raw_df):,} contratti  |  "
                f"{raw_df['expiry'].nunique()} scadenze  |  "
                f"DTE ≤ {max_days}  |  OI ≥ {oi_thresh}"
            )

# ── Idle state ────────────────────────────────────────────────────────────────
if "data" not in st.session_state:
    st.info("👈 Inserisci un ticker e premi **↻ CARICA DATI** per iniziare.")
    st.stop()

d         = st.session_state["data"]
raw_df    = d["raw"]
by_strike = d["by_strike"]
by_expiry = d["by_expiry"]
spot      = d["spot"]
cur_tick  = d["ticker"]
raw_full  = d.get("raw_full", raw_df)
intraday  = d.get("intraday")
ohlc_spx  = d.get("ohlc_spx")
vix_hist  = d.get("vix_hist")

# ── Auto-refresh every 20 minutes (intraday only) ────────────────────────────
_REFRESH_SEC = 20 * 60   # 20 minutes

try:
    from streamlit_autorefresh import st_autorefresh as _st_autorefresh
    _HAS_AUTOREFRESH = True
except ImportError:
    _HAS_AUTOREFRESH = False

# Toggle in sidebar — persisted in session_state
with st.sidebar:
    st.markdown("---")
    _ar_on = st.toggle(
        "🔄 Auto-refresh intraday (20 min)",
        value=st.session_state.get("_ar_enabled", False),
        key="_ar_toggle",
        help="Aggiorna le barre intraday e il prezzo automaticamente ogni 20 minuti, "
             "senza ri-scaricare il chain delle opzioni.",
    )
    st.session_state["_ar_enabled"] = _ar_on

    # Status row
    _lar = st.session_state.get("_last_auto_refresh")
    if _ar_on and _lar:
        _elapsed   = int((datetime.now() - _lar).total_seconds())
        _remaining = max(0, _REFRESH_SEC - _elapsed)
        _mm, _ss   = divmod(_remaining, 60)
        st.markdown(
            f"<p style='font-size:10px;color:#818CF8;text-align:center;margin:-6px 0 4px;'>"
            f"Ultimo: {_lar.strftime('%H:%M:%S')} &nbsp;|&nbsp; "
            f"Prossimo: {_mm:02d}:{_ss:02d}</p>",
            unsafe_allow_html=True,
        )
    elif _ar_on and not _HAS_AUTOREFRESH:
        st.warning("streamlit-autorefresh non installato — aggiungi al requirements.txt",
                   icon="⚠️")

    refresh_intra_btn = st.button("🔄 Aggiorna Intraday ora",
                                  use_container_width=True,
                                  disabled="data" not in st.session_state)
    if intraday is not None and isinstance(intraday, pd.DataFrame) and not intraday.empty:
        _last_bar = intraday['datetime'].iloc[-1]
        _ts = _last_bar.strftime('%H:%M') if hasattr(_last_bar, 'strftime') else str(_last_bar)
        st.markdown(f"<p style='font-size:10px;color:#818CF8;text-align:center;"
                    f"margin-top:-4px;'>Last bar: {_ts}</p>",
                    unsafe_allow_html=True)

# ── Auto-refresh trigger ──────────────────────────────────────────────────────
_did_auto_refresh = False
if _ar_on and _HAS_AUTOREFRESH and "data" in st.session_state:
    _count = _st_autorefresh(
        interval=_REFRESH_SEC * 1000,
        key="ar_intraday_ticker",
        debounce=False,
    )
    if _count > 0:
        _tick_ar = st.session_state["data"].get("ticker", "$SPX")
        _new     = fetch_intraday_history(_tick_ar, interval_min=5)
        if not _new.empty:
            st.session_state["data"]["intraday"] = _new
            intraday = _new
        st.session_state["_last_auto_refresh"] = datetime.now()
        _did_auto_refresh = True

# ── Manual refresh button ─────────────────────────────────────────────────────
if refresh_intra_btn and "data" in st.session_state:
    with st.spinner("⏳ Aggiornamento intraday…"):
        new_intra = fetch_intraday_history(cur_tick, interval_min=5)
    st.session_state["data"]["intraday"] = new_intra
    st.session_state["_last_auto_refresh"] = datetime.now()
    intraday = new_intra
    if not new_intra.empty:
        st.success(f"Intraday aggiornato — {len(new_intra)} barre  |  "
                   f"Last ${float(new_intra['close'].iloc[-1]):,.2f}")
    else:
        st.warning("Intraday non disponibile — verrà mostrato l'ultimo prezzo noto.")

# ── Delta → strike bounds ─────────────────────────────────────────────────────
# Use raw_full (complete unfiltered chain) for delta_strike_bounds so the
# nearest-expiry always has dense, realistic delta values regardless of what
# DTE / OI filters are active on raw_df.
strike_lo, strike_hi = delta_strike_bounds(raw_full, delta_range[0], delta_range[1])

# ── Status bar ────────────────────────────────────────────────────────────────
_fetched = st.session_state.get("data", {}).get("fetched_at")
if _fetched:
    _age_min = (datetime.now() - _fetched).total_seconds() / 60
    if _age_min < 10:
        _fresh = f"🟢 dati di {_age_min:.0f} min fa"
    elif _age_min < 60:
        _fresh = f"🟡 dati di {_age_min:.0f} min fa"
    else:
        _hrs = _age_min / 60
        _fresh = (f"🔴 dati di {_hrs:.1f}h fa — ricarica la chain"
                  if _hrs < 24 else f"🔴 dati di {_hrs/24:.0f}g fa — ricarica la chain")
    _fresh_txt = f"{_fetched.strftime('%H:%M')} ({_fresh})"
else:
    _fresh_txt = datetime.now().strftime("%H:%M:%S")
st.markdown(
    f"<div class='status-bar'>Dati caricati: {_fresh_txt} &nbsp;|&nbsp; "
    f"{len(raw_df):,} contracts &nbsp;|&nbsp; {raw_df['expiry'].nunique()} expiries &nbsp;|&nbsp; "
    f"Spot <b style='color:#ffd166'>${spot:.2f}</b> &nbsp;|&nbsp; "
    f"Δ filter: [{delta_range[0]:+.2f}, {delta_range[1]:+.2f}]</div>",
    unsafe_allow_html=True,
)

# ── Metric cards ──────────────────────────────────────────────────────────────
stats = build_stat_metrics(raw_df, by_strike, spot)

def _metric_help(label):
    if "GEX Regime" in label:
        return ("Regime basato su spot vs gamma flip. Se il GEX è negativo a tutti "
                "gli strike non esiste un flip e il regime segue il segno del GEX "
                "vicino allo spot. Può differire dal Gamma Regime (0DTE).")
    if label == "Gamma Flip":
        return ("Strike dove il GEX netto cambia segno. '—' = nessun flip reale "
                "(mercato interamente in un solo regime).")
    return None

def _render_metric_cards(items, per_row=4):
    """Custom HTML metric cards: full value always visible (wraps, never
    truncated), unlike st.metric which clips long values to '$73…' on narrow
    columns. Title shown in full with a tooltip when a help text exists."""
    for _i in range(0, len(items), per_row):
        _chunk = items[_i:_i + per_row]
        _cols = st.columns(len(_chunk))
        for _col, (label, val) in zip(_cols, _chunk):
            # val may be (value, color) tuple or plain string
            if isinstance(val, (tuple, list)):
                value = val[0]
            else:
                value = val
            _hlp = _metric_help(label)
            _title_attr = f' title="{_hlp}"' if _hlp else ''
            _help_mark = ' &#9432;' if _hlp else ''
            _html = (
                f'<div style="background:#FFFFFF;border:1px solid #E5E7EB;'
                f'border-radius:12px;box-shadow:0 1px 4px rgba(0,0,0,0.06),0 4px 12px rgba(0,0,0,0.04);'
                f'padding:14px 16px;min-height:78px;margin-bottom:12px;"{_title_attr}>'
                f'<div style="color:#9CA3AF;font-size:10.5px;font-weight:600;text-transform:uppercase;'
                f'letter-spacing:0.6px;margin-bottom:6px;line-height:1.25;">{label}{_help_mark}</div>'
                f'<div style="color:#111827;font-size:19px;font-weight:700;line-height:1.2;'
                f'word-break:break-word;font-family:\'Inter\',sans-serif;">{value}</div>'
                f'</div>'
            )
            _col.markdown(_html, unsafe_allow_html=True)


def cards(specs, per_row=5):
    """Render a list of metric cards from dicts with keys:
       label (str), value (str), delta (str, optional), delta_color
       ('green'|'red'|'grey', optional), help (str, optional).
    Values and titles wrap fully — never truncated. Use everywhere instead
    of st.metric so all tabs render consistently on landscape and portrait."""
    _dc = {'green': '#10B981', 'red': '#EF4444', 'grey': '#9CA3AF'}
    for _i in range(0, len(specs), per_row):
        _chunk = specs[_i:_i + per_row]
        _cols = st.columns(len(_chunk))
        for _col, _s in zip(_cols, _chunk):
            label = _s.get('label', '')
            value = _s.get('value', '—')
            delta = _s.get('delta')
            dcol  = _dc.get(_s.get('delta_color', 'grey'), '#9CA3AF')
            hlp   = _s.get('help')
            _title_attr = f' title="{hlp}"' if hlp else ''
            _mark = ' &#9432;' if hlp else ''
            _delta_html = (f"<div style='color:{dcol};font-size:11px;font-weight:600;margin-top:4px;'>{delta}</div>"
                           if delta else "")
            _html = (
                f'<div style="background:#FFFFFF;border:1px solid #E5E7EB;'
                f'border-radius:12px;box-shadow:0 1px 4px rgba(0,0,0,0.06),0 4px 12px rgba(0,0,0,0.04);'
                f'padding:14px 16px;min-height:78px;margin-bottom:12px;"{_title_attr}>'
                f'<div style="color:#9CA3AF;font-size:10.5px;font-weight:600;text-transform:uppercase;'
                f'letter-spacing:0.6px;margin-bottom:6px;line-height:1.25;">{label}{_mark}</div>'
                f'<div style="color:#111827;font-size:19px;font-weight:700;line-height:1.2;'
                f'word-break:break-word;font-family:\'Inter\',sans-serif;">{value}</div>'
                f'{_delta_html}</div>'
            )
            _col.markdown(_html, unsafe_allow_html=True)

_render_metric_cards(list(stats.items()), per_row=4)

st.markdown("---")

# ── Alert Flags Panel ─────────────────────────────────────────────────────────
if "data" in st.session_state and st.session_state["data"].get("raw") is not None:
    _d      = st.session_state["data"]
    _raw_f  = _d.get("raw_full")
    if _raw_f is None:
        _raw_f = _d.get("raw")
    _spot_a = _d.get("spot", 0)
    _dte0_m = compute_0dte_metrics(_raw_f, _spot_a) if _raw_f is not None else {}
    _ohlc_s = _d.get("ohlc_spx")
    _rvdf   = compute_rvol_all(_ohlc_s, window=126) \
              if _ohlc_s is not None else None
    _flags  = build_alert_flags(_raw_f, _spot_a, _dte0_m, _rvdf, _d.get("vix_hist"))

    # ── Alert journal: persist status transitions ──
    try:
        import csv as _csv
        _prev_states = st.session_state.get("_alert_states", {})
        _journal_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots")
        os.makedirs(_journal_dir, exist_ok=True)
        _journal = os.path.join(_journal_dir, "alert_journal.csv")
        _changes = [(f["name"], _prev_states.get(f["name"]), f["status"], f["value"])
                    for f in _flags
                    if _prev_states.get(f["name"]) not in (None, f["status"])]
        if _changes:
            _new_file = not os.path.exists(_journal)
            with open(_journal, "a", newline="") as _jf:
                _w = _csv.writer(_jf)
                if _new_file:
                    _w.writerow(["timestamp", "alert", "from", "to", "value"])
                for _nm, _fr, _to, _vl in _changes:
                    _w.writerow([datetime.now().isoformat(timespec="seconds"),
                                 _nm, _fr, _to, _vl])
        st.session_state["_alert_states"] = {f["name"]: f["status"] for f in _flags}
    except Exception:
        pass

    _icon   = {'RED':'🔴','AMBER':'🟡','GREEN':'🟢','GREY':'⚪'}
    _bg     = {'RED':'#FEF2F2','AMBER':'#FFFBEB','GREEN':'#F0FDF4','GREY':'#F9FAFB'}
    _border = {'RED':'#EF4444','AMBER':'#F59E0B','GREEN':'#10B981','GREY':'#D1D5DB'}
    _tcol   = {'RED':'#991B1B','AMBER':'#92400E','GREEN':'#065F46','GREY':'#6B7280'}

    flag_cols = st.columns(len(_flags))
    for col, fl in zip(flag_cols, _flags):
        s = fl['status']
        col.markdown(f"""
<div style="background:{_bg[s]};border:1.5px solid {_border[s]};border-radius:10px;
            padding:10px 8px;text-align:center;min-height:90px;">
  <div style="font-size:20px;line-height:1.2;">{_icon[s]}</div>
  <div style="font-size:10px;color:#6B7280;font-weight:700;text-transform:uppercase;
              letter-spacing:0.6px;margin:2px 0;">{fl['name']}</div>
  <div style="font-size:15px;font-weight:700;color:{_tcol[s]};line-height:1.2;">{fl['value']}</div>
  <div style="font-size:9px;color:#9CA3AF;margin-top:2px;">{fl['detail']}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:4px;text-align:right;font-size:9px;"
                f"color:#9CA3AF;'>Alert monitor — {datetime.now().strftime('%H:%M:%S')}</div>",
                unsafe_allow_html=True)

# ── 0DTE metrics (computed once, used in the first tab) ───────────────────────
dte0_metrics = compute_0dte_metrics(raw_full, spot)
has_0dte     = bool(dte0_metrics)

# ── Tabs — 0DTE is always first and shown by default ─────────────────────────
tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "⚡ 0DTE",
    "📊 GEX / DEX",
    "📐 Range & Skew",
    "💰 Put Monitor",
    "📈 Volatilità",
    "🔥 Open Interest",
    "📉 Realized Vol",
    "🤖 Long Vol Strat",
    "🌊 VolDex",
    "📉 Short Vol Strat",
    "💵 Specchietto Premi",
])

# ── Tab 0: 0DTE ───────────────────────────────────────────────────────────────
with tab0:
    if not has_0dte:
        st.info(
            "Nessuna opzione con scadenza **oggi** nel chain caricato.  \n"
            "Le 0DTE compaiono solo nei giorni in cui SPX ha una scadenza intraday "
            "(Lun, Mer, Ven per le settimanali; ogni giorno per SPXW).  \n"
            "Ricarica durante la sessione di trading oppure verifica che "
            "`FETCH_EXPIRY_DAYS` includa DTE=0."
        )
    else:
        m = dte0_metrics
        # ── Metric cards — row 1 ──────────────────────────────────────────
        pc = m.get('pc_ratio')
        charm_daily = (m['charm_exp'] / 252 / 1e6) if m.get('charm_exp') else None
        vanna_m = (m['vanna_exp'] / 1e6) if m.get('vanna_exp') else None
        gross = (m['gross_gex'] / 1e9) if m.get('gross_gex') else None
        cards([
            {'label': '0DTE ATM IV',
             'value': f"{m['atm_iv']*100:.1f}%" if m['atm_iv'] else "—"},
            {'label': 'Expected Move',
             'value': (f"±{m['exp_move_pts']:.1f} pts ({m['exp_move_pct']:.2f}%)"
                       if m['exp_move_pts'] else "—")},
            {'label': 'GEX Flip',
             'value': f"${m['gex_flip']:.0f}" if m['gex_flip'] else "—"},
            {'label': 'Max Gamma Strike',
             'value': f"${m['max_gex_strike']:.0f}" if m['max_gex_strike'] else "—"},
            {'label': 'Max Pain',
             'value': f"${m['max_pain']:.0f}" if m.get('max_pain') else "—",
             'help': 'Strike che minimizza il payout totale agli option buyer — pinning atteso a scadenza'},
            {'label': 'Total 0DTE GEX',
             'value': f"{'+'if m['total_gex']>=0 else ''}{m['total_gex']/1e9:.2f}B"},
        ], per_row=6)
        # ── Metric cards — row 2 ──────────────────────────────────────────
        cards([
            {'label': 'P/C OI Ratio', 'value': f"{pc:.2f}" if pc else "—",
             'delta': (("↑ Lean PUT" if pc > 1.2 else ("↓ Lean CALL" if pc < 0.8 else "Neutro")) if pc else None),
             'delta_color': ('red' if pc and pc > 1.2 else ('green' if pc and pc < 0.8 else 'grey')),
             'help': 'Put OI / Call OI. >1 = lean ribassista; <1 = lean rialzista'},
            {'label': 'Charm Exp (daily Δ)',
             'value': f"${charm_daily:+.1f}M/day" if charm_daily else "—",
             'help': 'Delta drift giornaliero dal solo passare del tempo (charm × OI × 100 × Spot / 252).'},
            {'label': 'Vanna Exp (per 1pt vol)',
             'value': f"${vanna_m:+.1f}M" if vanna_m else "—",
             'help': 'Variazione del delta-hedge quando la vol implicita si muove di 1 punto. Negativo = dealer compra quando la vol sale.'},
            {'label': 'Gross GEX 0DTE',
             'value': f"${gross:.2f}B" if gross else "—",
             'help': 'GEX lordo (|call_gex| + |put_gex|) senza netting. Intensità totale del gamma hedging.'},
            {'label': '0DTE Contracts', 'value': f"{m['n_contracts']:,}"},
        ], per_row=5)

        st.markdown("---")

        # ── GEX + DEX chart ──
        col_a, col_b = st.columns([1.4, 1])
        with col_a:
            try:
                st.plotly_chart(
                    gex_dex_0dte_chart(m, spot, cur_tick),
                    use_container_width=True,
                )
            except Exception as e:
                st.plotly_chart(empty_fig(str(e)), use_container_width=True)
        with col_b:
            try:
                st.plotly_chart(
                    smile_0dte_chart(m, spot, cur_tick),
                    use_container_width=True,
                )
            except Exception as e:
                st.plotly_chart(empty_fig(str(e)), use_container_width=True)

        # ── Call vs Put OI ──
        try:
            st.plotly_chart(oi_0dte_chart(m, spot, cur_tick),
                            use_container_width=True)
        except Exception as e:
            st.plotly_chart(empty_fig(str(e)), use_container_width=True)

        # ── Morning Brief PDF ──
        try:
            _ga_brief = compute_gex_analytics(raw_full,
                                              aggregate_by_strike(raw_full), spot)
            _dod_b    = compute_dod_changes(raw_full, spot, cur_tick)
            _flags_brief = build_alert_flags(
                raw_full, spot, m,
                None, st.session_state["data"].get("vix_hist"))
            _pdf = generate_morning_brief(m, _ga_brief, _flags_brief,
                                          spot, cur_tick, _dod_b)
            st.download_button("📄 Scarica Morning Brief (PDF)", _pdf,
                               file_name=f"morning_brief_{cur_tick.replace('$','')}_"
                                         f"{datetime.now().strftime('%Y%m%d')}.pdf",
                               mime="application/pdf", key="dl_brief")
        except Exception as _e:
            st.caption(f"Brief non disponibile: {_e}")

        # ── Gamma decay schedule ──────────────────────────────────────────
        with st.expander("⏱ Gamma Burn Schedule — Decadimento intraday del GEX 0DTE"):
            sched = compute_0dte_gamma_schedule(m['raw'], spot)
            if not sched.empty:
                sched_disp = sched.copy()
                sched_disp['gross_gex_B'] = (sched_disp['gross_gex']/1e9).round(3)
                sched_disp['net_gex_B']   = (sched_disp['net_gex']/1e9).round(3)
                sched_disp['Δ gross vs now'] = (
                    (sched_disp['gross_gex'] - sched_disp['gross_gex'].iloc[0])
                    / sched_disp['gross_gex'].iloc[0] * 100
                ).map(lambda x: f"{x:+.1f}%")
                sched_disp = sched_disp.rename(columns={
                    'time':'Ora ET','T_hours':'Ore rimaste',
                    'gross_gex_B':'Gross GEX ($B)','net_gex_B':'Net GEX ($B)'
                })[['Ora ET','Ore rimaste','Net GEX ($B)','Gross GEX ($B)','Δ gross vs now']]
                st.caption(
                    "Mostra come il GEX 0DTE cambia nel corso della sessione (IV e OI costanti). "
                    "Il gamma esplode nelle ultime 1-2 ore — questo è il meccanismo del pinning "
                    "e degli squeeze intraday di fine giornata.")
                st.dataframe(sched_disp.reset_index(drop=True),
                             use_container_width=True, hide_index=True)
        with st.expander("📋 Strikes chiave 0DTE — Greche complete"):
            bs0 = m['by_strike']
            if not bs0.empty:
                disp_cols = [c for c in
                             ['strike','net_gex','net_dex','gross_gex',
                              'charm_exp','vanna_exp',
                              'call_gex','put_gex','call_dex','put_dex','total_oi']
                             if c in bs0.columns]
                tbl = bs0[disp_cols].copy()
                # Convert to human-readable units
                for col in ['net_gex','net_dex','gross_gex','call_gex','put_gex',
                            'call_dex','put_dex']:
                    if col in tbl.columns:
                        tbl[col] = (tbl[col] / 1e6).round(1)
                for col in ['charm_exp','vanna_exp']:
                    if col in tbl.columns:
                        tbl[col] = (tbl[col] / 1e6).round(2)
                # Rename for clarity
                rename = {
                    'net_gex':'NetGEX($M)','net_dex':'NetDEX($M)',
                    'gross_gex':'GrossGEX($M)','charm_exp':'Charm($M/yr)',
                    'vanna_exp':'Vanna($M/pt)','call_gex':'CallGEX($M)',
                    'put_gex':'PutGEX($M)','call_dex':'CallDEX($M)',
                    'put_dex':'PutDEX($M)','total_oi':'OI'
                }
                tbl = tbl.rename(columns=rename).sort_values('NetGEX($M)',
                                                              ascending=False,
                                                              key=abs)
                st.dataframe(tbl.reset_index(drop=True),
                             use_container_width=True, hide_index=True)

# ── Tab 1: GEX / DEX ─────────────────────────────────────────────────────────
with tab1:
    # ── GEX Advanced Analytics ────────────────────────────────────────────
    gex_analytics = compute_gex_analytics(raw_df, by_strike, spot)
    if gex_analytics:
        ga = gex_analytics
        _hhi_tag = ("<span style='color:#10B981;font-size:11px;'>&#8593; Bassa</span>"
                    if ga['hhi'] <= 0.05 else
                    "<span style='color:#EF4444;font-size:11px;'>&#8593; Alta</span>")
        _an_cards = [
            ("GEX Center of Mass", f"${ga['center_of_mass']:,.0f}"),
            ("HHI Concentrazione", f"{ga['hhi']:.4f}<br>{_hhi_tag}"),
            ("Impact 1% Move", f"${ga['impact_1pct']/1e6:.0f}M Δ"),
            ("Impact 5% Move", f"${ga['impact_5pct']/1e6:.0f}M Δ"),
            ("Flip Zone", f"[{ga['flip_zone_lo']:,.0f} – {ga['flip_zone_hi']:,.0f}]"),
        ]
        _render_metric_cards(_an_cards, per_row=5)

        # ── Day-over-day changes + GEX percentile ──
        dod = compute_dod_changes(raw_full, spot, cur_tick)
        pct = compute_gex_percentile(raw_full, cur_tick)
        if dod or pct.get('percentile') is not None:
            _dod_cards = []
            if dod:
                _g = ("<span style='color:#10B981;font-size:11px;'>&#8593; "
                      f"{dod['d_gex_pct']:+.1f}%</span>" if dod.get('d_gex_pct') and dod['d_gex_pct'] >= 0
                      else (f"<span style='color:#EF4444;font-size:11px;'>&#8595; "
                            f"{dod['d_gex_pct']:+.1f}%</span>" if dod.get('d_gex_pct') else ""))
                _o = ("<span style='color:#10B981;font-size:11px;'>&#8593; "
                      f"{dod['d_oi_pct']:+.1f}%</span>" if dod.get('d_oi_pct') and dod['d_oi_pct'] >= 0
                      else (f"<span style='color:#EF4444;font-size:11px;'>&#8595; "
                            f"{dod['d_oi_pct']:+.1f}%</span>" if dod.get('d_oi_pct') else ""))
                _dod_cards.append(("Δ Net GEX vs " + dod['prev_date'][5:],
                                   f"{dod['d_net_gex']/1e9:+.2f}B" + (f"<br>{_g}" if _g else "")))
                _dod_cards.append(("Δ Total OI",
                                   f"{dod['d_total_oi']/1e3:+.0f}k" + (f"<br>{_o}" if _o else "")))
                _dod_cards.append(("Δ Spot", f"{dod['d_spot']:+.1f} pts"))
            if pct.get('percentile') is not None:
                _dod_cards.append(("GEX Percentile (storico)", f"{pct['percentile']:.0f}°"))
            elif pct.get('n_history') is not None:
                _dod_cards.append(("GEX Percentile", "—"))
            if _dod_cards:
                _render_metric_cards(_dod_cards, per_row=5)
            st.markdown("---")

        # ── GEX Flip per scadenza (term structure) ──
        with st.expander("🧱 GEX Flip per Scadenza — Term structure dei muri gamma"):
            flip_ts = compute_flip_by_expiry(raw_full, spot)
            if not flip_ts.empty:
                ft = flip_ts.copy()
                ft['flip'] = ft['flip'].map(lambda x: f"${x:,.0f}" if x else "—")
                ft['net_gex'] = (ft['net_gex']/1e9).round(2)
                ft['spot_vs_flip'] = ft['spot_vs_flip'].map(
                    lambda x: f"{x:+.0f} pts" if x is not None else "—")
                ft = ft.rename(columns={
                    'expiry':'Scadenza','T_days':'DTE','flip':'GEX Flip',
                    'net_gex':'Net GEX ($B)','regime':'Regime',
                    'spot_vs_flip':'Spot − Flip'})
                st.caption("Il flip aggregato può nascondere muri diversi per scadenza: "
                           "il muro 0DTE e quello monthly possono essere a livelli distinti.")
                st.dataframe(ft.reset_index(drop=True),
                             use_container_width=True, hide_index=True)
            else:
                st.info("Dati insufficienti per il calcolo per scadenza")

        # GEX Conditional Profile
        with st.expander("📐 Profilo GEX Condizionale — Come cambia il regime con il prezzo"):
            profile = compute_gex_profile(raw_df, spot)
            if not profile.empty:
                profile_disp = profile.copy()
                profile_disp['net_gex_B'] = (profile_disp['net_gex']/1e9).round(3)
                profile_disp['gross_gex_B'] = (profile_disp['gross_gex']/1e9).round(3)
                profile_disp['shift_pct'] = profile_disp['shift_pct'].map(lambda x: f"{x:+.0f}%")
                profile_disp['spot_level'] = profile_disp['spot_level'].map(lambda x: f"${x:,.0f}")
                profile_disp = profile_disp.rename(columns={
                    'shift_pct':'Movimento','spot_level':'Spot Level',
                    'net_gex_B':'Net GEX ($B)','gross_gex_B':'Gross GEX ($B)','regime':'Regime'
                })[['Movimento','Spot Level','Net GEX ($B)','Gross GEX ($B)','Regime']]
                st.caption(
                    "Mostra il GEX netto se il prezzo fosse a quel livello — IV e OI rimangono costanti. "
                    "Identifica dove il regime cambia con movimenti di mercato. "
                    "Nota: qui il regime a ogni livello segue il segno del Net GEX a *quel* prezzo "
                    "ipotetico — il livello in cui passa da SHORT a LONG è il gamma flip, "
                    "coerente con il 'GEX Regime' della scheda principale.")
                st.dataframe(profile_disp.reset_index(drop=True),
                             use_container_width=True, hide_index=True)
        st.markdown("---")
    # Price vs DEX Levels — the centrepiece: shared Y-axis chart
    try:
        _ga_lvl = gex_analytics or {}
        _max_pain = None
        try:
            _max_pain = compute_max_pain(raw_df)
        except Exception:
            pass
        _key_levels = {
            'Gamma Flip': _ga_lvl.get('gamma_flip'),
            'Peak GEX': by_strike.loc[by_strike['net_gex'].abs().idxmax(), 'strike']
                        if not by_strike.empty else None,
            'Max Pain': _max_pain,
        }
        st.plotly_chart(
            price_vs_dex_chart(by_strike, intraday, spot, cur_tick,
                               strike_lo=strike_lo, strike_hi=strike_hi,
                               key_levels=_key_levels),
            use_container_width=True,
        )
    except Exception as e:
        st.plotly_chart(empty_fig(str(e)), use_container_width=True)

    st.markdown("---")

    # ── Key levels + auto-narrative (MenthorQ-style read, own nomenclature) ──
    _ga_kl = gex_analytics or {}
    _klv = compute_key_levels(by_strike, spot, _ga_kl.get('gamma_flip'))
    _gex_levels = {
        'Call Resistance': _klv.get('call_resistance'),
        'Put Support': _klv.get('put_support'),
        'Gamma Flip': _klv.get('gamma_flip'),
    }
    try:
        _narr = gex_regime_narrative(spot, _ga_kl.get('gamma_flip'),
                                     _ga_kl.get('regime'),
                                     _ga_kl.get('net_gex_total'), _klv)
        _is_short = _ga_kl.get('regime') and 'SHORT' in _ga_kl.get('regime')
        _nbg = '#FEF3C7' if _is_short else '#ECFDF5'
        _ntx = '#92400E' if _is_short else '#065F46'
        _nbd = '#F59E0B' if _is_short else '#10B981'
        st.markdown(
            f'<div style="background:{_nbg};border-left:4px solid {_nbd};'
            f'border-radius:8px;padding:12px 15px;margin-bottom:14px;">'
            f'<div style="color:{_ntx};font-size:11px;font-weight:700;'
            f'text-transform:uppercase;letter-spacing:0.6px;margin-bottom:4px;">'
            f'🗺️ Lettura della struttura GEX</div>'
            f'<span style="color:{_ntx};font-size:13px;line-height:1.5;">{_narr}'
            f'</span></div>',
            unsafe_allow_html=True)
    except Exception:
        pass

    col_l, col_r = st.columns(2)
    with col_l:
        try:
            st.plotly_chart(gex_bar_chart(by_strike, spot, cur_tick,
                                          strike_lo=strike_lo, strike_hi=strike_hi,
                                          key_levels=_gex_levels),
                            use_container_width=True)
        except Exception as e:
            st.plotly_chart(empty_fig(str(e)), use_container_width=True)
    with col_r:
        try:
            st.plotly_chart(dex_bar_chart(by_strike, spot, cur_tick,
                                          strike_lo=strike_lo, strike_hi=strike_hi),
                            use_container_width=True)
        except Exception as e:
            st.plotly_chart(empty_fig(str(e)), use_container_width=True)
    try:
        st.plotly_chart(gex_expiry_chart(by_expiry, cur_tick), use_container_width=True)
    except Exception as e:
        st.plotly_chart(empty_fig(str(e)), use_container_width=True)

    # Cumulative GEX profile — the curve whose zero-crossing IS the gamma flip
    st.markdown("---")
    st.markdown("##### GEX Cumulato — dove (e se) c'è il gamma flip")
    st.caption("La curva del GEX cumulato attraversa lo zero esattamente al gamma flip. "
               "Se non attraversa mai lo zero vicino allo spot, non esiste un flip reale "
               "e il mercato è interamente in un solo regime.")
    try:
        _flip_cum = (gex_analytics or {}).get('gamma_flip')
        st.plotly_chart(cumulative_gex_chart(by_strike, spot, _flip_cum, cur_tick),
                        use_container_width=True)
    except Exception as e:
        st.plotly_chart(empty_fig(str(e)), use_container_width=True)

    # Signal health — internal consistency checks
    st.markdown("---")
    st.markdown("##### 🩺 Coerenza dei segnali")
    try:
        _m0_hc = compute_0dte_metrics(raw_df, spot) or {}
        _checks = signal_health_check(raw_df, by_strike, spot, _m0_hc)
        # High-contrast palette: (bg, border, text) per level — dark text always
        _hc_style = {
            'ok':   ('#ECFDF5', '#10B981', '#065F46', '✓'),
            'warn': ('#FEF3C7', '#F59E0B', '#92400E', '⚠️'),
            'info': ('#EFF6FF', '#3B82F6', '#1E40AF', 'ℹ️'),
        }
        for _ck in _checks:
            _bg, _bd, _tx, _ic = _hc_style.get(_ck['level'], _hc_style['info'])
            _msg = _ck['message'].replace('<', '&lt;').replace('>', '&gt;')
            st.markdown(
                f'<div style="background:{_bg};border-left:4px solid {_bd};'
                f'border-radius:8px;padding:11px 14px;margin-bottom:9px;">'
                f'<span style="color:{_tx};font-size:13px;line-height:1.45;">'
                f'{_ic} {_msg}</span></div>',
                unsafe_allow_html=True)
    except Exception as e:
        st.caption(f"Check non disponibile: {e}")

    # GEX/DEX metrics history — the TREND, not just today's value
    st.markdown("---")
    st.markdown("##### 📈 Storico metriche GEX/DEX")
    st.caption("Per il long-vol il *cambiamento* del regime conta più del livello "
               "assoluto. Salva uno snapshot a ogni caricamento (o lascia fare al "
               "motore automatico) per costruire la serie.")
    _gh1, _gh2 = st.columns([3, 1])
    with _gh2:
        if st.button("💾 Salva metriche oggi", key="save_gexhist"):
            try:
                save_gex_metrics_snapshot(raw_df, by_strike, spot)
                st.success("Salvato.")
                st.rerun()
            except Exception as e:
                st.error(f"Errore: {e}")
    _ghist = load_gex_metrics_history()
    with _gh1:
        st.caption(f"Storico: {len(_ghist)} giorni salvati.")
    if len(_ghist) >= 2:
        _metric_choice = st.selectbox(
            "Metrica da visualizzare",
            options=['net_gex', 'gamma_flip', 'hhi', 'pc_ratio', 'total_dex'],
            format_func=lambda m: {
                'net_gex': 'Net GEX', 'gamma_flip': 'Gamma Flip vs Spot',
                'hhi': 'HHI (concentrazione)', 'pc_ratio': 'Put/Call OI',
                'total_dex': 'Total DEX'}.get(m, m),
            key="gexhist_metric")
        try:
            st.plotly_chart(gex_metrics_history_chart(_ghist, _metric_choice),
                            use_container_width=True)
        except Exception as e:
            st.plotly_chart(empty_fig(str(e)), use_container_width=True)
        # Regime timeline
        if 'regime' in _ghist.columns:
            _reg_recent = _ghist[['date', 'regime']].tail(10).iloc[::-1]
            _reg_recent['date'] = _reg_recent['date'].dt.strftime('%Y-%m-%d')
            st.caption("Regime negli ultimi giorni:")
            st.dataframe(_reg_recent, use_container_width=True, hide_index=True)
        st.download_button("⬇ Esporta storico metriche",
                           _ghist.to_csv(index=False).encode(),
                           file_name="gex_metrics_history.csv", mime="text/csv",
                           key="dl_gexhist")
    else:
        st.info("Servono almeno 2 giorni di dati per il grafico storico. "
                "Si popolerà con i prossimi salvataggi (manuali o automatici).")

# ── Tab 2: Range & Skew ───────────────────────────────────────────────────────
with tab2:
    st.markdown("##### Implied Daily Range  (Δ 0.30 IV)")
    try:
        st.plotly_chart(daily_range_chart(raw_full, spot, cur_tick),
                        use_container_width=True)
    except Exception as e:
        st.plotly_chart(empty_fig(str(e)), use_container_width=True)

    st.markdown("---")
    st.markdown("##### Put IV Skew Overlay")

    # Controls
    skew_col1, skew_col2 = st.columns([1, 2])
    with skew_col1:
        iv_levels = st.multiselect(
            "IV reference levels",
            options=[0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.25],
            default=[0.16, 0.18, 0.20],
            format_func=lambda x: f"{int(x*100)}%",
        )
    with skew_col2:
        all_expiries = sorted(raw_full["expiry"].unique())
        near_expiries = [e for e in all_expiries
                         if (raw_full[raw_full["expiry"]==e]["T_days"].iloc[0]) <= MAX_EXPIRY_DAYS]
        selected_exp = st.multiselect(
            "Scadenze (vuoto = tutte ≤ MAX_EXPIRY_DAYS)",
            options=all_expiries,
            default=[],
        )

    try:
        exp_for_chart = selected_exp if selected_exp else near_expiries
        st.plotly_chart(
            iv_skew_overlay_chart(raw_full, spot, cur_tick,
                                  selected_expiries=exp_for_chart,
                                  iv_levels=iv_levels or [0.16, 0.18, 0.20]),
            use_container_width=True,
        )
    except Exception as e:
        st.plotly_chart(empty_fig(str(e)), use_container_width=True)

# ── Tab 3: Put Monitor ────────────────────────────────────────────────────────
with tab3:
    try:
        st.plotly_chart(put_monitor_chart(raw_full, spot, cur_tick),
                        use_container_width=True)
    except Exception as e:
        st.plotly_chart(empty_fig(str(e)), use_container_width=True)

    with st.expander("📋 Tabella dettaglio premi put", expanded=True):
        try:
            pm = get_put_monitor(raw_full, spot)
            if pm.empty:
                st.info("Nessuna scadenza mensile/trimestrale nel chain caricato.")
            else:
                disp = pm[["expiry","type","T_days","delta_target",
                            "strike","mid","iv_pct","mid_pct_spot","delta_actual"]].copy()
                disp.columns = ["Expiry","Tipo","DTE","Δ Target",
                                "Strike","Mid (pts)","IV %","Mid % Spot","Δ Eff."]

                def _color_mid(col):
                    """Yellow→Red gradient without matplotlib."""
                    vals = col.to_numpy(dtype=float)
                    mn, mx = vals.min(), vals.max()
                    norm = (vals - mn) / (mx - mn + 1e-9)
                    colors = []
                    for v in norm:
                        r = int(255)
                        g = int(255 * (1 - v * 0.75))
                        b = int(max(0, 255 * (1 - v * 1.5)))
                        colors.append(f'background-color: rgb({r},{g},{b})')
                    return colors

                st.dataframe(
                    disp.style
                        .format({"Mid (pts)":"{:.2f}", "IV %":"{:.1f}",
                                 "Mid % Spot":"{:.3f}%", "Δ Eff.":"{:.3f}",
                                 "Δ Target":"{:.2f}"})
                        .apply(_color_mid, subset=["Mid % Spot"]),
                    use_container_width=True,
                    hide_index=True,
                )
        except Exception as e:
            st.error(str(e))

# ── Tab 4: Volatilità ─────────────────────────────────────────────────────────
with tab4:
    col_a, col_b = st.columns(2)
    with col_a:
        try:
            st.plotly_chart(vol_smile_chart(raw_df, spot, cur_tick), use_container_width=True)
        except Exception as e:
            st.plotly_chart(empty_fig(str(e)), use_container_width=True)
    with col_b:
        try:
            st.plotly_chart(term_structure_chart(raw_df, spot, cur_tick), use_container_width=True)
        except Exception as e:
            st.plotly_chart(empty_fig(str(e)), use_container_width=True)
    try:
        st.plotly_chart(skew_chart(raw_df, spot, cur_tick), use_container_width=True)
    except Exception as e:
        st.plotly_chart(empty_fig(str(e)), use_container_width=True)

    st.markdown("#### 🌐 IV Surface (3D)")
    expiry_opts    = sorted(raw_df["expiry"].unique())
    sel_expiries   = st.multiselect("Scadenze per IV Surface", options=expiry_opts,
                                    default=expiry_opts[:6])
    try:
        st.plotly_chart(iv_surface_chart(raw_df, spot, cur_tick, expiries_list=sel_expiries),
                        use_container_width=True)
    except Exception as e:
        st.plotly_chart(empty_fig(str(e)), use_container_width=True)

    # ── Market Profile / Value Area (VA-80) ───────────────────────────────
    st.markdown("---")
    st.markdown("#### 📊 Market Profile / Value Area (VA-80)")
    st.caption("La value area contiene l'80% del volume della sessione. POC = "
               "livello più scambiato. Il rapporto tra prezzo e VA definisce il "
               "bias direzionale (usato dal Ralendar).")
    if intraday is None or (hasattr(intraday, 'empty') and intraday.empty):
        st.info("Dati intraday non disponibili — la value area richiede le barre "
                "intraday (Yahoo). Riprova a mercato aperto o ricarica.")
    else:
        try:
            _va = compute_value_area(intraday)
            if _va.get('va_high') is None:
                st.info("Value area non calcolabile con i dati intraday attuali.")
            else:
                _vp = value_area_position(spot, _va)
                _zone_label = {
                    'above_va': '🔴 Sopra la Value Area', 'upper_va': '🟠 Bordo alto VA',
                    'inside_va': '🔵 Dentro la Value Area', 'lower_va': '🟢 Bordo basso VA',
                    'below_va': '🟢 Sotto la Value Area',
                }.get(_vp['zone'], '—')
                _bias_label = {
                    'delta_negativo': 'bias delta NEGATIVO',
                    'delta_positivo': 'bias delta POSITIVO',
                    'neutro': 'nessun bias direzionale',
                }.get(_vp['bias_hint'], '—')
                cards([
                    {'label': 'POC', 'value': f"{_va['poc']:.0f}"},
                    {'label': 'VA-High (VA-80)', 'value': f"{_va['va_high']:.0f}"},
                    {'label': 'VA-Low', 'value': f"{_va['va_low']:.0f}"},
                    {'label': 'Copertura', 'value': f"{_va['va_pct']*100:.0f}%"},
                    {'label': 'Metodo', 'value': _va['method'].upper()},
                ], per_row=5)
                _cvl, _cvr = st.columns([2, 1])
                with _cvl:
                    st.plotly_chart(value_area_chart(_va, spot, cur_tick),
                                    use_container_width=True)
                with _cvr:
                    _bias_color = ('#FEE2E2' if 'negativo' in _vp['bias_hint']
                                   else '#DCFCE7' if 'positivo' in _vp['bias_hint']
                                   else '#F3F4F6')
                    _bias_txt = ('#991B1B' if 'negativo' in _vp['bias_hint']
                                 else '#166534' if 'positivo' in _vp['bias_hint']
                                 else '#374151')
                    st.markdown(
                        f'<div style="background:{_bias_color};border-radius:10px;'
                        f'padding:14px 16px;margin-top:8px;">'
                        f'<div style="color:{_bias_txt};font-size:13px;font-weight:700;'
                        f'margin-bottom:6px;">{_zone_label}</div>'
                        f'<div style="color:{_bias_txt};font-size:12px;line-height:1.5;">'
                        f'Spot {spot:.0f} · {_bias_label}.<br>'
                        f'Distanza dal POC: {_vp["distance_pct"]:+.2f}%</div></div>',
                        unsafe_allow_html=True)
        except Exception as e:
            st.caption(f"Errore value area: {e}")

# ── Tab 5: Open Interest ──────────────────────────────────────────────────────
with tab5:
    try:
        st.plotly_chart(oi_heatmap(raw_df, spot, cur_tick,
                                   strike_lo=strike_lo, strike_hi=strike_hi),
                        use_container_width=True)
    except Exception as e:
        st.plotly_chart(empty_fig(str(e)), use_container_width=True)

    with st.expander("📋 Dati per strike (tabella)"):
        cols_to_show = ["strike","net_gex","net_dex","call_gex","put_gex",
                        "call_dex","put_dex","total_oi"]
        available = [c for c in cols_to_show if c in by_strike.columns]
        _oi_tbl = by_strike[available].sort_values("net_gex", ascending=False).reset_index(drop=True)
        st.dataframe(_oi_tbl, use_container_width=True)
        st.download_button("⬇ Esporta CSV", _oi_tbl.to_csv(index=False).encode(),
                           file_name=f"by_strike_{cur_tick.replace('$','')}.csv",
                           mime="text/csv", key="dl_by_strike")

# ── Tab 6: Realized Vol ───────────────────────────────────────────────────────
with tab6:
    has_ohlc = ohlc_spx is not None and isinstance(ohlc_spx, pd.DataFrame) and not ohlc_spx.empty
    has_vix  = vix_hist is not None and isinstance(vix_hist, pd.DataFrame) and not vix_hist.empty

    if not has_ohlc:
        st.info(
            "Dati OHLC storici non disponibili.  \n"
            "Premi **⬇ CARICA FULL CHAIN** per avviare il download da Barchart "
            "(storico 6 mesi $SPX + VIX).  \n"
            "Se il messaggio persiste, l'endpoint Barchart storico potrebbe richiedere "
            "un abbonamento premium — il log del terminale mostra i dettagli."
        )
    else:
        # Rolling window = 6 months (126 trading days)
        # Display horizon = last 1 year (252 trading days)
        rvol_window  = 126
        rvol_df_full = compute_rvol_all(ohlc_spx, window=rvol_window)
        rvol_df      = rvol_df_full.tail(252) if not rvol_df_full.empty else rvol_df_full
        cones_df     = compute_rvol_cones(ohlc_spx)
        intra_rv     = compute_intraday_rvol(intraday) if intraday is not None else None

        # HAR-RV 1-day forecast
        har_result   = compute_har_rv(ohlc_spx)
        har_forecast = har_result.get('forecast')
        har_r2       = har_result.get('r2')

        # Vol regime detection
        yz_series    = rvol_df_full['Yang-Zhang'] if not rvol_df_full.empty and 'Yang-Zhang' in rvol_df_full.columns else pd.Series(dtype=float)
        vol_regime, vol_z = detect_vol_regime(yz_series)
        regime_colors = {'HIGH': '🔴', 'NORMAL': '🟡', 'LOW': '🟢'}
        regime_icon   = regime_colors.get(vol_regime, '⚪')

        # Align VIX to the same 1-year window via nearest-date merge
        vix_plot = pd.DataFrame()
        if has_vix and not rvol_df.empty:
            rvol_dates = rvol_df.index.to_frame(name='date').reset_index(drop=True)
            vix_sorted = vix_hist.sort_values('date')
            vix_plot   = pd.merge_asof(
                rvol_dates,
                vix_sorted,
                on='date',
                direction='nearest',
                tolerance=pd.Timedelta('5D'),
            ).dropna()

        # ── Metric cards ──────────────────────────────────────────────────
        last_yz  = float(rvol_df['Yang-Zhang'].iloc[-1])    if not rvol_df.empty and 'Yang-Zhang'    in rvol_df.columns else None
        last_yz21= float(rvol_df['Yang-Zhang 21d'].iloc[-1]) if not rvol_df.empty and 'Yang-Zhang 21d' in rvol_df.columns else None
        last_ewma= float(rvol_df['EWMA λ=0.94'].iloc[-1])   if not rvol_df.empty and 'EWMA λ=0.94'   in rvol_df.columns else None
        last_yz5 = float(rvol_df['Yang-Zhang 5d'].iloc[-1])  if not rvol_df.empty and 'Yang-Zhang 5d'  in rvol_df.columns else None
        last_vix = float(vix_hist['vix'].iloc[-1] * 100) if has_vix else None
        vol_premium     = (last_vix - last_yz)   if (last_vix and last_yz)   else None
        vol_premium_ewma= (last_vix - last_ewma) if (last_vix and last_ewma) else None

        cards([
            {'label': 'Intraday RVol', 'value': f"{intra_rv*100:.1f}%" if intra_rv else "—",
             'help': 'RVol annualizzata sessione corrente (barre 5-min)'},
            {'label': 'EWMA λ=0.94 ★ più reattivo', 'value': f"{last_ewma:.1f}%" if last_ewma else "—",
             'delta': (f"vs VIX {vol_premium_ewma:+.1f}%" if vol_premium_ewma else None),
             'delta_color': ('red' if vol_premium_ewma and vol_premium_ewma > 0 else 'green'),
             'help': 'RiskMetrics EWMA: half-life ~11 giorni. Il più reattivo agli spike.'},
            {'label': 'Yang-Zhang 21d (≈ VIX window)', 'value': f"{last_yz21:.1f}%" if last_yz21 else "—",
             'help': 'Finestra 21 giorni — confronto più corretto con il VIX 30d'},
            {'label': 'Yang-Zhang 126d', 'value': f"{last_yz:.1f}%" if last_yz else "—",
             'help': 'Finestra 6 mesi — stima lenta, livello strutturale'},
            {'label': 'VIX', 'value': f"{last_vix:.1f}%" if last_vix else "—"},
            {'label': 'Vol Premium (VIX−EWMA)',
             'value': ((f"+{vol_premium_ewma:.1f}%" if vol_premium_ewma > 0 else f"{vol_premium_ewma:.1f}%")
                       if vol_premium_ewma is not None else "—"),
             'delta': f"{vol_premium_ewma:.1f}%" if vol_premium_ewma else None,
             'delta_color': ('green' if vol_premium_ewma and vol_premium_ewma > 0 else 'red'),
             'help': 'VIX vs EWMA: valuta se il mercato paga premium o discount sulla vol'},
        ], per_row=6)

        st.markdown("---")

        # Row 2: HAR forecast + regime + YZ-5d
        cards([
            {'label': 'HAR-RV Forecast (1d)', 'value': f"{har_forecast:.1f}%" if har_forecast else "—",
             'delta': (f"vs EWMA {(har_forecast-last_ewma):+.1f}%" if har_forecast and last_ewma else None),
             'delta_color': 'grey',
             'help': ('HAR-RV Corsi 2009. ' + (f"68% CI [{har_result.get('ci_68_lo',0):.1f}–"
                      f"{har_result.get('ci_68_hi',0):.1f}%] R²={har_r2:.2f}" if har_r2 else ""))},
            {'label': f"{regime_icon} Vol Regime", 'value': vol_regime,
             'delta': f"z = {vol_z:+.2f}", 'delta_color': 'grey',
             'help': 'Z-score rolling 1Y RVol YZ-126d. |z|>1.5 = HIGH/LOW'},
            {'label': 'Yang-Zhang 5d', 'value': f"{last_yz5:.1f}%" if last_yz5 else "—",
             'help': 'Finestra 5 giorni — vol settimanale corrente'},
        ], per_row=3)

        st.markdown("---")

        # ── Modelli RVol vs VIX ───────────────────────────────────────────
        try:
            st.plotly_chart(rvol_models_chart(rvol_df,
                                              vix_plot if not vix_plot.empty else vix_hist,
                                              '$SPX'),
                            use_container_width=True)
        except Exception as e:
            st.plotly_chart(empty_fig(str(e)), use_container_width=True)

        # ── Volatility Cones ──────────────────────────────────────────────
        try:
            st.plotly_chart(rvol_cones_chart(cones_df, '$SPX'),
                            use_container_width=True)
        except Exception as e:
            st.plotly_chart(empty_fig(str(e)), use_container_width=True)

        # ── Backtest ──────────────────────────────────────────────────────
        with st.expander("📊 Backtest — Validazione HAR-RV e Vol Premium", expanded=False):
            bt_col1, bt_col2 = st.columns(2)

            with bt_col1:
                st.markdown("**HAR-RV Out-of-Sample (60/40 split)**")
                har_bt = backtest_har_oos(ohlc_spx)
                if har_bt:
                    cards([
                        {'label': 'RMSE (oos)', 'value': f"{har_bt['rmse']:.2f}%"},
                        {'label': 'MAE (oos)',  'value': f"{har_bt['mae']:.2f}%"},
                        {'label': 'Dir. Acc.',  'value': f"{har_bt['dir_acc']*100:.1f}%"},
                        {'label': 'R² oos',     'value': f"{har_bt['r2_oos']:.3f}"},
                    ], per_row=4)
                    beat = "✅ batte" if har_bt['vs_mean'] else "❌ non batte"
                    st.caption(f"N oos = {har_bt['n_oos']} sessioni  ·  "
                               f"Benchmark RMSE (mean): {har_bt['bench_rmse']:.2f}%  ·  "
                               f"HAR {beat} il random walk")
                else:
                    st.info("Dati insufficienti per il backtest HAR (min 60 sessioni)")

            with bt_col2:
                st.markdown("**Vol Premium Capture (VIX−RVol > 5%)**")
                vp_bt = backtest_vol_premium(rvol_df_full, vix_hist, threshold=5.0)
                if vp_bt and vp_bt.get('signal_days', 0) > 0:
                    st.caption(f"Giorni segnale: {vp_bt['signal_days']}  "
                               f"(soglia +5%)")
                    vp_data = []
                    for h in [5, 10, 22]:
                        avg_rv  = vp_bt.get(f'h{h}_avg_rv')
                        avg_vix = vp_bt.get(f'h{h}_avg_vix')
                        pct_up  = vp_bt.get(f'h{h}_rv_gt_start')
                        if avg_rv:
                            vp_data.append({
                                'Orizzonte': f'{h}d',
                                'RVol media fwd': f'{avg_rv:.1f}%',
                                'VIX medio fwd':  f'{avg_vix:.1f}%' if avg_vix else '—',
                                'RVol > t0':      f'{pct_up*100:.0f}%' if pct_up else '—',
                            })
                    if vp_data:
                        st.dataframe(pd.DataFrame(vp_data),
                                     use_container_width=True, hide_index=True)
                        st.caption("RVol > t0: % volte in cui la RVol è aumentata"
                                   " dopo il segnale (contro la tesi mean-reversion)")
                else:
                    st.info("Segnali insufficienti per l'analisi vol premium")

        # ── Descrizione modelli ───────────────────────────────────────────
        with st.expander("📖 Specifiche dei modelli di Realized Volatility", expanded=False):
            st.markdown("""
<div style="font-family:'Inter',sans-serif; font-size:13px; line-height:1.75;
            color:#374151; padding:4px 0;">

<p style="margin-bottom:10px; color:#6B7280; font-size:12px;">
Tutti i modelli calcolano la volatilità annualizzata su una finestra mobile di
<b>30 giorni di trading</b> (252 giorni/anno). Il <b>VIX</b> è usato come proxy
della volatilità implicita 30-day degli OTM options SPX. Un VIX strutturalmente
sopra la RVol indica un <em>vol risk premium</em> positivo a favore dei venditori
di opzioni.
</p>

<ul style="list-style:none; padding:0; margin:0;">

<li style="margin-bottom:14px;">
  <span style="display:inline-block; width:12px; height:12px; border-radius:50%;
               background:#6C63FF; margin-right:8px; vertical-align:middle;"></span>
  <b>Standard Deviation</b> — modello base.
  Annualizza la deviazione standard dei log-return giornalieri <em>ln(C_t/C_{t-1})</em>
  su finestra mobile. Assume log-normalità e nessun drift. Il più semplice ma anche il
  meno efficiente statisticamente: richiede più osservazioni per ridurre l'errore di
  stima e non sfrutta le informazioni intraday (High/Low/Open).
</li>

<li style="margin-bottom:14px;">
  <span style="display:inline-block; width:12px; height:12px; border-radius:50%;
               background:#10B981; margin-right:8px; vertical-align:middle;"></span>
  <b>Parkinson (1980)</b> — stimatore High-Low.
  Usa solo il range giornaliero <em>ln(H/L)²</em> con fattore correttivo <em>1/(4·ln2)</em>.
  Circa <b>5×</b> più efficiente dello Std Dev sotto diffusione continua senza drift.
  Sottostima la vol in presenza di <em>gap overnight</em> e di trend sostenuti
  perché ignora il close e l'open.
</li>

<li style="margin-bottom:14px;">
  <span style="display:inline-block; width:12px; height:12px; border-radius:50%;
               background:#3B82F6; margin-right:8px; vertical-align:middle;"></span>
  <b>Garman-Klass (1980)</b> — OHLC completo.
  Estende Parkinson aggiungendo il contributo di open e close:
  <em>0.5·ln(H/L)² − (2·ln2−1)·ln(C/O)²</em>. Circa <b>8×</b> più efficiente
  dello Std Dev. Assume assenza di gap overnight e nessun drift — sovrastima
  la vol in presenza di trend direzionali forti.
</li>

<li style="margin-bottom:14px;">
  <span style="display:inline-block; width:12px; height:12px; border-radius:50%;
               background:#F59E0B; margin-right:8px; vertical-align:middle;"></span>
  <b>Hodges-Tompkins (1992)</b> — Std Dev bias-corretto.
  Applica un fattore di correzione <em>√(N/(N−1))</em> per ridurre il downward bias
  introdotto dalle finestre mobili sovrapposte (overlapping windows). Produce stime
  leggermente più elevate dello Std Dev standard, particolarmente rilevante per
  finestre brevi (3-10 giorni) dove il bias è più pronunciato.
</li>

<li style="margin-bottom:14px;">
  <span style="display:inline-block; width:12px; height:12px; border-radius:50%;
               background:#EC4899; margin-right:8px; vertical-align:middle;"></span>
  <b>Rogers-Satchell (1991)</b> — drift non-zero, no overnight.
  Formula: <em>ln(H/O)·ln(H/C) + ln(L/O)·ln(L/C)</em>. Non assume drift nullo —
  stima correttamente la vol anche in presenza di trend direzionali sostenuti.
  Limitazione: non cattura la varianza dei <em>gap overnight</em> (salti
  open-to-previous-close), sottostimando la vol in mercati con gap frequenti.
</li>

<li style="margin-bottom:14px;">
  <span style="display:inline-block; width:12px; height:12px; border-radius:50%;
               background:#EAB308; margin-right:8px; vertical-align:middle;"></span>
  <b>Yang-Zhang (2000)</b> — stimatore di minima varianza ★
  Il modello più completo. Combina tre componenti con peso ottimale
  <em>k = 0.34/(1.34 + (N+1)/(N−1))</em>:
  <ul style="margin:4px 0 0 20px; padding:0; list-style:disc;">
    <li>Varianza <b>overnight</b> <em>ln(O_t/C_{t-1})</em> — cattura i gap</li>
    <li>Varianza <b>open-to-close</b> <em>ln(C/O)</em> — componente direzionale</li>
    <li>Componente <b>Rogers-Satchell</b> — varianza intraday con drift</li>
  </ul>
  Invariante rispetto al drift, minima varianza tra tutti gli stimatori OHLC,
  gestisce sia gap overnight che trend intraday. <b>Usato come modello di riferimento
  per i Volatility Cones.</b>
</li>

</ul>
</div>
""", unsafe_allow_html=True)

# ── Tab 7: Paper Trading (visualizzazione — engine automatico) ────────────────
with tab7:
    st.markdown("### 🤖 Paper Trading — Long Volatility (automatico)")
    st.caption(
        "Il motore di simulazione gira automaticamente ogni giorno feriale dopo la "
        "chiusura USA tramite GitHub Actions: valuta il segnale long-vol, apre/chiude "
        "le posizioni simulate e committa lo storico nel repo. Questa tab mostra lo "
        "stato — non serve premere nulla. Segnale da SPX, conto paper da 5.000 $ su XSP.")

    try:
        import trading_lite as tl

        _pos = tl.load_positions()
        _cs  = tl.capital_status()

        # ── Live unrealized P&L for open positions (read-only) ────────────────
        _unreal = {'by_id': {}, 'total_unrealized': 0.0, 'n_open': 0}
        _has_data = ("data" in st.session_state and
                     st.session_state["data"].get("raw_full") is not None)
        if _has_data and _cs['n_open'] > 0:
            _dd0 = st.session_state["data"]
            _spot0 = _dd0.get("spot", 0)
            _m00 = compute_0dte_metrics(_dd0.get("raw_full"), _spot0) or {}
            _iv0 = (_m00.get('atm_iv') or 0.15)
            _iv0 = _iv0 * 100 if _iv0 < 1 else _iv0
            try:
                _unreal = tl.compute_unrealized(_spot0, _iv0)
            except Exception:
                pass

        # ── Capital status ────────────────────────────────────────────────────
        _cap_color = ('#10B981' if _cs['is_safe'] and not _cs['danger_zone']
                      else '#F59E0B' if _cs['is_safe'] else '#EF4444')
        _total_equity = _cs['account_value'] + _unreal['total_unrealized']
        cards([
            {'label': 'Capitale paper', 'value': f"${_cs['account_value']:,.0f}",
             'help': '5.000 $ iniziali + P&L realizzato dei trade chiusi'},
            {'label': 'P&L realizzato', 'value': f"${_cs['cumulative_pnl']:+,.0f}",
             'delta_color': ('green' if _cs['cumulative_pnl'] >= 0 else 'red'),
             'help': 'Somma dei trade chiusi'},
            {'label': 'P&L latente (aperte)',
             'value': f"${_unreal['total_unrealized']:+,.0f}" if _has_data else "—",
             'delta_color': ('green' if _has_data and _unreal['total_unrealized'] >= 0 else 'red'),
             'help': 'Valore attuale delle posizioni aperte (richiede chain caricata)'},
            {'label': 'Equity totale',
             'value': f"${_total_equity:,.0f}" if _has_data else f"${_cs['account_value']:,.0f}",
             'help': 'Capitale + P&L latente'},
            {'label': 'Posizioni', 'value': f"{_cs['n_open']} aperte / {_cs['n_closed']} chiuse"},
        ], per_row=5)

        if not _cs['is_safe']:
            st.error(f"🚨 Hard stop: drawdown {_cs['drawdown_pct']:.0f}% ha superato il 50%.")
        elif _cs['danger_zone']:
            st.warning(f"⚠️ Zona di pericolo: drawdown {_cs['drawdown_pct']:.0f}%.")
        if not _has_data and _cs['n_open'] > 0:
            st.caption("💡 Carica la chain (CARICA FULL CHAIN) per vedere il P&L latente "
                       "in tempo reale delle posizioni aperte.")

        # ── Oversized legacy positions warning ────────────────────────────────
        _oversized = tl.find_oversized_open()
        if _oversized:
            st.warning(
                f"⚠️ **{len(_oversized)} posizione/i sovradimensionata/e** — aperte da una "
                f"versione precedente del motore, con premio oltre il limite attuale "
                f"dell'{tl.MAX_PREMIUM_PCT:.0f}% del capitale. Queste sono la causa di "
                f"perdite sproporzionate rispetto a un conto da ${tl.INITIAL_CAPITAL:.0f}.")
            for _ov in _oversized:
                _c1, _c2 = st.columns([3, 1])
                _c1.markdown(
                    f"Trade **#{_ov['id']}** ({_ov['strategy']}): premio "
                    f"${_ov['entry_premium_usd']:.0f} = **{_ov['pct_of_capital']:.0f}%** "
                    f"del capitale (limite ${_ov['cap_usd']:.0f})")
                if _has_data:
                    if _c2.button(f"Chiudi #{_ov['id']}", key=f"close_ov_{_ov['id']}"):
                        _dd_ov = st.session_state["data"]
                        _m0_ov = compute_0dte_metrics(_dd_ov.get("raw_full"),
                                                      _dd_ov.get("spot", 0)) or {}
                        _iv_ov = (_m0_ov.get('atm_iv') or 0.15)
                        _iv_ov = _iv_ov*100 if _iv_ov < 1 else _iv_ov
                        _res = tl.close_position_manual(_ov['id'],
                                                        _dd_ov.get("spot", 0), _iv_ov)
                        if _res.get('closed'):
                            st.success(f"Posizione #{_ov['id']} chiusa, "
                                       f"P&L ${_res['pnl_usd']:+.0f}.")
                            st.rerun()
                else:
                    _c2.caption("Carica chain per chiudere")

        # ── Live signal preview (informational — engine decides at close) ─────
        _has_data = ("data" in st.session_state and
                     st.session_state["data"].get("raw_full") is not None)
        if _has_data:
            _dd = st.session_state["data"]
            _spot = _dd.get("spot", 0)
            _ga = compute_gex_analytics(_dd.get("raw_full"), _dd.get("by_strike"), _spot) or {}
            _m0 = compute_0dte_metrics(_dd.get("raw_full"), _spot) or {}
            _regime_full = _ga.get('regime')
            _regime = 'LONG' if (_regime_full and 'LONG' in _regime_full) else (
                      'SHORT' if _regime_full else
                      ('LONG' if _ga.get('net_gex_total', 0) >= 0 else 'SHORT'))
            _hhi = _ga.get('hhi', 0.0)
            _atm_iv = _m0.get('atm_iv')
            _vp = None
            _vix_h = _dd.get("vix_hist"); _ohlc = _dd.get("ohlc_spx")
            if _vix_h is not None and not _vix_h.empty and _ohlc is not None and not _ohlc.empty:
                try:
                    _rv = compute_rvol_all(_ohlc, window=126)
                    _vixv = float(_vix_h["vix"].iloc[-1] * 100)
                    if "EWMA λ=0.94" in _rv.columns:
                        _vp = _vixv - float(_rv["EWMA λ=0.94"].iloc[-1])
                except Exception:
                    pass
            # VolDex suite — same signals the automatic engine uses
            _vx_prev = compute_voldex(_dd.get("raw_full"), _spot)
            _voldex_p = _calldex_p = _putdex_p = _taildex_p = _skewtr_p = None
            if not _vx_prev.get('error'):
                _voldex_p  = _vx_prev['voldex']
                _calldex_p = _vx_prev['calldex']
                _putdex_p  = _vx_prev['putdex']
                _taildex_p = _vx_prev['taildex']
                try:
                    _vhist = load_voldex_history()
                    if not _vhist.empty and len(_vhist) >= 3 and _putdex_p and _calldex_p:
                        _today_skew = _putdex_p - _calldex_p
                        _hskew = (_vhist['putdex'] - _vhist['calldex']).dropna()
                        if len(_hskew) >= 2:
                            _skewtr_p = round(_today_skew - float(_hskew.tail(10).mean()), 2)
                except Exception:
                    pass

            _dec = tl.evaluate_signal(_spot, _regime, _hhi, _vp, _atm_iv,
                                      _m0.get('exp_move_pts'),
                                      voldex=_voldex_p, calldex=_calldex_p,
                                      putdex=_putdex_p, taildex=_taildex_p,
                                      skew_trend=_skewtr_p)
            if _voldex_p is not None:
                st.caption(f"📐 Segnali VolDex usati nell'anteprima: VolDex {_voldex_p:.1f}% · "
                          f"CallDex {_calldex_p:.1f}% · PutDex {_putdex_p:.1f}% · "
                          f"TailDex {_taildex_p:.1f}%"
                          + (f" · Skew trend {_skewtr_p:+.2f}pt" if _skewtr_p is not None else ""))
            st.markdown("---")
            _preview_banner = (
                '<div style="background:#EFF6FF;border:1px solid #BFDBFE;'
                'border-left:4px solid #3B82F6;border-radius:10px;padding:12px 16px;'
                'margin-bottom:14px;">'
                '<div style="color:#1D4ED8;font-size:12px;font-weight:700;'
                'text-transform:uppercase;letter-spacing:0.7px;margin-bottom:3px;">'
                '&#128269; Anteprima segnale &mdash; NON ancora eseguito</div>'
                '<div style="color:#3B82F6;font-size:11.5px;line-height:1.4;">'
                'Questa &egrave; una simulazione di cosa farebbe il sistema con i dati '
                'attuali. <b>Non &egrave; una posizione aperta.</b> Le posizioni reali '
                'le apre solo il motore automatico dopo la chiusura USA, e compaiono '
                'nelle card "Posizioni" e nello "Storico" qui sotto.</div>'
                '</div>'
            )
            st.markdown(_preview_banner, unsafe_allow_html=True)
            cards([
                {'label': 'Regime GEX', 'value': _regime},
                {'label': 'HHI', 'value': f"{_hhi:.4f}"},
                {'label': 'Vol Premium', 'value': f"{_vp:+.1f}%" if _vp is not None else "—"},
                {'label': 'ATM IV',
                 'value': (f"{(_atm_iv*100 if _atm_iv and _atm_iv<1 else _atm_iv):.1f}%"
                           if _atm_iv else "—")},
            ], per_row=4)
            st.markdown(_dec.summary_html, unsafe_allow_html=True)

            # Explicit option legs (what would be bought/sold)
            if _dec.action == 'TRADE' and _dec.legs:
                st.markdown("<span style='font-size:12px;color:#3B82F6;'>"
                            "<b>Gambe che il sistema aprirebbe (XSP) &mdash; ipotetiche:"
                            "</b></span>", unsafe_allow_html=True)
                _legrows = [{
                    'Lato': l['side'], 'Tipo': l['type'],
                    'Strike': f"{l['strike']:.0f}",
                    'Premio (punti)': f"{l['premium_pts']:.2f}",
                    'Premio ($)': f"${l['premium_usd']:.0f}",
                } for l in _dec.legs]
                _legrows.append({
                    'Lato': '', 'Tipo': 'TOTALE', 'Strike': '',
                    'Premio (punti)': f"{_dec.est_premium:.2f}",
                    'Premio ($)': f"${_dec.est_premium*100:.0f}",
                })
                st.dataframe(pd.DataFrame(_legrows), use_container_width=True,
                             hide_index=True)

            # Per-condition checklist + explanation
            if _dec.conditions:
                _cccols = st.columns(2)
                for _i, _c in enumerate(_dec.conditions):
                    _icon = "✅" if _c['met'] else "❌"
                    _cccols[_i % 2].markdown(
                        f"<span style='font-size:12px;'>{_icon} <b>{_c['label']}</b><br>"
                        f"<span style='color:#6B7280;'>{_c['detail']}</span></span>",
                        unsafe_allow_html=True)
            if _dec.explanation:
                if _dec.action == 'TRADE':
                    _exp_bg, _exp_border, _exp_text = '#ECFDF5', '#10B981', '#065F46'
                else:
                    _exp_bg, _exp_border, _exp_text = '#F3F4F6', '#9CA3AF', '#374151'
                st.markdown(
                    f"<div style='background:{_exp_bg};border-left:3px solid {_exp_border};"
                    f"border-radius:4px;padding:10px 14px;margin-top:8px;font-size:13px;"
                    f"color:{_exp_text};'>"
                    + _dec.explanation.replace(chr(10), '<br>') + "</div>",
                    unsafe_allow_html=True)

        st.markdown("---")

        # ── Positions history ─────────────────────────────────────────────────
        if not _pos.empty:
            st.markdown("**📋 Storico posizioni simulate** "
                        "<span style='font-size:11px;color:#9CA3AF;'>(generato "
                        "automaticamente dal motore)</span>", unsafe_allow_html=True)
            _show = _pos[['id','open_date','strategy','xsp_strike','entry_premium',
                          'confidence','status','close_date','pnl_usd','exit_reason']].copy()
            # Add live unrealized P&L column for open positions
            def _live_pnl(row):
                if str(row['status']) == 'OPEN':
                    info = _unreal['by_id'].get(int(row['id']))
                    if info:
                        return f"${info['unrealized']:+.0f} ({info['ratio']:.1f}×)"
                    return "—"
                return ""
            _show['P&L latente'] = _pos.apply(_live_pnl, axis=1)
            # Add peak ratio reached (relevant for trailing-stop transparency)
            if 'peak_ratio' in _pos.columns:
                _show['Picco'] = _pos['peak_ratio'].apply(
                    lambda v: f"{float(v):.2f}×" if pd.notna(v) and str(v).strip() != '' else "—")
            st.dataframe(_show, use_container_width=True, hide_index=True)
            st.caption("**Regole di uscita:** profit target 2.5× · trailing stop "
                       "(si arma a 1.5×, chiude se ripiega 25% dal picco) · uscita VolDex "
                       "(vol salita +40% dall'entrata) · stop loss 50% · DTE floor 14g. "
                       "La colonna *Picco* mostra il massimo raggiunto da ogni posizione.")

            # Backfill: reconstruct legs for past trades missing them
            _missing_legs = 0
            if 'legs_json' in _pos.columns:
                _missing_legs = int(_pos['legs_json'].apply(
                    lambda x: not (isinstance(x, str) and x.strip())).sum())
            else:
                _missing_legs = len(_pos)
            if _missing_legs > 0:
                _bc1, _bc2 = st.columns([3, 1])
                _bc1.caption(f"⚠️ {_missing_legs} trade passati non hanno il dettaglio "
                             "delle gambe (registrati prima dell'aggiornamento).")
                if _bc2.button("🔧 Ricostruisci gambe", key="backfill_legs"):
                    _res = tl.backfill_legs()
                    st.success(f"Ricostruite le gambe per {_res['filled']} trade "
                               f"(saltati {_res['skipped']}). Sono STIME, non i dati "
                               f"originali del momento dell'apertura.")
                    st.rerun()

            # Per-trade decision rationale (why each trade was opened)
            if 'decision_note' in _pos.columns:
                with st.expander("📝 Motivazione di ogni trade (perché è stato aperto)",
                                 expanded=False):
                    for _, _r in _pos.iterrows():
                        _note = _r.get('decision_note')
                        if isinstance(_note, str) and _note.strip():
                            st.markdown(
                                f"**Trade #{int(_r['id'])}** — {_r['open_date']} · "
                                f"{_r['strategy']} · XSP {_r['xsp_strike']}")
                            # Show explicit legs if available
                            _lj = _r.get('legs_json')
                            if isinstance(_lj, str) and _lj.strip():
                                try:
                                    import json as _json
                                    _legs = _json.loads(_lj)
                                    _is_recon = any(l.get('reconstructed') for l in _legs)
                                    _lr = [{
                                        'Lato': l['side'], 'Tipo': l['type'],
                                        'Strike': f"{l['strike']:.0f}",
                                        'Premio (punti)': f"{l['premium_pts']:.2f}",
                                        'Premio ($)': f"${l['premium_usd']:.0f}",
                                    } for l in _legs]
                                    st.dataframe(pd.DataFrame(_lr),
                                                 use_container_width=True, hide_index=True)
                                    if _is_recon:
                                        st.caption("⚠️ Gambe RICOSTRUITE (stime ricalcolate, "
                                                   "non i dati originali dell'apertura).")
                                except Exception:
                                    pass
                            st.markdown(
                                f"<div style='font-size:12px;color:#4B5563;"
                                f"margin-bottom:10px;'>"
                                + str(_note).replace(chr(10), '<br>') + "</div>",
                                unsafe_allow_html=True)

            # Equity curve from closed trades
            _closed = _pos[_pos['status'] == 'CLOSED'].copy()
            if not _closed.empty:
                _closed['pnl_num'] = pd.to_numeric(_closed['pnl_usd'], errors='coerce').fillna(0)
                _eq = _closed['pnl_num'].cumsum().tolist()
                _efig = go.Figure()
                _efig.add_scatter(y=_eq, mode='lines+markers',
                                  line=dict(color='#6C63FF', width=2.5),
                                  fill='tozeroy', fillcolor='rgba(108,99,255,0.08)')
                _efig.add_hline(y=0, line_color='#E5E7EB', line_width=1)
                _efig.update_layout(height=240, paper_bgcolor='rgba(0,0,0,0)',
                                    plot_bgcolor='#F9FAFB',
                                    yaxis_title='P&L cumulativo ($)', xaxis_title='Trade #',
                                    margin=dict(t=10,b=30,l=50,r=10), showlegend=False)
                st.plotly_chart(_efig, use_container_width=True)

            _exp1, _exp2 = st.columns([2, 2])
            _exp1.download_button("⬇ Esporta CSV", _pos.to_csv(index=False).encode(),
                               file_name="paper_trades.csv", mime="text/csv", key="dl_paper")
            with _exp2:
                with st.popover("🗄 Archivia e riparti pulito"):
                    st.markdown("**Archivia lo storico e ricomincia da zero**")
                    st.caption("Lo storico attuale viene salvato in un file "
                               "`positions_archive_...csv` (nulla viene cancellato) e "
                               "il paper trading riparte da 5.000 $ con zero posizioni. "
                               "Utile per lasciarsi alle spalle le posizioni del vecchio bug "
                               "e validare il sistema corretto da una linea netta.")
                    _confirm = st.checkbox("Confermo: archivia e azzera lo storico",
                                           key="confirm_archive")
                    if st.button("🗄 Procedi", key="do_archive", disabled=not _confirm):
                        _res = tl.archive_and_reset()
                        if _res['archived']:
                            st.success(f"Archiviate {_res['rows_archived']} posizioni in "
                                       f"`{os.path.basename(_res['archive_path'])}`. "
                                       "Storico azzerato — si riparte pulito.")
                            st.rerun()
                        else:
                            st.info(_res.get('reason', 'Niente da archiviare.'))
            _archives = tl.list_archives()
            if _archives:
                st.caption(f"📁 Archivi salvati: {len(_archives)} "
                           f"(più recente: {_archives[0]})")
        else:
            st.info("Nessuna posizione ancora. Il motore automatico registrerà il primo "
                    "trade quando il segnale sarà favorevole (la maggior parte dei giorni "
                    "è SKIP — comportamento corretto per il long-vol).")

        # ── Process metrics ───────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("**📊 Metriche di processo (fase di validazione)**")
        _pm = tl.process_metrics(SNAPSHOT_DIR)
        cards([
            {'label': 'Snapshot raccolti', 'value': f"{_pm['snapshots']}"},
            {'label': 'Trade chiusi', 'value': f"{_pm['n_closed']}/{_pm['sample_target']}",
             'delta': f"{_pm['sample_pct']:.0f}%", 'delta_color': 'grey'},
            {'label': 'Win rate',
             'value': f"{_pm['win_rate']*100:.0f}%" if _pm['win_rate'] is not None else "—"},
            {'label': 'Posizioni aperte', 'value': f"{_pm['n_open']}"},
        ], per_row=4)
        st.caption("Obiettivo Fase 1: ≥ 20 trade chiusi, processo coerente. "
                   "Il profitto è secondario — conta validare che la logica funzioni in automatico.")

        # ── SKIP analysis — is the selection logic too strict, and why? ───────
        st.markdown("---")
        st.markdown("**🔍 Analisi delle decisioni (perché entra o salta)**")
        _sk = tl.skip_log_summary()
        if _sk['total_days'] == 0:
            st.caption("Ancora nessuna decisione registrata. Il motore automatico "
                       "registra ogni giorno se ha fatto TRADE o SKIP e perché — "
                       "questi dati servono a capire se i filtri aiutano o tagliano "
                       "troppo. Si popolerà con i prossimi run automatici.")
        else:
            cards([
                {'label': 'Giorni valutati', 'value': f"{_sk['total_days']}"},
                {'label': 'Trade', 'value': f"{_sk['n_trade']}",
                 'delta': f"{_sk['trade_rate']:.0f}% dei giorni", 'delta_color': 'grey'},
                {'label': 'SKIP', 'value': f"{_sk['n_skip']}"},
            ], per_row=3)
            if _sk['skip_breakdown']:
                st.caption("**Motivi dei SKIP** — se un motivo domina, è la leva su "
                           "cui ragionare (es. troppi 'Conviction bassa' → soglia forse "
                           "troppo alta; troppi 'IV troppo cara' → mercato sfavorevole, "
                           "lo SKIP è corretto):")
                _bd = pd.DataFrame(
                    [{'Motivo': k, 'Giorni': v,
                      '% sui SKIP': f"{v/_sk['n_skip']*100:.0f}%"}
                     for k, v in sorted(_sk['skip_breakdown'].items(),
                                        key=lambda x: -x[1])])
                st.dataframe(_bd, use_container_width=True, hide_index=True)
            _sklog = tl.load_skip_log()
            if not _sklog.empty:
                st.download_button("⬇ Esporta log decisioni",
                                   _sklog.to_csv(index=False).encode(),
                                   file_name="skip_log.csv", mime="text/csv",
                                   key="dl_skiplog")

        # ── Naive benchmark — does the smart system beat doing the dumb thing? ─
        st.markdown("---")
        st.markdown("**📏 Benchmark — il sistema batte la versione 'stupida'?**")
        _bm = tl.compute_naive_benchmark(SNAPSHOT_DIR)
        if not _bm.get('available'):
            st.caption(f"⏳ {_bm.get('reason', 'Benchmark non ancora disponibile.')} "
                       "Il confronto con una strategia long-vol naive (struttura fissa "
                       "a cadenza regolare, senza segnali) è la verifica chiave: se il "
                       "sistema sofisticato non batte quello stupido, i segnali "
                       "VolDex/GEX non stanno aggiungendo valore.")
        else:
            st.caption(f"Serie snapshot pronta ({_bm['n_snapshots']} giorni, dal "
                       f"{_bm['first']} al {_bm['last']}). {_bm['note']}")
        st.info("💡 La domanda di validazione che conta non è *'ho guadagnato?'* ma "
                "*'la mia logica di selezione aggiunge valore rispetto al caso e a una "
                "strategia banale?'*. Per questo servono: abbastanza trade chiusi, il "
                "log delle decisioni qui sopra, e il confronto con il benchmark.")

    except ImportError:
        st.error("Modulo trading_lite.py non trovato nella root del progetto.")
    except Exception as _te:
        st.error(f"Errore: {_te}")
        import traceback as _tb
        st.code(_tb.format_exc())


# ── Tab 8: VolDex Suite (replica indipendente, SPX) ───────────────────────────
with tab8:
    st.markdown("### 🌊 VolDex Suite — Implied Volatility ATM (replica SPX)")
    st.caption(
        "Replica indipendente della metodologia pubblicata da Nations Indexes "
        "(VolDex®) e Nasdaq (VOLQ®): volatilità implicita ATM a 30 giorni, "
        "calcolata con la formula closed-form di Brenner-Subrahmanyam su opzioni "
        "esattamente at-the-money, interpolata su più scadenze. Qui applicata "
        "all'indice SPX usando i dati già caricati dalla dashboard.")

    with st.expander("ℹ️ Nota su trademark e differenze rispetto all'indice ufficiale",
                     expanded=False):
        st.markdown(
            "<div style='color:#1A1A2E;font-size:13px;line-height:1.6;'>"
            "<ul style='margin:0;padding-left:18px;'>"
            "<li><b>Non è il prodotto Nations Indexes/Nasdaq licenziato.</b> "
            "VolDex®, VOLQ®, CallDex®, PutDex® e TailDex® sono marchi registrati "
            "dei rispettivi proprietari.</li>"
            "<li>Questa è un'implementazione originale della <b>metodologia "
            "pubblicata</b> (formula di Brenner-Subrahmanyam su strike ATM, pesati "
            "con kernel triangolare, interpolati a 30 giorni), calcolata qui su "
            "<b>SPX</b> invece che su NDX.</li>"
            "<li>I prezzi usati sono <b>mid-quote</b> dal feed dati di questa "
            "dashboard, non NBBO real-time come l'indice ufficiale.</li>"
            "<li>I valori assoluti <b>non coincideranno</b> con il ticker "
            "VOLQ/VolDex ufficiale — è una misura indipendente, stessa metodologia, "
            "sottostante diverso.</li>"
            "</ul></div>",
            unsafe_allow_html=True)

    try:
        _has_data8 = ("data" in st.session_state and
                      st.session_state["data"].get("raw_full") is not None)

        if not _has_data8:
            st.info("Premi **CARICA FULL CHAIN** per calcolare il VolDex corrente.")
        else:
            _dd8 = st.session_state["data"]
            _spot8 = _dd8.get("spot", 0)
            _raw8  = _dd8.get("raw_full")

            _vx = compute_voldex(_raw8, _spot8)

            if _vx.get('error'):
                st.warning(f"⚠️ {_vx['error']}")
            else:
                # ── Headline metrics ────────────────────────────────────────────
                cards([
                    {'label': 'VolDex (ATM 30d)', 'value': f"{_vx['voldex']:.2f}%",
                     'help': 'Volatilità implicita ATM a 30 giorni — ciò che i practitioner guardano per primo'},
                    {'label': 'CallDex (~16Δ call)',
                     'value': f"{_vx['calldex']:.2f}%" if _vx['calldex'] else "—",
                     'help': 'Costo normalizzato della call OTM ~1 dev. std'},
                    {'label': 'PutDex (~16Δ put)',
                     'value': f"{_vx['putdex']:.2f}%" if _vx['putdex'] else "—",
                     'help': 'Costo normalizzato della put OTM ~1 dev. std'},
                    {'label': 'TailDex (~10Δ put)',
                     'value': f"{_vx['taildex']:.2f}%" if _vx['taildex'] else "—",
                     'help': 'Costo della put più OTM — proxy di tail risk'},
                ], per_row=4)

                # Skew reading
                if _vx['putdex'] and _vx['calldex']:
                    _skew_pts = _vx['putdex'] - _vx['calldex']
                    _skew_txt = (f"Skew Put−Call: **{_skew_pts:+.2f} punti**. "
                                 + ("Skew tipico (put più care — copertura al ribasso "
                                    "più richiesta)." if _skew_pts > 0 else
                                    "Skew invertito — raro, da verificare."))
                    st.caption(_skew_txt)

                st.markdown("---")

                # ── Save + historical chart ─────────────────────────────────────
                _vcol1, _vcol2 = st.columns([3, 1])
                with _vcol2:
                    if st.button("💾 Salva snapshot di oggi", key="save_voldex"):
                        save_voldex_snapshot(_vx)
                        st.success("Salvato nello storico.")
                        st.rerun()

                _hist = load_voldex_history()
                with _vcol1:
                    st.caption(f"Storico: {len(_hist)} giorni salvati. "
                               "Premi 'Salva snapshot' ogni volta che carichi la chain "
                               "per far crescere la serie (oppure automatizza via "
                               "GitHub Actions, vedi sotto).")

                st.plotly_chart(voldex_history_chart(_hist), use_container_width=True)

                if not _hist.empty:
                    st.dataframe(_hist.tail(10).iloc[::-1], use_container_width=True,
                                hide_index=True)
                    st.download_button("⬇ Esporta storico CSV",
                                       _hist.to_csv(index=False).encode(),
                                       file_name="voldex_history.csv", mime="text/csv",
                                       key="dl_voldex")

                # ── Diagnostic table (transparency) ─────────────────────────────
                with st.expander("🔍 Dettaglio calcolo per scadenza (trasparenza)",
                                 expanded=False):
                    st.caption(
                        "Per ogni scadenza: forward price (via put-call parity), "
                        "gli strike usati per il prezzo ATM sintetico, i pesi del "
                        "kernel triangolare, e la volatilità implicita closed-form "
                        "(Brenner-Subrahmanyam) per call e put.")
                    _trows = []
                    for _t in _vx['terms']:
                        _trows.append({
                            'Scadenza': _t['expiry'],
                            'T (giorni)': f"{_t['T_days']:.1f}",
                            'Forward': f"{_t['forward']:.1f}",
                            'Strike usati': ', '.join(f"{s:.0f}" for s in _t['strikes']),
                            'Pesi': ', '.join(f"{w:.2f}" for w in _t['weights']),
                            'ATM Call': f"{_t['atm_call']:.2f}",
                            'ATM Put': f"{_t['atm_put']:.2f}",
                            'CFIV Call': f"{_t['cfiv_call']*100:.2f}%",
                            'CFIV Put': f"{_t['cfiv_put']*100:.2f}%",
                        })
                    st.dataframe(pd.DataFrame(_trows), use_container_width=True,
                                hide_index=True)

                st.caption(
                    "💡 Per accumulare lo storico in automatico, estendi il workflow "
                    "`snapshot.yml` perché chiami anche `compute_voldex()` e "
                    "`save_voldex_snapshot()` subito dopo aver scaricato la chain "
                    "giornaliera, così come già fa per gli snapshot delle opzioni.")

    except Exception as _vxe:
        st.error(f"Errore nel calcolo VolDex: {_vxe}")
        import traceback as _tb
        st.code(_tb.format_exc())


# ── Tab 9: Short Vol (sistema speculare, conto separato) ───────────────────────
with tab9:
    import short_vol_lite as sv
    import dex_gex_dashboard as dg
    st.markdown("### 📉 Short Vol — sistema speculare")
    st.caption("Sistema simulato che VENDE volatilità con strutture a rischio "
               "definito (credit spread, iron condor) quando le condizioni sono "
               "opposte al long-vol: vol cara + regime LONG gamma + niente stress. "
               "Conto separato da $5.000, indipendente dal long-vol.")

    # Banner che chiarisce la relazione coi due sistemi
    st.markdown(
        '<div style="background:#FEF3C7;border-left:4px solid #F59E0B;'
        'border-radius:8px;padding:11px 14px;margin-bottom:14px;">'
        '<span style="color:#92400E;font-size:12.5px;line-height:1.45;">'
        '⚖️ <b>Sistema complementare al long-vol.</b> Guadagna nei periodi calmi in '
        'cui il long-vol brucia theta, e perde negli shock in cui il long-vol esplode. '
        'I due conti sono separati e si validano in modo indipendente.</span></div>',
        unsafe_allow_html=True)

    # Account summary
    _has_data_sv = ("data" in st.session_state and
                    st.session_state["data"].get("raw_full") is not None)
    _spot_sv = st.session_state["data"].get("spot") if _has_data_sv else None
    _iv_sv = None
    if _has_data_sv:
        try:
            _m0_sv = compute_0dte_metrics(st.session_state["data"].get("raw_full"),
                                          _spot_sv) or {}
            _iv_sv = (_m0_sv.get('atm_iv') or 0.18) * 100
        except Exception:
            _iv_sv = 18.0

    _sv_acc = sv.account_summary(_spot_sv, _iv_sv)
    cards([
        {'label': 'Capitale Short', 'value': f"${_sv_acc['capital']:,.0f}"},
        {'label': 'P&L realizzato',
         'value': f"${_sv_acc['realized']:+,.0f}",
         'delta_color': 'green' if _sv_acc['realized'] >= 0 else 'red'},
        {'label': 'P&L latente (aperte)',
         'value': f"${_sv_acc['unrealized']:+,.0f}",
         'delta_color': 'green' if _sv_acc['unrealized'] >= 0 else 'red'},
        {'label': 'Equity totale', 'value': f"${_sv_acc['equity']:,.0f}"},
        {'label': 'Posizioni',
         'value': f"{_sv_acc['n_open']} aperte / {_sv_acc['n_closed']} chiuse"},
    ], per_row=5)

    st.markdown("---")

    # Live signal preview
    if _has_data_sv:
        _dd_sv = st.session_state["data"]
        try:
            _raw_sv = _dd_sv.get("raw_full")
            _bs_sv = dg.aggregate_by_strike(_raw_sv)
            _ga_sv = dg.compute_gex_analytics(_raw_sv, _bs_sv, _spot_sv) or {}
            _reg_full_sv = _ga_sv.get('regime')
            _regime_sv = ('LONG' if (_reg_full_sv and 'LONG' in _reg_full_sv) else
                          'SHORT' if _reg_full_sv else
                          ('LONG' if _ga_sv.get('net_gex_total', 0) >= 0 else 'SHORT'))
            _hhi_sv = _ga_sv.get('hhi', 0.0)
            # vol premium + voldex
            _vp_sv = None
            _voldex_sv = _calldex_sv = _putdex_sv = _taildex_sv = None
            try:
                _vx_sv = dg.compute_voldex(_raw_sv, _spot_sv)
                if _vx_sv:
                    _voldex_sv = _vx_sv.get('voldex')
                    _calldex_sv = _vx_sv.get('calldex')
                    _putdex_sv = _vx_sv.get('putdex')
                    _taildex_sv = _vx_sv.get('taildex')
            except Exception:
                pass
            _atm_iv_sv = _m0_sv.get('atm_iv') if _has_data_sv else None
            _em_sv = _m0_sv.get('exp_move_pts') if _has_data_sv else None
            try:
                _vph = dg.load_voldex_history()
                if _vph is not None and len(_vph) >= 3 and _putdex_sv and _calldex_sv:
                    _recent = _vph.tail(10)
                    _sk_now = _putdex_sv - _calldex_sv
                    _sk_avg = (_recent['putdex'] - _recent['calldex']).mean()
                    _skew_trend_sv = _sk_now - _sk_avg
                else:
                    _skew_trend_sv = None
            except Exception:
                _skew_trend_sv = None

            # Term structure state for crash-protection filter
            _term_sv = None
            _term_state_sv = None
            try:
                if _vx_sv:
                    _term_sv = dg.compute_term_structure_slope(_vx_sv)
                    _term_state_sv = _term_sv.get('state') if _term_sv else None
            except Exception:
                pass

            st.markdown(
                '<div style="background:#EFF6FF;border-left:4px solid #3B82F6;'
                'border-radius:10px;padding:12px 16px;margin-bottom:14px;">'
                '<div style="color:#1D4ED8;font-size:12px;font-weight:700;'
                'text-transform:uppercase;letter-spacing:0.7px;margin-bottom:3px;">'
                '&#128269; Anteprima segnale Short Vol &mdash; NON ancora eseguito</div>'
                '<div style="color:#3B82F6;font-size:11.5px;line-height:1.4;">'
                'Simulazione di cosa farebbe il sistema short-vol con i dati attuali. '
                '<b>Non &egrave; una posizione aperta.</b> Le posizioni reali le apre '
                'il motore automatico dopo la chiusura USA.</div></div>',
                unsafe_allow_html=True)

            _dec_sv = sv.evaluate_signal(_spot_sv, _regime_sv, _hhi_sv, _vp_sv,
                                         _atm_iv_sv, _em_sv, voldex=_voldex_sv,
                                         calldex=_calldex_sv, putdex=_putdex_sv,
                                         taildex=_taildex_sv, skew_trend=_skew_trend_sv,
                                         term_state=_term_state_sv)

            cards([
                {'label': 'Regime GEX', 'value': _regime_sv,
                 'delta': 'favorevole' if _regime_sv == 'LONG' else 'sfavorevole',
                 'delta_color': 'green' if _regime_sv == 'LONG' else 'red'},
                {'label': 'HHI (pinning)', 'value': f"{_hhi_sv:.4f}"},
                {'label': 'VolDex', 'value': f"{_voldex_sv:.1f}%" if _voldex_sv else "—"},
                {'label': 'Skew trend',
                 'value': (f"{_skew_trend_sv:+.2f}pt" if _skew_trend_sv is not None else "—")},
                {'label': 'Term structure',
                 'value': ({'contango': '📈 Contango', 'backwardation': '📉 Backwardation',
                            'flat': '➖ Flat'}.get(_term_state_sv, '—')),
                 'delta': (f"{_term_sv['slope']:+.1f}pt" if _term_sv and _term_sv.get('slope') is not None else None),
                 'delta_color': ('green' if _term_state_sv == 'contango'
                                 else 'red' if _term_state_sv == 'backwardation' else 'grey')},
            ], per_row=5)

            # Next macro event card (FOMC/CPI/NFP from official 2026 calendar)
            _nev = sv.next_macro_event()
            if _nev:
                _ev_block = _nev['days'] <= sv.EVENT_BLOCK_DAYS_BEFORE
                _when_txt = 'oggi' if _nev['days'] == 0 else f"tra {_nev['days']}g"
                cards([
                    {'label': 'Prossimo evento macro',
                     'value': f"{_nev['label']} {_when_txt}",
                     'delta': (f"⛔ blocca vendite ({_nev['date']})" if _ev_block
                               else f"libero ({_nev['date']})"),
                     'delta_color': 'red' if _ev_block else 'green'},
                ], per_row=1)

            st.markdown(_dec_sv.summary_html, unsafe_allow_html=True)

            if _dec_sv.action == 'TRADE' and _dec_sv.legs:
                st.markdown("<span style='font-size:12px;color:#3B82F6;'>"
                            "<b>Gambe che il sistema venderebbe (XSP) &mdash; "
                            "ipotetiche, rischio definito:</b></span>",
                            unsafe_allow_html=True)
                _legrows_sv = [{
                    'Lato': l['side'], 'Tipo': l['type'],
                    'Strike': f"{l['strike']:.0f}",
                } for l in _dec_sv.legs]
                st.dataframe(pd.DataFrame(_legrows_sv), use_container_width=True,
                             hide_index=True)
                cards([
                    {'label': 'Credito incassato',
                     'value': f"${_dec_sv.credit_usd:.0f}", 'delta_color': 'green'},
                    {'label': 'Perdita massima',
                     'value': f"${_dec_sv.max_loss_usd:.0f}", 'delta_color': 'red'},
                ], per_row=2)

            if _dec_sv.conditions:
                _cc_sv = st.columns(2)
                for _i, _c in enumerate(_dec_sv.conditions):
                    _icon = "✅" if _c['met'] else "❌"
                    _cc_sv[_i % 2].markdown(
                        f"<span style='font-size:12px;'>{_icon} <b>{_c['label']}</b><br>"
                        f"<span style='color:#6B7280;'>{_c['detail']}</span></span>",
                        unsafe_allow_html=True)

            if _dec_sv.explanation:
                _exp_bg = '#ECFDF5' if _dec_sv.action == 'TRADE' else '#F3F4F6'
                _exp_tx = '#065F46' if _dec_sv.action == 'TRADE' else '#374151'
                st.markdown(
                    f"<div style='background:{_exp_bg};border-radius:8px;padding:11px 14px;"
                    f"margin-top:10px;'><span style='color:{_exp_tx};font-size:12.5px;"
                    f"line-height:1.45;'>{_dec_sv.explanation}</span></div>",
                    unsafe_allow_html=True)
        except Exception as _sve:
            st.error(f"Errore anteprima short-vol: {_sve}")
    else:
        st.info("Carica la full chain per vedere l'anteprima del segnale short-vol.")

    # Positions + decision log
    st.markdown("---")
    st.markdown("**📋 Storico posizioni short-vol**")
    _svpos = sv.load_positions()
    if _svpos.empty:
        st.caption("Nessuna posizione short-vol ancora. Le aprirà il motore automatico "
                   "quando le condizioni saranno favorevoli.")
    else:
        _show_cols = ['id', 'open_date', 'status', 'strategy', 'credit',
                      'max_loss', 'entry_iv', 'close_reason', 'pnl_usd']
        _show = _svpos[[c for c in _show_cols if c in _svpos.columns]]
        st.dataframe(_show, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("**🔍 Analisi delle decisioni short-vol**")
    _sk_sv = sv.skip_log_summary()
    if _sk_sv['total_days'] == 0:
        st.caption("Ancora nessuna decisione registrata. Si popolerà con i run "
                   "automatici del motore.")
    else:
        cards([
            {'label': 'Giorni valutati', 'value': f"{_sk_sv['total_days']}"},
            {'label': 'Trade', 'value': f"{_sk_sv['n_trade']}",
             'delta': f"{_sk_sv['trade_rate']:.0f}% dei giorni"},
            {'label': 'SKIP', 'value': f"{_sk_sv['n_skip']}"},
        ], per_row=3)
        if _sk_sv['skip_breakdown']:
            st.caption("**Motivi dei SKIP:**")
            _bd_sv = pd.DataFrame(
                [{'Motivo': k, 'Giorni': v}
                 for k, v in sorted(_sk_sv['skip_breakdown'].items(),
                                    key=lambda x: -x[1])])
            st.dataframe(_bd_sv, use_container_width=True, hide_index=True)

    st.info("💡 Lo short-vol va validato in modo indipendente dal long-vol, su un "
            "periodo che includa almeno un episodio di stress: è lì che si vede se "
            "le strutture a rischio definito proteggono davvero. Non confrontare i "
            "P&L dei due sistemi su periodi troppo brevi.")


# ── Tab 10: Specchietto Premi (premi + vol per scadenza, confronto storico) ────
with tab10:
    import dex_gex_dashboard as dg
    st.markdown("### 💵 Premiums vs Vol")
    st.caption("Volatilità e premi (call/put) per strike, una scadenza alla volta. "
               "Le modalità differenziali confrontano con una data storica: servono a "
               "capire se stai per comprare caro o vendere sottoprezzato.")

    _prem_dates = dg.list_premium_dates()

    if not _prem_dates:
        st.info("📭 Nessuno snapshot premi ancora salvato. Lo storico si costruisce "
                "in avanti: il motore serale (o lo snapshot automatico) salva ogni "
                "giorno i premi/volatilità di SPX (strike da −25% a +15% dallo spot, "
                "scadenze entro ~6 mesi). Torna tra qualche giorno.")
    else:
        _day_today = _prem_dates[-1]
        _snap = dg.load_premium_snapshot(_day_today)
        if _snap.empty:
            st.warning("Snapshot di oggi vuoto.")
        else:
            _spot_now = float(_snap['spot'].iloc[0]) if 'spot' in _snap.columns else None
            st.markdown(f"**📅 Oggi:** `{_day_today}`"
                        + (f"  ·  **Spot:** `{_spot_now:.0f}`" if _spot_now else "")
                        + f"  ·  **Snapshot:** {len(_prem_dates)}")

            # ── Row 1: expiry + mode + side (the essentials) ──────────────
            _exps_all = sorted(_snap['expiry'].unique(),
                               key=lambda e: _snap[_snap['expiry']==e]['T_days'].iloc[0])
            _exp_lab = {e: f"{e}  ({int(_snap[_snap['expiry']==e]['T_days'].iloc[0])}g)"
                        for e in _exps_all}
            _c1, _c2, _c3 = st.columns([1.4, 1.4, 0.9])
            with _c1:
                _sel_exp = st.selectbox("Scadenza", options=_exps_all,
                                        format_func=lambda e: _exp_lab[e], key="pv_exp")
            with _c2:
                _mode_view = st.selectbox("Modalità",
                    ['Volatilità', 'Premi', 'Differenziale Vol', 'Differenza Premi'],
                    key="pv_mode")
            with _c3:
                _side_view = st.selectbox("Lato", ['Call', 'Put'], key="pv_side")

            # ── Reference date (only for differential modes) ──────────────
            _is_diff = _mode_view in ('Differenziale Vol', 'Differenza Premi')
            _cmp_snap = None
            if _is_diff:
                _ref_opts = [d for d in _prem_dates if d != _day_today][::-1]
                if not _ref_opts:
                    st.warning("Serve almeno un secondo giorno salvato per la modalità "
                               "differenziale.")
                else:
                    _day_ref = st.selectbox("📅 Confronta con:", options=_ref_opts,
                                            key="pv_refdate")
                    _cmp_snap = dg.load_premium_snapshot(_day_ref)

            # ── Row 2: compact strike filters (always visible) ────────────
            _all_strikes = sorted(_snap['strike'].unique())
            _k_min, _k_max = int(_all_strikes[0]), int(_all_strikes[-1])
            _step_guess = int(_all_strikes[1] - _all_strikes[0]) if len(_all_strikes) > 1 else 50
            _f1, _f2, _f3 = st.columns(3)
            with _f1:
                _k_start = st.number_input("Strike da", value=_k_min,
                                           step=_step_guess, key="pv_kstart")
            with _f2:
                _k_end = st.number_input("Strike a", value=_k_max,
                                         step=_step_guess, key="pv_kend")
            with _f3:
                _k_int = st.number_input("Intervallo", value=_step_guess,
                                         min_value=_step_guess, step=_step_guess,
                                         key="pv_kint")

            _side_l = 'call' if _side_view == 'Call' else 'put'
            _chart_mode = {'Volatilità': f'vol_{_side_l}', 'Premi': f'prem_{_side_l}',
                           'Differenziale Vol': f'diff_vol_{_side_l}',
                           'Differenza Premi': f'diff_prem_{_side_l}'}[_mode_view]

            def _filter_strikes(df):
                if df is None or df.empty:
                    return df
                d = df[(df['strike'] >= _k_start) & (df['strike'] <= _k_end)].copy()
                if _k_int > _step_guess:
                    d = d[((d['strike'] - _k_start) % _k_int == 0)]
                return d

            _snap_f = _filter_strikes(_snap)
            _cmp_f = _filter_strikes(_cmp_snap) if _cmp_snap is not None else None

            # ── One big chart for the selected expiry ─────────────────────
            if _is_diff and (_cmp_f is None or _cmp_f.empty):
                st.info("Seleziona una data di confronto valida per i differenziali.")
            else:
                _msub = _snap_f[_snap_f['expiry']==_sel_exp]
                _tdays = int(_msub['T_days'].iloc[0]) if not _msub.empty else 0
                st.markdown(f"#### {_sel_exp} · {_tdays}g · {_mode_view} {_side_view}"
                            + (f"  ·  spot {_spot_now:.0f}" if _spot_now else ""))
                try:
                    _fig = dg.premium_bar_chart(_snap_f, _sel_exp, _chart_mode,
                                                compare_df=_cmp_f, spot=_spot_now)
                    _fig.update_layout(height=460)
                    st.plotly_chart(_fig, use_container_width=True, key="pv_main")
                except Exception as _e:
                    st.caption(f"Errore grafico: {_e}")

                if _is_diff:
                    st.caption("💡 Le barre mostrano quanto vol/premi sono cambiati "
                               f"rispetto al {_day_ref}. Se ciò che vorresti aprire è "
                               "già molto mosso a sfavore, forse sei in ritardo.")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"<div style='text-align:center;font-size:10px;color:#9CA3AF;padding:8px 0;'>"
    f"DEX/GEX Analytics v{APP_VERSION} · build {APP_BUILD} · "
    f"Dati: Barchart (chain) + Yahoo Finance (storico/intraday) · "
    f"Modello: Black-Scholes-Merton con q implicita</div>",
    unsafe_allow_html=True,
)
