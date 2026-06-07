"""
DEX / GEX Options Exposure Dashboard — Streamlit version
"""

import warnings
warnings.filterwarnings("ignore")

import os
import time

import streamlit as st
import plotly.graph_objects as go

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
    color: #111827 !important; font-size: 22px !important; font-weight: 700;
    font-family: 'Inter', sans-serif !important;
}

/* ── Tabs ────────────────────────────────────────────── */
div[data-baseweb="tab-list"] {
    background: #FFFFFF; border-radius: 10px; padding: 4px;
    border-bottom: none; gap: 2px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
button[data-baseweb="tab"] {
    color: #6B7280 !important; font-family: 'Inter', sans-serif !important;
    font-size: 13px !important; font-weight: 500; border-radius: 7px; padding: 6px 14px;
}
button[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #6C63FF 0%, #8B5CF6 100%) !important;
    color: #FFFFFF !important; border: none !important;
    box-shadow: 0 2px 8px rgba(108,99,255,0.4);
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
        compute_0dte_gamma_schedule,
        fetch_price_history,
        fetch_intraday_history,
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
                                         vix_hist=vix_hist)

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

# ── Refresh Intraday (lightweight — no chain re-download) ─────────────────────
with st.sidebar:
    st.markdown("---")
    refresh_intra_btn = st.button("🔄 Aggiorna Intraday", use_container_width=True)
    if intraday is not None and not (intraday.empty if hasattr(intraday,'empty') else True):
        last_bar = intraday['datetime'].iloc[-1]
        ts_str   = last_bar.strftime('%H:%M') if hasattr(last_bar, 'strftime') else str(last_bar)
        st.markdown(f"<p style='font-size:10px;color:#818CF8;text-align:center;"
                    f"margin-top:-8px;'>intraday last bar: {ts_str}</p>",
                    unsafe_allow_html=True)

if refresh_intra_btn and "data" in st.session_state:
    with st.spinner("⏳ Aggiornamento intraday…"):
        new_intra = fetch_intraday_history(cur_tick, interval_min=5)
    st.session_state["data"]["intraday"] = new_intra
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
from datetime import datetime
ts = datetime.now().strftime("%H:%M:%S")
st.markdown(
    f"<div class='status-bar'>Last updated: {ts} &nbsp;|&nbsp; "
    f"{len(raw_df):,} contracts &nbsp;|&nbsp; {raw_df['expiry'].nunique()} expiries &nbsp;|&nbsp; "
    f"Spot <b style='color:#ffd166'>${spot:.2f}</b> &nbsp;|&nbsp; "
    f"Δ filter: [{delta_range[0]:+.2f}, {delta_range[1]:+.2f}]</div>",
    unsafe_allow_html=True,
)

# ── Metric cards ──────────────────────────────────────────────────────────────
stats = build_stat_metrics(raw_df, by_strike, spot)
metric_cols = st.columns(len(stats))
for col, (label, (value, _color)) in zip(metric_cols, stats.items()):
    col.metric(label=label, value=value)

st.markdown("---")

st.markdown("---")

st.markdown("---")

# ── Alert Flags Panel ─────────────────────────────────────────────────────────
if "data" in st.session_state and st.session_state["data"].get("raw") is not None:
    _d      = st.session_state["data"]
    _raw_f  = _d.get("raw_full")
    if _raw_f is None:
        _raw_f = _d.get("raw")
    _spot_a = _d.get("spot", 0)
    _dte0_m = compute_0dte_metrics(_raw_f, _spot_a) if _raw_f is not None else {}
    _rvdf   = compute_rvol_all(_d.get("ohlc_spx", pd.DataFrame()), window=126) \
              if _d.get("ohlc_spx") is not None else pd.DataFrame()
    _flags  = build_alert_flags(_raw_f, _spot_a, _dte0_m, _rvdf, _d.get("vix_hist"))

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
tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "⚡ 0DTE",
    "📊 GEX / DEX",
    "📐 Range & Skew",
    "💰 Put Monitor",
    "📈 Volatilità",
    "🔥 Open Interest",
    "📉 Realized Vol",
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
        mc = st.columns(6)
        mc[0].metric("0DTE ATM IV",
                     f"{m['atm_iv']*100:.1f}%" if m['atm_iv'] else "—")
        mc[1].metric("Expected Move",
                     f"±{m['exp_move_pts']:.1f} pts  ({m['exp_move_pct']:.2f}%)"
                     if m['exp_move_pts'] else "—")
        mc[2].metric("GEX Flip",
                     f"${m['gex_flip']:.0f}" if m['gex_flip'] else "—")
        mc[3].metric("Max Gamma Strike",
                     f"${m['max_gex_strike']:.0f}" if m['max_gex_strike'] else "—")
        mc[4].metric("Max Pain",
                     f"${m['max_pain']:.0f}" if m.get('max_pain') else "—",
                     help="Strike che minimizza il payout totale agli option buyer — livello di pinning atteso a scadenza")
        mc[5].metric("Total 0DTE GEX",
                     f"{'+'if m['total_gex']>=0 else ''}{m['total_gex']/1e9:.2f}B")

        # ── Metric cards — row 2 ──────────────────────────────────────────
        mc2 = st.columns(5)
        pc = m.get('pc_ratio')
        pc_delta = f"{pc:.2f}" if pc else None
        mc2[0].metric("P/C OI Ratio",
                      f"{pc:.2f}" if pc else "—",
                      delta=("↑ Lean PUT" if pc and pc > 1.2 else ("↓ Lean CALL" if pc and pc < 0.8 else "Neutro")) if pc else None,
                      help="Put OI / Call OI. >1 = lean ribassista; <1 = lean rialzista")

        charm_daily = (m['charm_exp'] / 252 / 1e6) if m.get('charm_exp') else None
        mc2[1].metric("Charm Exp (daily Δ)",
                      f"${charm_daily:+.1f}M/day" if charm_daily else "—",
                      help="Delta drift giornaliero dal solo passare del tempo (charm × OI × 100 × Spot / 252). Il dealer deve hedgiare questo delta ogni giorno anche senza movimenti di prezzo.")

        vanna_m = (m['vanna_exp'] / 1e6) if m.get('vanna_exp') else None
        mc2[2].metric("Vanna Exp (per 1pt vol)",
                      f"${vanna_m:+.1f}M" if vanna_m else "—",
                      help="Variazione del delta-hedge quando la vol implicita si muove di 1 punto. Negativo = dealer deve comprare when vol spike (amplifica sell-off).")

        gross = (m['gross_gex'] / 1e9) if m.get('gross_gex') else None
        mc2[3].metric("Gross GEX 0DTE",
                      f"${gross:.2f}B" if gross else "—",
                      help="GEX lordo (|call_gex| + |put_gex|) senza netting. Misura l'intensità totale del gamma hedging indipendentemente dal segno.")
        mc2[4].metric("0DTE Contracts",
                      f"{m['n_contracts']:,}")

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
        am = st.columns(5)
        am[0].metric(
            "GEX Center of Mass",
            f"${ga['center_of_mass']:,.0f}",
            help="Strike baricentro del GEX lordo — il livello di maggior attrazione gravitazionale per il prezzo (più informativo del flip)")
        am[1].metric(
            "HHI Concentrazione",
            f"{ga['hhi']:.4f}",
            delta="Alta" if ga['hhi'] > 0.05 else "Bassa",
            help="Herfindahl-Hirschman Index del GEX per strike. >0.05 = GEX concentrato (pinning forte). <0.01 = distribuito (pinning debole)")
        am[2].metric(
            "Impact 1% Move",
            f"${ga['impact_1pct']/1e6:.0f}M Δ",
            help="$ di delta da hedgiare se il mercato si muove dell'1%. Misura la pressione di hedging concreta dei dealer")
        am[3].metric(
            "Impact 5% Move",
            f"${ga['impact_5pct']/1e6:.0f}M Δ",
            help="$ di delta da hedgiare per un movimento del 5%")
        am[4].metric(
            "Flip Zone",
            f"[{ga['flip_zone_lo']:,.0f} – {ga['flip_zone_hi']:,.0f}]",
            help="Zona di ambiguità del regime: strikes dove il GEX è < 10% del picco. All'interno di questa banda il regime è instabile")

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
                    "Identifica dove il regime cambia con movimenti di mercato.")
                st.dataframe(profile_disp.reset_index(drop=True),
                             use_container_width=True, hide_index=True)
        st.markdown("---")
    # Price vs DEX Levels — the centrepiece: shared Y-axis chart
    try:
        st.plotly_chart(
            price_vs_dex_chart(by_strike, intraday, spot, cur_tick,
                               strike_lo=strike_lo, strike_hi=strike_hi),
            use_container_width=True,
        )
    except Exception as e:
        st.plotly_chart(empty_fig(str(e)), use_container_width=True)

    st.markdown("---")
    col_l, col_r = st.columns(2)
    with col_l:
        try:
            st.plotly_chart(gex_bar_chart(by_strike, spot, cur_tick,
                                          strike_lo=strike_lo, strike_hi=strike_hi),
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
                st.dataframe(
                    disp.style
                        .format({"Mid (pts)":"{:.2f}", "IV %":"{:.1f}",
                                 "Mid % Spot":"{:.3f}%", "Δ Eff.":"{:.3f}",
                                 "Δ Target":"{:.2f}"})
                        .background_gradient(subset=["Mid % Spot"], cmap="YlOrRd"),
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
        st.dataframe(
            by_strike[available].sort_values("net_gex", ascending=False).reset_index(drop=True),
            use_container_width=True,
        )

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
        mc = st.columns(6)
        last_yz  = float(rvol_df['Yang-Zhang'].iloc[-1]) if not rvol_df.empty else None
        last_vix = float(vix_hist['vix'].iloc[-1] * 100) if has_vix else None
        vol_premium = (last_vix - last_yz) if (last_vix and last_yz) else None

        mc[0].metric("Intraday RVol",
                     f"{intra_rv*100:.1f}%" if intra_rv else "—",
                     help="RVol annualizzata sessione corrente (barre 5-min)")
        mc[1].metric("126d RVol  (Yang-Zhang)",
                     f"{last_yz:.1f}%" if last_yz else "—",
                     help="Finestra 6 mesi — modello minima varianza")
        mc[2].metric("VIX",
                     f"{last_vix:.1f}%" if last_vix else "—")
        mc[3].metric("Vol Premium  (VIX−RVol)",
                     (f"+{vol_premium:.1f}%" if vol_premium > 0 else f"{vol_premium:.1f}%")
                     if vol_premium is not None else "—",
                     delta=f"{vol_premium:.1f}%" if vol_premium else None)
        mc[4].metric("HAR-RV Forecast (1d)",
                     f"{har_forecast:.1f}%" if har_forecast else "—",
                     help=(f"HAR-RV (Corsi 2009) — previsione RVol 1g. "
                           + (f"68% CI [{har_result.get('ci_68_lo',0):.1f}–{har_result.get('ci_68_hi',0):.1f}%]  "
                              f"R²={har_r2:.2f}" if har_r2 else "")),
                     delta=(f"vs RVol {(har_forecast-last_yz):+.1f}%" if har_forecast and last_yz else None))
        mc[5].metric(f"{regime_icon} Vol Regime",
                     vol_regime,
                     delta=f"z = {vol_z:+.2f}",
                     help="Regime via z-score rolling 1Y della RVol Yang-Zhang. "
                          "|z|>1.5 = HIGH/LOW; else NORMAL.")

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
                    bmc = st.columns(4)
                    bmc[0].metric("RMSE (oos)", f"{har_bt['rmse']:.2f}%")
                    bmc[1].metric("MAE (oos)",  f"{har_bt['mae']:.2f}%")
                    bmc[2].metric("Dir. Acc.",  f"{har_bt['dir_acc']*100:.1f}%")
                    bmc[3].metric("R² oos",     f"{har_bt['r2_oos']:.3f}")
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
