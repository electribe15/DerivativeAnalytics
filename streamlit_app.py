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
        fetch_price_history,
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

        # Fetch last 5 sessions of OHLC — lightweight, non-fatal if unavailable
        with st.spinner("⏳ Storico prezzi…"):
            ohlc = fetch_price_history(ticker, n_days=5)

        st.session_state["data"] = dict(raw=raw_df, by_strike=by_strike,
                                         by_expiry=by_expiry, spot=spot,
                                         ticker=ticker, raw_full=raw_full,
                                         ohlc=ohlc)

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
raw_full  = d.get("raw_full", raw_df)   # unfiltered chain for vol analytics
ohlc      = d.get("ohlc")

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

# ── 0DTE metrics (computed once, used in the first tab) ───────────────────────
dte0_metrics = compute_0dte_metrics(raw_full, spot)
has_0dte     = bool(dte0_metrics)

# ── Tabs — 0DTE is always first and shown by default ─────────────────────────
tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "⚡ 0DTE",
    "📊 GEX / DEX",
    "📐 Range & Skew",
    "💰 Put Monitor",
    "📈 Volatilità",
    "🔥 Open Interest",
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
        # ── Metric cards ──
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
        mc[4].metric("Total 0DTE GEX",
                     f"{'+'if m['total_gex']>=0 else ''}{m['total_gex']/1e9:.2f}B")
        mc[5].metric("0DTE Contracts", f"{m['n_contracts']:,}")

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

        # ── Key strikes table ──
        with st.expander("📋 Strikes chiave 0DTE"):
            bs0 = m['by_strike']
            if not bs0.empty:
                tbl = bs0.copy()
                tbl = tbl.sort_values('net_gex', ascending=False)
                disp_cols = [c for c in
                             ['strike','net_gex','net_dex','call_gex','put_gex',
                              'call_dex','put_dex','total_oi']
                             if c in tbl.columns]
                st.dataframe(
                    tbl[disp_cols].reset_index(drop=True),
                    use_container_width=True,
                    hide_index=True,
                )

# ── Tab 1: GEX / DEX ─────────────────────────────────────────────────────────
with tab1:
    # Price vs DEX Levels — the centrepiece: shared Y-axis chart
    try:
        st.plotly_chart(
            price_vs_dex_chart(by_strike, ohlc, spot, cur_tick,
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
