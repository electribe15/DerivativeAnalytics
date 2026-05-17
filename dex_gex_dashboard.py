#!/usr/bin/env python3
"""
Converted script from the notebook `.devcontainer/dex_gex_dashboard.ipynb`.
This script preserves the notebook's imports, helpers, data fetching, chart builders
and the Dash app. Run it from the repository root.

Usage:
  python .devcontainer/dex_gex_dashboard.py

Note: running this will start the Dash app and bind to `PORT` as defined in the config.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
import yfinance as yf
import socket
from datetime import datetime, date
# Defer Plotly imports to inside plotting functions so this module can be
# imported without requiring Plotly to be installed. Plotly is heavy and
# only needed when rendering charts (Streamlit/ Dash). See individual
# chart functions for local imports.

# ── Dashboard config ──────────────────────────────────────────────────────────
DEFAULT_TICKER   = 'SPY'   # Change to any ticker (SPY, QQQ, AAPL, TSLA …)
RISK_FREE_RATE   = 0.053221  # Update to current Fed funds rate
OI_THRESHOLD     = 100     # Minimum open interest to keep a strike
MAX_EXPIRY_DAYS  = 90      # Only look at expiries within this window
PORT             = 8051    # Browser port (safe default)

print(f'Config loaded. Default ticker: {DEFAULT_TICKER}')

# ── Black-Scholes helpers ─────────────────────────────────────────────

def bs_delta(S, K, T, r, sigma, flag):
    """Black-Scholes delta. flag='c' for call, 'p' for put."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if flag == 'c':
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1


def bs_gamma(S, K, T, r, sigma):
    """Black-Scholes gamma (same for calls & puts)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def implied_vol(price, S, K, T, r, flag, tol=1e-6):
    """Implied volatility via Brent's method. Returns NaN on failure."""
    if T <= 0 or price <= 0:
        return np.nan
    intrinsic = max(0, S - K) if flag == 'c' else max(0, K - S)
    if price < intrinsic:
        return np.nan
    try:
        def objective(sigma):
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            if flag == 'c':
                return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2) - price
            else:
                return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1) - price
        return brentq(objective, 1e-6, 10.0, xtol=tol)
    except Exception:
        return np.nan

print('Black-Scholes functions ready.')

# ── Data fetching & processing ────────────────────────────────────────

def fetch_options_data(ticker: str, max_days: int = MAX_EXPIRY_DAYS,
                        oi_thresh: int = OI_THRESHOLD, r: float = RISK_FREE_RATE):
    """
    Fetches options chain from Yahoo Finance and computes:
    - Implied Volatility
    - Delta per contract
    - Gamma per contract
    - DEX = delta * OI * 100
    - GEX = gamma * OI * 100 * spot  (dealer convention: calls +, puts -)
    """
    print(f'Fetching data for {ticker} …')
    tk   = yf.Ticker(ticker)
    info = tk.fast_info
    S    = info.last_price
    print(f'  Spot price: ${S:.2f}')

    today      = date.today()
    expiries   = tk.options           # tuple of expiry strings
    rows       = []

    for exp_str in expiries:
        exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
        T_days   = (exp_date - today).days
        if T_days < 1 or T_days > max_days:
            continue
        T_years = T_days / 365.0

        chain = tk.option_chain(exp_str)

        for flag, df in [('c', chain.calls), ('p', chain.puts)]:
            df = df.copy()
            df['flag']       = flag
            df['expiry']     = exp_str
            df['T_days']     = T_days
            df['T_years']    = T_years
            df['spot']       = S
            df['mid']        = (df['bid'] + df['ask']) / 2
            rows.append(df)

    if not rows:
        raise ValueError('No options data found within the expiry window.')

    data = pd.concat(rows, ignore_index=True)

    # Filter low OI
    data = data[data['openInterest'] >= oi_thresh].copy()

    # Compute IV
    data['iv'] = data.apply(
        lambda row: implied_vol(row['mid'], S, row['strike'], row['T_years'], r, row['flag']),
        axis=1
    )
    data.dropna(subset=['iv'], inplace=True)
    data = data[data['iv'] > 0].copy()

    # Greeks
    data['delta'] = data.apply(
        lambda row: bs_delta(S, row['strike'], row['T_years'], r, row['iv'], row['flag']),
        axis=1,
    )
    data['gamma'] = data.apply(
        lambda row: bs_gamma(S, row['strike'], row['T_years'], r, row['iv']),
        axis=1,
    )

    # DEX = net delta dollars per 1-point move  (delta * OI * 100)
    data['dex'] = data['delta'] * data['openInterest'] * 100

    # GEX = gamma * OI * 100 * spot  (dealer flip: puts negative)
    sign = data['flag'].map({'c': 1, 'p': -1})
    data['gex'] = sign * data['gamma'] * data['openInterest'] * 100 * S

    print(f'  Loaded {len(data):,} option contracts across {data["expiry"].nunique()} expiries.')
    return data, S


def aggregate_by_strike(data: pd.DataFrame):
    """Roll up DEX and GEX to per-strike totals."""
    agg = data.groupby('strike').agg(
        net_dex = ('dex', 'sum'),
        net_gex = ('gex', 'sum'),
        call_dex = ('dex', lambda x: x[data.loc[x.index, 'flag'] == 'c'].sum()),
        put_dex  = ('dex', lambda x: x[data.loc[x.index, 'flag'] == 'p'].sum()),
        call_gex = ('gex', lambda x: x[data.loc[x.index, 'flag'] == 'c'].sum()),
        put_gex  = ('gex', lambda x: x[data.loc[x.index, 'flag'] == 'p'].sum()),
        total_oi = ('openInterest', 'sum'),
    ).reset_index()
    return agg


def aggregate_by_expiry(data: pd.DataFrame):
    """Roll up DEX and GEX to per-expiry totals."""
    return data.groupby(['expiry', 'T_days']).agg(
        net_dex = ('dex', 'sum'),
        net_gex = ('gex', 'sum'),
        total_oi = ('openInterest', 'sum'),
    ).reset_index().sort_values('T_days')

print('Data functions ready.')

# ── Load initial data (lazy: do not fetch until run) ─────────────────────────────────

# Note: fetching at import time can be slow and requires network; keep commented by default.
# raw_data, spot_price = fetch_options_data(DEFAULT_TICKER)
# by_strike  = aggregate_by_strike(raw_data)
# by_expiry  = aggregate_by_expiry(raw_data)

# ── Chart builders ────────────────────────────────────────────────────

DARK_BG    = '#0d0f14'
CARD_BG    = '#13161e'
ACCENT_GRN = '#00e5a0'
ACCENT_RED = '#ff4d6d'
ACCENT_BLU = '#4db8ff'
ACCENT_YLW = '#ffd166'
GRID_COL   = '#1e2130'
TEXT_COL   = '#c9d1e0'


def base_layout(title='', height=420):
    return dict(
        title=dict(text=title, font=dict(color=TEXT_COL, size=14, family='Courier New')),
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color=TEXT_COL, family='Courier New'),
        height=height,
        xaxis=dict(gridcolor=GRID_COL, zerolinecolor=GRID_COL),
        yaxis=dict(gridcolor=GRID_COL, zerolinecolor=GRID_COL),
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=11))
    )


def gex_bar_chart(by_strike_df, spot, ticker, window_pct=0.10):
    import plotly.graph_objects as go

    lo = spot * (1 - window_pct)
    hi = spot * (1 + window_pct)
    df = by_strike_df[(by_strike_df['strike'] >= lo) & (by_strike_df['strike'] <= hi)].copy()

    colors = [ACCENT_GRN if v >= 0 else ACCENT_RED for v in df['net_gex']]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df['strike'], y=df['net_gex'] / 1e6,
        marker_color=colors,
        name='Net GEX'
    ))
    fig.add_vline(x=spot, line_color=ACCENT_YLW, line_dash='dash',
                  annotation_text=f' Spot ${spot:.1f}',
                  annotation_font_color=ACCENT_YLW)
    fig.update_layout(**base_layout(f'GEX by Strike — {ticker}'),
                       yaxis_title='GEX ($M)',
                       xaxis_title='Strike')
    return fig


def dex_bar_chart(by_strike_df, spot, ticker, window_pct=0.10):
    import plotly.graph_objects as go

    lo = spot * (1 - window_pct)
    hi = spot * (1 + window_pct)
    df = by_strike_df[(by_strike_df['strike'] >= lo) & (by_strike_df['strike'] <= hi)].copy()

    fig = go.Figure()
    fig.add_trace(go.Bar(x=df['strike'], y=df['call_dex'] / 1e6,
                          name='Call DEX', marker_color=ACCENT_GRN))
    fig.add_trace(go.Bar(x=df['strike'], y=df['put_dex'] / 1e6,
                          name='Put DEX', marker_color=ACCENT_RED))
    fig.add_trace(go.Scatter(x=df['strike'], y=df['net_dex'] / 1e6,
                              mode='lines+markers', name='Net DEX',
                              line=dict(color=ACCENT_YLW, width=2)))
    fig.add_vline(x=spot, line_color=ACCENT_BLU, line_dash='dash',
                  annotation_text=f' Spot ${spot:.1f}',
                  annotation_font_color=ACCENT_BLU)
    fig.update_layout(**base_layout(f'DEX by Strike — {ticker}'),
                       barmode='relative',
                       yaxis_title='DEX ($M)',
                       xaxis_title='Strike')
    return fig


def gex_expiry_chart(by_expiry_df, ticker):
    import plotly.graph_objects as go

    fig = go.Figure()
    colors = [ACCENT_GRN if v >= 0 else ACCENT_RED for v in by_expiry_df['net_gex']]
    fig.add_trace(go.Bar(
        x=by_expiry_df['expiry'], y=by_expiry_df['net_gex'] / 1e6,
        marker_color=colors, name='Net GEX'
    ))
    fig.update_layout(**base_layout(f'GEX by Expiry — {ticker}'),
                       yaxis_title='GEX ($M)', xaxis_title='Expiry')
    return fig


def oi_heatmap(raw_df, spot, ticker, window_pct=0.12):
    import plotly.graph_objects as go

    lo = spot * (1 - window_pct)
    hi = spot * (1 + window_pct)
    df = raw_df[(raw_df['strike'] >= lo) & (raw_df['strike'] <= hi)].copy()

    pivot = df.pivot_table(index='expiry', columns='strike',
                            values='openInterest', aggfunc='sum', fill_value=0)
    fig = go.Figure(go.Heatmap(
        z=pivot.values / 1000,
        x=[str(c) for c in pivot.columns],
        y=pivot.index.tolist(),
        colorscale='Viridis',
        colorbar=dict(title='OI (K)', tickfont=dict(color=TEXT_COL)),
        hoverongaps=False
    ))
    fig.update_layout(**base_layout(f'Open Interest Heatmap — {ticker}', height=380),
                       xaxis_title='Strike', yaxis_title='Expiry')
    return fig


def vol_smile_chart(raw_df, spot, ticker):
    """IV smile or surface for calls.

    - If there are 4 or fewer expiries, show individual IV smiles (lines).
    - If there are more expiries, interpolate each expiry's call IVs onto a
      common strike grid and render a continuous heatmap (expiry vs strike).
    """
    expiries = sorted(raw_df['expiry'].unique())
    n_exp = len(expiries)

    # Local import so module import doesn't require Plotly
    import plotly.graph_objects as go

    # Keep the compact multi-line view for a small number of expiries
    if n_exp <= 4:
        expiries = expiries[:4]
        colors = [ACCENT_GRN, ACCENT_BLU, ACCENT_YLW, ACCENT_RED]
        fig = go.Figure()
        for exp, col in zip(expiries, colors):
            sub = raw_df[(raw_df['expiry'] == exp) & (raw_df['flag'] == 'c')].sort_values('strike')
            fig.add_trace(go.Scatter(
                x=sub['strike'], y=sub['iv'] * 100,
                mode='lines', name=exp, line=dict(color=col, width=2)
            ))
        fig.add_vline(x=spot, line_color=ACCENT_YLW, line_dash='dot')
        fig.update_layout(**base_layout(f'IV Smile (Calls) — {ticker}'),
                           yaxis_title='Implied Vol (%)', xaxis_title='Strike')
        return fig

    # For many expiries, build a continuous IV surface by interpolating onto
    # a common strike grid so that the surface/heatmap is smooth and continuous.
    calls = raw_df[raw_df['flag'] == 'c']
    if calls.empty:
        # fallback to empty figure
        fig = go.Figure()
        fig.update_layout(**base_layout(f'IV Surface (Calls) — {ticker}'))
        return fig

    # Determine strike grid (dense enough for a smooth surface)
    min_strike = calls['strike'].min()
    max_strike = calls['strike'].max()
    if pd.isna(min_strike) or pd.isna(max_strike) or min_strike >= max_strike:
        fig = go.Figure()
        fig.update_layout(**base_layout(f'IV Surface (Calls) — {ticker}'))
        return fig

    strike_grid = np.linspace(min_strike, max_strike, 140)

    # Order expiries by time to expiry (T_days) for a natural y-axis
    exp_meta = raw_df[['expiry', 'T_days']].drop_duplicates().set_index('expiry')
    expiries_sorted = exp_meta.loc[expiries].sort_values('T_days').index.tolist()

    z_rows = []
    y_labels = []
    for exp in expiries_sorted:
        sub = calls[calls['expiry'] == exp].sort_values('strike')
        strikes = sub['strike'].values
        ivs = sub['iv'].values
        if len(strikes) < 2:
            row = np.full(strike_grid.shape, np.nan)
        else:
            # Interpolate; values outside known strikes set to nan for clarity
            row = np.interp(strike_grid, strikes, ivs, left=np.nan, right=np.nan)
        z_rows.append(row * 100)  # convert to percent
        y_labels.append(exp)

    z = np.vstack(z_rows)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=np.round(strike_grid, 2),
        y=y_labels,
        colorscale='Viridis',
        colorbar=dict(title='Implied Vol (%)', tickfont=dict(color=TEXT_COL)),
        hovertemplate='Expiry: %{y}<br>Strike: %{x}<br>IV: %{z:.2f}%',
        zmin=np.nanpercentile(z, 2) if np.isfinite(z).any() else None,
        zmax=np.nanpercentile(z, 98) if np.isfinite(z).any() else None,
        hoverongaps=False,
    ))

    fig.update_layout(**base_layout(f'IV Surface (Calls) — {ticker}', height=420),
                       xaxis_title='Strike', yaxis_title='Expiry')
    fig.add_vline(x=spot, line_color=ACCENT_YLW, line_dash='dot')
    return fig


# --- Additional chart builders (from notebook)
def iv_surface_chart(raw_df, spot, ticker, expiries_to_show=6, expiries_list=None):
    """
    IV Surface 3D con:
    - Contour lines sovrapposte alla superficie
    - ATM line verticale (strike = spot)
    - Colorscale RdYlGn (verde=bassa IV, rosso=alta IV)
    - Filtri dati sporchi + interpolazione su griglia densa per superficie continua
    """
    from scipy.interpolate import griddata
    import numpy as np
    import plotly.graph_objects as go

    # ── Selezione scadenze ────────────────────────────────────────────
    if expiries_list is not None and len(expiries_list) > 0:
        exps = list(expiries_list)
    else:
        exps = sorted(raw_df['expiry'].unique())[:expiries_to_show]

    sub = raw_df[raw_df['expiry'].isin(exps) & (raw_df['flag'] == 'c')].copy()

    # ── Filtri dati sporchi ───────────────────────────────────────────
    sub = sub[sub['iv'] > 0.01]          # rimuovi IV quasi zero
    sub = sub[sub['iv'] < 5.0]           # rimuovi outlier > 500%
    sub = sub[sub['openInterest'] > 50]  # solo contratti liquidi
    sub = sub[sub['bid'] > 0]            # solo con bid valido

    if sub.empty:
        return go.Figure().update_layout(**base_layout(f'IV Surface — {ticker} (no data)'))

    # ── Converti expiry in numero (T_days) per asse Y numerico ────────
    exp_to_tdays = sub[['expiry', 'T_days']].drop_duplicates().set_index('expiry')['T_days'].to_dict()
    sub['t_num'] = sub['expiry'].map(exp_to_tdays)

    # ── Griglia densa per interpolazione ─────────────────────────────
    strike_min = sub['strike'].min()
    strike_max = sub['strike'].max()
    t_min      = sub['t_num'].min()
    t_max      = sub['t_num'].max()

    # Use a coarser grid to improve stability when data are sparse
    n_strike = 40
    n_expiry = 20
    grid_strikes  = np.linspace(strike_min, strike_max, n_strike)
    grid_tdays    = np.linspace(t_min, t_max, n_expiry)
    grid_x, grid_y = np.meshgrid(grid_strikes, grid_tdays)

    # Interpolate onto the grid. Prefer linear interpolation for stability,
    # fall back to nearest where linear yields NaNs (sparse regions).
    points = sub[['strike', 't_num']].values
    values = sub['iv'].values * 100  # percentuale

    # Try linear first (stable), then nearest to fill remaining gaps
    z_lin = griddata(points, values, (grid_x, grid_y), method='linear')
    if np.isfinite(z_lin).any():
        z_grid = z_lin
    else:
        # If linear produced no finite values (extremely sparse), use nearest
        z_grid = griddata(points, values, (grid_x, grid_y), method='nearest')

    # Fill any remaining NaNs with nearest-neighbor
    z_nn = griddata(points, values, (grid_x, grid_y), method='nearest')
    z_grid = np.where(np.isfinite(z_grid), z_grid, z_nn)

    # If interpolation failed to produce any finite values, return a placeholder
    finite_pct = np.count_nonzero(np.isfinite(z_grid)) / z_grid.size
    if not np.isfinite(z_grid).any() or finite_pct < 0.02:
        # If less than 2% of grid has data, consider it insufficient
        fig = go.Figure()
        fig.update_layout(**base_layout(f'IV Surface — {ticker} (insufficient data)'))
        return fig

    # Clip valori estremi (2°-98° percentile)
    z_lo = np.nanpercentile(z_grid, 2)
    z_hi = np.nanpercentile(z_grid, 98)
    z_grid = np.clip(z_grid, z_lo, z_hi)

    # Etichette Y (expiry string) per ogni riga della griglia
    tdays_sorted   = sorted(exp_to_tdays.values())
    expiry_by_tday = {v: k for k, v in exp_to_tdays.items()}
    # Mappa ogni valore numerico della griglia Y all'expiry string più vicina
    y_labels = []
    for td in grid_tdays:
        closest = min(tdays_sorted, key=lambda x: abs(x - td))
        y_labels.append(expiry_by_tday[closest])

    # ── Figura ────────────────────────────────────────────────────────
    fig = go.Figure()

    # 1) Superficie principale con colorscale RdYlGn
    fig.add_trace(go.Surface(
        z=z_grid,
        x=grid_strikes,
        y=y_labels,
        colorscale='RdYlGn_r',   # verde=bassa IV, rosso=alta IV
        reversescale=False,
        cmin=z_lo,
        cmax=z_hi,
        colorbar=dict(
            title='IV (%)',
            tickfont=dict(color=TEXT_COL, family='Courier New'),
            titlefont=dict(color=TEXT_COL, family='Courier New'),
        ),
        opacity=0.92,
        # 2) Contour lines sovrapposte
        contours=dict(
            z=dict(
                show=True,
                usecolorscale=True,
                project=dict(z=True),
                highlightcolor=ACCENT_YLW,
                width=1.5,
            )
        ),
        hovertemplate=(
            'Strike: %{x:.1f}<br>'
            'Expiry: %{y}<br>'
            'IV: %{z:.2f}%<extra></extra>'
        ),
        name='IV Surface',
    ))

    # 3) ATM Line — linea verticale allo strike più vicino a spot
    atm_col = int(np.argmin(np.abs(grid_strikes - spot)))
    z_atm = z_grid[:, atm_col]
    # Replace any NaNs in the ATM column with the row-wise mean (best-effort)
    if not np.isfinite(z_atm).all():
        row_means = np.nanmean(z_grid, axis=1)
        z_atm = np.where(np.isfinite(z_atm), z_atm, row_means)

    fig.add_trace(go.Scatter3d(
        x=[atm_strike] * n_expiry,
        y=y_labels,
        z=z_atm,
        mode='lines',
        line=dict(color=ACCENT_YLW, width=6),
        name=f'ATM  ${spot:.1f}',
        hovertemplate='ATM Strike: %{x}<br>Expiry: %{y}<br>IV: %{z:.2f}%<extra></extra>',
    ))

    # ── Layout ────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=f'IV Surface — {ticker}',
            font=dict(color=TEXT_COL, size=14, family='Courier New'),
        ),
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color=TEXT_COL, family='Courier New'),
        height=540,
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            font=dict(size=11, color=TEXT_COL),
            x=0.01, y=0.99,
        ),
        scene=dict(
            bgcolor=DARK_BG,
            xaxis=dict(
                title='Strike',
                gridcolor=GRID_COL,
                backgroundcolor=CARD_BG,
                titlefont=dict(color=TEXT_COL),
                tickfont=dict(color=TEXT_COL),
            ),
            yaxis=dict(
                title='Expiry',
                gridcolor=GRID_COL,
                backgroundcolor=CARD_BG,
                titlefont=dict(color=TEXT_COL),
                tickfont=dict(color=TEXT_COL, size=9),
            ),
            zaxis=dict(
                title='IV (%)',
                gridcolor=GRID_COL,
                backgroundcolor=CARD_BG,
                titlefont=dict(color=TEXT_COL),
                tickfont=dict(color=TEXT_COL),
            ),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.8)),
        ),
    )

    return fig



def vega_heatmap(raw_df, spot, ticker, window_pct=0.12):
    import plotly.graph_objects as go

    lo = spot * (1 - window_pct)
    hi = spot * (1 + window_pct)
    df = raw_df[(raw_df['strike'] >= lo) & (raw_df['strike'] <= hi)].copy()
    pivot = df.pivot_table(index='expiry', columns='strike', values='vega_exposure', aggfunc='sum', fill_value=0)
    z = pivot.values / 1e6  # show in $M
    fig = go.Figure(go.Heatmap(
        z=z,
        x=[str(c) for c in pivot.columns],
        y=pivot.index.tolist(),
        colorscale='RdYlGn',
        colorbar=dict(title='Vega Exposure ($M)', tickfont=dict(color=TEXT_COL)),
        hoverongaps=False
    ))
    fig.update_layout(**base_layout(f'Vega Exposure Heatmap — {ticker}', height=420),
                       xaxis_title='Strike', yaxis_title='Expiry')
    return fig


def term_structure_chart(raw_df, spot, ticker):
    import plotly.graph_objects as go

    rows = []
    for exp in sorted(raw_df['expiry'].unique()):
        sub = raw_df[(raw_df['expiry']==exp)].copy()
        # pick strike closest to spot (ATM)
        atm = sub.loc[(sub['strike'] - spot).abs().idxmin()]
        rows.append((exp, atm['iv']*100))
    df = pd.DataFrame(rows, columns=['expiry','atm_iv'])
    fig = go.Figure(go.Scatter(x=df['expiry'], y=df['atm_iv'], mode='lines+markers', line=dict(color=ACCENT_BLU)))
    fig.update_layout(**base_layout(f'ATM Term Structure — {ticker}', height=360), yaxis_title='ATM IV (%)', xaxis_title='Expiry')
    return fig


def skew_chart(raw_df, spot, ticker):
    # For each expiry, find approx 25-delta call and put vols
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    rows = []
    for exp in sorted(raw_df['expiry'].unique()):
        sub = raw_df[raw_df['expiry'] == exp].copy()
        calls = sub[sub['flag']=='c']
        puts  = sub[sub['flag']=='p']
        if calls.empty or puts.empty:
            continue
        # call delta target 0.25
        call_idx = (calls['delta'] - 0.25).abs().idxmin()
        put_idx  = (puts['delta'] + 0.25).abs().idxmin()
        call_iv = calls.loc[call_idx, 'iv'] * 100
        put_iv  = puts.loc[put_idx, 'iv'] * 100
        rows.append((exp, call_iv, put_iv, put_iv - call_iv))
    df = pd.DataFrame(rows, columns=['expiry','call25_iv','put25_iv','skew25'])
    fig = make_subplots(rows=1, cols=2, subplot_titles=('25Δ Call/Put Vols','25Δ Skew'))
    fig.add_trace(go.Scatter(x=df['expiry'], y=df['call25_iv'], mode='lines+markers', name='25Δ Call', line=dict(color=ACCENT_GRN)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['expiry'], y=df['put25_iv'], mode='lines+markers', name='25Δ Put', line=dict(color=ACCENT_RED)), row=1, col=1)
    fig.add_trace(go.Bar(x=df['expiry'], y=df['skew25'], name='25Δ Skew', marker_color=ACCENT_YLW), row=1, col=2)
    fig.update_layout(**base_layout(f'Skew Metrics — {ticker}', height=360))
    return fig


print('Chart builders ready.')


if __name__ == '__main__':
    # Delayed imports for Dash so that importing this module for Streamlit
    # doesn't require Dash to be installed or initialize the Dash app.
    import dash
    from dash import dcc, html, Input, Output, State, ctx
    import dash_bootstrap_components as dbc
    import plotly.graph_objects as go

    # Shared state will be initialized lazily after first fetch
    store = {
        'raw':       None,
        'by_strike': None,
        'by_expiry': None,
        'spot':      None,
        'ticker':    DEFAULT_TICKER,
    }

    # Stat card helper
    def stat_card(label, value, color=ACCENT_GRN):
        return html.Div([
            html.P(label, style={'color': '#7a8399', 'fontSize': '11px',
                                  'margin': '0', 'letterSpacing': '1.5px',
                                  'textTransform': 'uppercase', 'fontFamily': 'Courier New'}),
            html.H4(value, style={'color': color, 'margin': '4px 0 0',
                                   'fontFamily': 'Courier New', 'fontSize': '20px'})
        ], style={
            'background': '#13161e',
            'border': f'1px solid {GRID_COL}',
            'borderTop': f'3px solid {color}',
            'borderRadius': '4px',
            'padding': '14px 18px',
            'flex': '1',
            'minWidth': '160px',
        })

    def build_stats(raw_df, by_strike_df, spot):
        total_gex_val = raw_df['gex'].sum()
        total_dex_val = raw_df['dex'].sum()
        largest_strike = by_strike_df.loc[by_strike_df['net_gex'].abs().idxmax(), 'strike']
        call_oi = int(raw_df[raw_df['flag'] == 'c']['openInterest'].sum())
        put_oi = int(raw_df[raw_df['flag'] == 'p']['openInterest'].sum())
        pcr = put_oi / call_oi if call_oi else 0
        regime = '🟢 POSITIVE' if total_gex_val > 0 else '🔴 NEGATIVE'
        regime_col = ACCENT_GRN if total_gex_val > 0 else ACCENT_RED
        return [
            stat_card('Spot Price', f'${spot:.2f}', ACCENT_YLW),
            stat_card('Total GEX', f'${total_gex_val/1e9:.2f}B', ACCENT_GRN if total_gex_val > 0 else ACCENT_RED),
            stat_card('Total DEX', f'${total_dex_val/1e6:.0f}M', ACCENT_BLU),
            stat_card('GEX Regime', regime, regime_col),
            stat_card('Peak GEX Strike', f'{largest_strike}', ACCENT_GRN),
            stat_card('Put/Call OI Ratio', f'{pcr:.2f}', ACCENT_YLW),
        ]

    # App
    # Create an explicit Flask server with a defined instance_path to avoid
    # Flask attempting to auto-discover package paths (which can fail in some
    # container/mounted environments). Pass this server into Dash.
    import flask
    import os
    instance_dir = os.path.join(os.getcwd(), 'dash_instance')
    os.makedirs(instance_dir, exist_ok=True)
    # Use a known import name (here 'flask') so Flask can locate a loader
    # in environments where the local module/package is not importable
    # (mounted filesystems, containers, etc.). The import_name is used
    # only to compute resource paths; choosing a stable installed package
    # avoids pkgutil.get_loader returning None.
    server = flask.Flask('flask', instance_path=instance_dir)

    app = dash.Dash(
        __name__,
        server=server,
        external_stylesheets=[dbc.themes.CYBORG],
        title='DEX / GEX Dashboard'
     )

    HEADER = html.Div([
        html.Div([
            html.H2('⚡ DEX / GEX', style={'color': ACCENT_GRN, 'margin': '0',
                                          'fontFamily': 'Courier New', 'fontSize': '28px',
                                          'letterSpacing': '4px'}),
            html.P('Options Exposure Dashboard', style={'color': '#7a8399', 'margin': '0',
                                                        'fontFamily': 'Courier New',
                                                        'fontSize': '12px', 'letterSpacing': '2px'}),
        ]),
        html.Div([
            dcc.Input(id='ticker-input', value=DEFAULT_TICKER, type='text',
                      placeholder='Ticker …',
                      style={'background': '#1a1d27', 'border': f'1px solid {GRID_COL}',
                             'color': TEXT_COL, 'padding': '8px 12px', 'borderRadius': '4px',
                             'fontFamily': 'Courier New', 'fontSize': '14px',
                             'width': '100px', 'textTransform': 'uppercase'}),
            html.Div(
                dcc.Slider(
                    id='window-slider',
                    min=5,
                    max=25,
                    step=5,
                    value=10,
                    marks={v: f'{v}%' for v in [5, 10, 15, 20, 25]},
                    tooltip={'always_visible': False},
                    className='mx-3',
                ),
                style={'width': '200px'}
            ),
            html.Button('↻ REFRESH', id='refresh-btn',
                        style={'background': 'transparent', 'border': f'1px solid {ACCENT_GRN}',
                               'color': ACCENT_GRN, 'padding': '8px 20px',
                               'borderRadius': '4px', 'cursor': 'pointer',
                               'fontFamily': 'Courier New', 'fontSize': '13px',
                               'letterSpacing': '1px'}),
        ], style={'display': 'flex', 'alignItems': 'center', 'gap': '12px'}),
    ], style={
        'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center',
        'padding': '18px 28px', 'background': '#0d0f14',
        'borderBottom': f'1px solid {GRID_COL}',
    })

    app.layout = html.Div([
        HEADER,

        html.Div(id='status-bar', style={'padding': '8px 28px', 'fontSize': '12px',
                                         'color': '#7a8399', 'fontFamily': 'Courier New',
                                         'borderBottom': f'1px solid {GRID_COL}'}),

        html.Div(id='stat-cards',
                 style={'display': 'flex', 'gap': '12px', 'padding': '16px 28px',
                        'flexWrap': 'wrap'}),

        html.Div([
            html.Div(dcc.Graph(id='gex-bar'), style={'flex': '1', 'minWidth': '400px'}),
            html.Div(dcc.Graph(id='dex-bar'), style={'flex': '1', 'minWidth': '400px'}),
        ], style={'display': 'flex', 'gap': '12px', 'padding': '0 28px 12px'}),

        html.Div([
            html.Div(dcc.Graph(id='gex-expiry'), style={'flex': '1', 'minWidth': '300px'}),
            html.Div(dcc.Graph(id='oi-heatmap'), style={'flex': '1.3', 'minWidth': '380px'}),
            html.Div(dcc.Graph(id='vol-smile'), style={'flex': '1', 'minWidth': '300px'}),
        ], style={'display': 'flex', 'gap': '12px', 'padding': '0 28px 28px'}),

        dcc.Loading(html.Div(id='_dummy'), type='circle', color=ACCENT_GRN),

    ], style={'background': DARK_BG, 'minHeight': '100vh', 'fontFamily': 'Courier New'})


    # Callbacks
    @app.callback(
        Output('stat-cards', 'children'),
        Output('gex-bar', 'figure'),
        Output('dex-bar', 'figure'),
        Output('gex-expiry', 'figure'),
        Output('oi-heatmap', 'figure'),
        Output('vol-smile', 'figure'),
        Output('status-bar', 'children'),
        Output('_dummy', 'children'),
        Input('refresh-btn', 'n_clicks'),
        Input('window-slider', 'value'),
        State('ticker-input', 'value'),
        prevent_initial_call=False,
     )
    def update_dashboard(n_clicks, window_pct, ticker_val):
        triggered = ctx.triggered_id
        ticker = (ticker_val or DEFAULT_TICKER).strip().upper()
        wpct = (window_pct or 10) / 100

        if triggered == 'refresh-btn' or store['raw'] is None or ticker != store['ticker']:
            try:
                raw_df, spot = fetch_options_data(ticker)
                store['raw'] = raw_df
                store['by_strike'] = aggregate_by_strike(raw_df)
                store['by_expiry'] = aggregate_by_expiry(raw_df)
                store['spot'] = spot
                store['ticker'] = ticker
            except Exception as e:
                status = f'⚠  Error fetching {ticker}: {e}'
                empty = go.Figure().update_layout(**base_layout())
                return [], empty, empty, empty, empty, empty, status, ''

        raw_df = store['raw']
        bs_df = store['by_strike']
        be_df = store['by_expiry']
        spot = store['spot']
        ticker = store['ticker']
        ts = datetime.now().strftime('%H:%M:%S')

        stats = build_stats(raw_df, bs_df, spot)
        fig_gex = gex_bar_chart(bs_df, spot, ticker, wpct)
        fig_dex = dex_bar_chart(bs_df, spot, ticker, wpct)
        fig_exp = gex_expiry_chart(be_df, ticker)
        fig_oi = oi_heatmap(raw_df, spot, ticker, wpct)
        fig_vol = vol_smile_chart(raw_df, spot, ticker)
        status = f'Last updated: {ts}  |  {len(raw_df):,} contracts  |  {raw_df["expiry"].nunique()} expiries  |  Spot ${spot:.2f}'

        return stats, fig_gex, fig_dex, fig_exp, fig_oi, fig_vol, status, ''


    def _pick_port(preferred: list[int]) -> int:
        """Return the first available port from preferred, or an OS-assigned free port."""
        for p in preferred:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('127.0.0.1', p))
                s.close()
                return p
            except OSError:
                continue
        # Fallback: ask OS for a free port
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
        s.close()
        return port


    preferred_ports = [PORT, 8051, 8052]
    selected_port = _pick_port(preferred_ports)
    print('Dash app built.')
    print(f"\n🚀  Starting dashboard on  http://127.0.0.1:{selected_port}")
    if selected_port != PORT:
        print(f"  Note: preferred port {PORT} was unavailable; using {selected_port} instead.")
    print('   Press  Ctrl+C  in this terminal to stop.\n')
    app.run(debug=False, port=selected_port, host='127.0.0.1')
