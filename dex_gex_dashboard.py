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

import re
import time
import tempfile
import threading
import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
import requests
import socket
from datetime import datetime, date
# Defer Plotly imports to inside plotting functions so this module can be
# imported without requiring Plotly to be installed. Plotly is heavy and
# only needed when rendering charts (Streamlit/ Dash). See individual
# chart functions for local imports.

# ── Dashboard config ──────────────────────────────────────────────────────────
DEFAULT_TICKER    = '$SPX'  # Use $ prefix for indices (SPY, QQQ, AAPL work without)
RISK_FREE_RATE    = 0.053221  # Update to current Fed funds rate
SPX_DIV_YIELD     = 0.014     # SPX continuous dividend yield ≈ 1.4%
                              # Used in Merton (1973) extension of BS for index options.
                              # Forward: F = S·e^{(r-q)T}  →  adjusts delta, gamma,
                              # charm, vanna systematically.  Without q, delta is
                              # overstated for ITM calls and understated for ITM puts.
OI_THRESHOLD      = 100     # Minimum open interest to keep a strike
FETCH_EXPIRY_DAYS = 270     # Max DTE when downloading the chain.
MAX_EXPIRY_DAYS   = 90      # Default display window (sidebar DTE filter, post-fetch)
PORT              = 8051    # Browser port (safe default)

print(f'Config loaded. Default ticker: {DEFAULT_TICKER}')

# ── Black-Scholes helpers (Merton 1973 — continuous dividend yield q) ─────────

def _bs_d1d2(S, K, T, r, sigma, q=SPX_DIV_YIELD):
    """Internal helper: compute d1, d2 with dividend yield q."""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)


def bs_delta(S, K, T, r, sigma, flag, q=SPX_DIV_YIELD):
    """Black-Scholes-Merton delta with continuous dividend yield q."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, _ = _bs_d1d2(S, K, T, r, sigma, q)
    disc  = np.exp(-q * T)
    if flag == 'c':
        return float(disc * norm.cdf(d1))
    else:
        return float(disc * (norm.cdf(d1) - 1.0))


def bs_gamma(S, K, T, r, sigma, q=SPX_DIV_YIELD):
    """Black-Scholes-Merton gamma with continuous dividend yield q."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, _ = _bs_d1d2(S, K, T, r, sigma, q)
    return float(np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T)))


def bs_charm(S, K, T, r, sigma, flag, q=SPX_DIV_YIELD):
    """Charm = dDelta/dTime (delta decay per calendar day, annualised basis).

    Key for 0DTE: charm drives the delta-hedge the dealer must execute simply
    from the passage of time, independent of price movement.  Expressed as
    delta-units per year — divide by 252 to get daily delta drift.
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, d2 = _bs_d1d2(S, K, T, r, sigma, q)
    nd1    = norm.pdf(d1)
    sqrtT  = np.sqrt(T)
    disc   = np.exp(-q * T)
    # Common term
    inner  = 2.0 * (r - q) * T - d2 * sigma * sqrtT
    term   = disc * nd1 * inner / (2.0 * T * sigma * sqrtT)
    if flag == 'c':
        return float(-term + q * disc * norm.cdf(d1))
    else:
        return float(-term - q * disc * norm.cdf(-d1))


def bs_vanna(S, K, T, r, sigma, q=SPX_DIV_YIELD):
    """Vanna = dDelta/dVol = d2Vega/dS.

    Measures how much delta changes when implied vol moves.  When the market
    sells off and IV spikes simultaneously, dealers with short vanna must
    buy futures — amplifying the move.  Critical for understanding vol→spot
    correlation flows.
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, d2 = _bs_d1d2(S, K, T, r, sigma, q)
    return float(-np.exp(-q * T) * norm.pdf(d1) * d2 / sigma)


def bs_vomma(S, K, T, r, sigma, q=SPX_DIV_YIELD):
    """Vomma (Volga) = dVega/dVol.  Convexity of vega wrt implied vol."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, d2 = _bs_d1d2(S, K, T, r, sigma, q)
    vega   = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
    return float(vega * d1 * d2 / sigma)



def bs_rho(S, K, T, r, sigma, flag, q=SPX_DIV_YIELD):
    """Rho = dV/dr per 1 basis point (0.01% rate change).

    Material for positions held >60d with rates at 5%.
    Call rho positive (higher rates → higher call value via forward).
    Put rho negative.  Expressed as $ change per 1bp.
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    _, d2 = _bs_d1d2(S, K, T, r, sigma, q)
    if flag == 'c':
        return float(K * T * np.exp(-r * T) * norm.cdf(d2)  / 100.0)
    else:
        return float(-K * T * np.exp(-r * T) * norm.cdf(-d2) / 100.0)


def bs_greeks_vectorized(S, K, T, r, sigma, flag, q=SPX_DIV_YIELD):
    """Vectorized BSM delta and gamma with continuous dividend yield."""
    K     = np.asarray(K,     dtype=float)
    T     = np.asarray(T,     dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    flag  = np.asarray(flag)

    delta = np.zeros_like(K, dtype=float)
    gamma = np.zeros_like(K, dtype=float)

    valid = (T > 0) & (sigma > 0) & np.isfinite(K) & np.isfinite(T) & np.isfinite(sigma)
    if not np.any(valid):
        return delta, gamma

    sqT   = np.sqrt(T[valid])
    d1    = (np.log(S / K[valid]) + (r - q + 0.5 * sigma[valid]**2) * T[valid]) / (sigma[valid] * sqT)
    disc  = np.exp(-q * T[valid])
    cdfD1 = norm.cdf(d1)

    delta[valid] = np.where(flag[valid] == 'c', disc * cdfD1, disc * (cdfD1 - 1.0))
    gamma[valid] = disc * norm.pdf(d1) / (S * sigma[valid] * sqT)
    return delta, gamma


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

# ── In-memory data cache ───────────────────────────────────────────────
_DATA_CACHE: dict = {}
_CACHE_TTL = 600  # seconds (10 minutes)


def _cache_key(ticker, max_days, oi_thresh, r):
    return (ticker.upper(), max_days, oi_thresh, round(r, 6))


def _cache_get(key):
    entry = _DATA_CACHE.get(key)
    if entry and time.time() - entry[2] < _CACHE_TTL:
        return entry[0], entry[1]
    return None, None


def _cache_set(key, data, spot):
    _DATA_CACHE[key] = (data, spot, time.time())


# ── CSV file cache ─────────────────────────────────────────────────────
import os, json as _json

_CSV_CACHE_DIR = os.path.join(tempfile.gettempdir(), 'dex_gex_cache')
os.makedirs(_CSV_CACHE_DIR, exist_ok=True)
_CSV_MAX_AGE_HOURS = 8  # treat file as stale after this many hours


def _csv_paths(ticker: str):
    safe = ticker.upper().lstrip('$')
    return (
        os.path.join(_CSV_CACHE_DIR, f'{safe}_options.csv'),
        os.path.join(_CSV_CACHE_DIR, f'{safe}_meta.json'),
    )


def _csv_load(ticker: str):
    csv_path, meta_path = _csv_paths(ticker)
    if not (os.path.exists(csv_path) and os.path.exists(meta_path)):
        return None, None
    age_hours = (time.time() - os.path.getmtime(csv_path)) / 3600
    if age_hours > _CSV_MAX_AGE_HOURS:
        return None, None
    try:
        data = pd.read_csv(csv_path, low_memory=False)
        with open(meta_path) as f:
            meta = _json.load(f)
        print(f'CSV cache hit for {ticker} ({age_hours:.1f}h old).')
        return data, float(meta['spot'])
    except Exception as exc:
        print(f'CSV cache read failed: {exc}')
        return None, None


def _csv_save(ticker: str, data: pd.DataFrame, spot: float):
    csv_path, meta_path = _csv_paths(ticker)
    try:
        data.to_csv(csv_path, index=False)
        with open(meta_path, 'w') as f:
            _json.dump({'spot': spot, 'saved_at': time.time()}, f)
        print(f'CSV cache saved → {csv_path}')
    except Exception as exc:
        print(f'CSV cache write failed: {exc}')


# ── Data fetching & processing ────────────────────────────────────────

BARCHART_USER_AGENT = (
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
    'AppleWebKit/537.36 (KHTML, like Gecko) '
    'Chrome/124.0.0.0 Safari/537.36'
)
BARCHART_API_URL  = 'https://www.barchart.com/proxies/core-api/v1/options/get'
BARCHART_HIST_URL = 'https://www.barchart.com/proxies/core-api/v1/historical/get'
BARCHART_FIELDS = (
    'symbol,baseSymbol,strikePrice,expirationDate,moneyness,bidPrice,midpoint,'
    'askPrice,lastPrice,priceChange,percentChange,volume,openInterest,'
    'openInterestChange,volatility,delta,optionType,daysToExpiration,'
    'tradeTime,averageVolatility,historicVolatility30d,baseNextEarningsDate,'
    'dividendExDate,baseTimeCode,expirationType,impliedVolatilityRank1y'
)


def build_barchart_session(ticker: str) -> requests.Session:
    session = requests.Session()
    session.headers.update({
        'User-Agent': BARCHART_USER_AGENT,
        'Accept-Language': 'en-US,en;q=0.9',
    })
    response = session.get(barchart_page_url(ticker), timeout=30)
    response.raise_for_status()
    return session


def resolve_barchart_symbol(ticker: str) -> str:
    raw = (ticker or '').strip().upper()
    if not raw:
        raise ValueError('Ticker is required.')

    candidates = [raw]
    if not raw.startswith('$'):
        candidates.append(f'${raw}')

    # Only one possible form (e.g. an index already in canonical '$' form like
    # $SPX / $VIX / $NDX): there is nothing to disambiguate, so skip the network
    # probe entirely — it cannot change the result and can only add latency or
    # hang. build_barchart_session() validates the symbol on the next step.
    if len(candidates) == 1:
        return raw

    for candidate in candidates:
        try:
            response = requests.get(
                barchart_page_url(candidate),
                headers={
                    'User-Agent': BARCHART_USER_AGENT,
                    'Accept-Language': 'en-US,en;q=0.9',
                },
                timeout=8,
            )
            if response.ok and 'Page not found' not in response.text:
                return candidate
        except requests.RequestException:
            continue

    return raw


def barchart_page_url(ticker: str) -> str:
    return f'https://www.barchart.com/stocks/quotes/{ticker}/options'


def barchart_api_headers(session: requests.Session, ticker: str) -> dict:
    xsrf_token = requests.utils.unquote(session.cookies.get('XSRF-TOKEN', ''))
    return {
        'User-Agent': BARCHART_USER_AGENT,
        'Accept': 'application/json, text/plain, */*',
        'Referer': barchart_page_url(ticker),
        'Origin': 'https://www.barchart.com',
        'X-Requested-With': 'XMLHttpRequest',
        'X-XSRF-TOKEN': xsrf_token,
    }


def barchart_api_params(ticker: str, expiration: str) -> dict:
    return {
        'baseSymbol': ticker,
        'fields': BARCHART_FIELDS,
        'groupBy': 'optionType',
        'expirationDate': expiration,
        'meta': 'field.shortName,expirations',
        'orderBy': 'strikePrice',
        'orderDir': 'asc',
        'optionsOverview': 'true',
    }


def flatten_barchart_expirations(meta: dict, max_dte: int = FETCH_EXPIRY_DAYS) -> list[str]:
    """Return future expirations (weekly + monthly) within max_dte days.

    Caps at FETCH_EXPIRY_DAYS (1 year) by default so the download never touches
    thin, slow LEAP expirations beyond that horizon. Display filtering is handled
    separately by apply_dashboard_filters / the sidebar DTE slider.
    """
    expirations = []
    exp_meta = meta.get('expirations', {}) if isinstance(meta, dict) else {}
    for group_name in ('weekly', 'monthly'):
        expirations.extend(exp_meta.get(group_name, []))

    today = date.today()
    filtered = []
    for expiration in sorted(set(expirations)):
        exp_date = datetime.strptime(expiration, '%Y-%m-%d').date()
        dte = (exp_date - today).days
        if 0 <= dte <= max_dte:   # 0 = today (0DTE), previously excluded
            filtered.append(expiration)
    return filtered


def to_float(value):
    if value in (None, '', 'N/A', 'unch', '--'):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = str(value).replace(',', '').replace('%', '')
    try:
        return float(cleaned)
    except ValueError:
        return np.nan


def normalize_barchart_row(row: dict, expiration: str, option_type: str) -> dict:
    return {
        'contractSymbol': row.get('symbol'),
        'strike': to_float(row.get('strikePrice')),
        'bid': to_float(row.get('bidPrice')),
        'ask': to_float(row.get('askPrice')),
        'lastPrice': to_float(row.get('lastPrice')),
        'volume': to_float(row.get('volume')),
        'openInterest': to_float(row.get('openInterest')),
        'impliedVolatility': to_float(row.get('volatility')),
        'delta_barchart': to_float(row.get('delta')),
        'moneyness': to_float(row.get('moneyness')),
        'optionType': option_type.lower(),
        'expiry': expiration,
        'expirationType': row.get('expirationType'),
        'tradeTime': row.get('tradeTime'),
    }


def barchart_payload_to_frame(payload: dict, expiration: str) -> pd.DataFrame:
    rows = []
    data = payload.get('data', {}) if isinstance(payload, dict) else {}
    for option_type in ('Call', 'Put'):
        for contract in data.get(option_type, []):
            if isinstance(contract, dict):
                rows.append(normalize_barchart_row(contract, expiration, option_type))
    return pd.DataFrame(rows)


def fetch_barchart_payload(session: requests.Session, ticker: str, expiration: str) -> dict:
    response = session.get(
        BARCHART_API_URL,
        headers=barchart_api_headers(session, ticker),
        params=barchart_api_params(ticker, expiration),
        timeout=_FETCH_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def fetch_barchart_spot(session: requests.Session, ticker: str) -> float:
    response = session.get(barchart_page_url(ticker), timeout=30)
    response.raise_for_status()

    match = re.search(r'"lastPrice":"([0-9,]+(?:\.[0-9]+)?)"', response.text)
    if not match:
        raise ValueError(f'Unable to parse Barchart spot quote for {ticker}.')

    spot = float(match.group(1).replace(',', ''))
    if not np.isfinite(spot):
        raise ValueError(f'Parsed Barchart spot for {ticker} is not finite.')
    return spot


_FETCH_ATTEMPTS   = 3
_FETCH_DELAY      = 0.1   # seconds between expiration requests
_FETCH_TIMEOUT    = 15    # seconds per request



def compute_implied_div_yield(data: pd.DataFrame, S: float,
                               r: float = RISK_FREE_RATE) -> dict:
    """Implied dividend yield per expiry via put-call parity on the ATM pair.

    C - P = S·e^{-qT} - K·e^{-rT}   →   q = -ln((C-P + K·e^{-rT}) / S) / T

    Uses the strike closest to spot for each expiry.  Falls back to the
    constant SPX_DIV_YIELD on any numerical failure.
    """
    q_map = {}
    for exp, grp in data.groupby('expiry'):
        T = float(grp['T_years'].iloc[0]) if 'T_years' in grp.columns else 0.0
        if T <= 0:
            q_map[exp] = SPX_DIV_YIELD
            continue
        calls = grp[grp['flag'] == 'c']
        puts  = grp[grp['flag'] == 'p']
        if calls.empty or puts.empty:
            q_map[exp] = SPX_DIV_YIELD
            continue
        # ATM strike
        K = float(calls.iloc[(calls['strike'] - S).abs().argsort().iloc[0]]['strike'])
        c_rows = calls[calls['strike'] == K]
        p_rows = puts[puts['strike'] == K]
        if c_rows.empty or p_rows.empty:
            q_map[exp] = SPX_DIV_YIELD
            continue
        C = float(c_rows['mid'].iloc[0])
        P = float(p_rows['mid'].iloc[0])
        if C <= 0 or P <= 0:
            q_map[exp] = SPX_DIV_YIELD
            continue
        try:
            rhs = C - P + K * np.exp(-r * T)
            if rhs <= 0 or rhs >= S * 2:
                q_map[exp] = SPX_DIV_YIELD
                continue
            q_impl = -np.log(rhs / S) / T
            q_map[exp] = float(np.clip(q_impl, -0.01, 0.06))
        except Exception:
            q_map[exp] = SPX_DIV_YIELD
    return q_map


def validate_iv(data: pd.DataFrame) -> pd.DataFrame:
    """IV sanity filter: remove contracts with clearly erroneous volatility.

    Hard bounds: IV < 0.5% or IV > 500%.
    Soft outlier: per-expiry IV > median + 5×IQR (fat-tailed upper bound).
    Removes ~0.5-2% of contracts from typical SPX chains.
    """
    data = data[(data['iv'] >= 0.005) & (data['iv'] <= 5.0)].copy()
    cleaned = []
    for _, grp in data.groupby('expiry', sort=False):
        iv_med = grp['iv'].median()
        iv_iqr = max(grp['iv'].quantile(0.75) - grp['iv'].quantile(0.25), 0.01)
        upper  = iv_med + 5.0 * iv_iqr
        cleaned.append(grp[grp['iv'] <= upper])
    if not cleaned:
        return data
    return pd.concat(cleaned).reset_index(drop=True)


def fetch_intraday_history(ticker: str, interval_min: int = 5) -> pd.DataFrame:
    """Fetch today's intraday OHLC bars.

    Primary source: yfinance (free, no auth, up to 60 days of 5-min history).
    Fallback: Barchart historical endpoint (requires premium subscription).

    Returns DataFrame(datetime, open, high, low, close, volume) sorted
    ascending for today's session, or an empty DataFrame if unavailable
    (market closed, weekend). The caller falls back to showing the spot
    price as a horizontal line when this returns empty.
    """
    from datetime import date as _date, datetime as _dt, timezone as _tz
    import pytz

    interval_str = f'{interval_min}m'

    # ── 1. yfinance (primary) ─────────────────────────────────────────────────
    yf_sym = _yf_ticker(ticker)
    try:
        import yfinance as yf

        def _parse_yf_intraday(raw) -> pd.DataFrame:
            """Normalise a yfinance download result to (datetime, ohlcv)."""
            if raw is None or raw.empty:
                return pd.DataFrame()
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [c[0] for c in raw.columns]
            raw.columns = [str(c).lower() for c in raw.columns]
            raw = raw.reset_index()
            dt_col = next((c for c in raw.columns
                           if 'datetime' in c.lower() or c.lower() == 'date'), None)
            if not dt_col:
                return pd.DataFrame()
            raw = raw.rename(columns={dt_col: 'datetime'})
            raw['datetime'] = pd.to_datetime(raw['datetime'])
            if raw['datetime'].dt.tz is not None:
                et = pytz.timezone('America/New_York')
                raw['datetime'] = (raw['datetime']
                                   .dt.tz_convert(et)
                                   .dt.tz_localize(None))
            keep = [c for c in
                    ['datetime', 'open', 'high', 'low', 'close', 'volume']
                    if c in raw.columns]
            return (raw[keep]
                      .dropna(subset=['datetime', 'close'])
                      .query('close > 0')
                      .sort_values('datetime')
                      .reset_index(drop=True))

        # Try today's session first
        raw_today = yf.download(yf_sym, period='1d', interval=interval_str,
                                auto_adjust=True, progress=False)
        df = _parse_yf_intraday(raw_today)
        if not df.empty:
            print(f'  [intraday/yf] {yf_sym}: {len(df)} bars today '
                  f'({df["datetime"].iloc[0].strftime("%H:%M")}–'
                  f'{df["datetime"].iloc[-1].strftime("%H:%M")} ET)')
            return df

        # Market closed or pre-open: fall back to last available session
        print(f'  [intraday/yf] today empty → fetching last available session')
        raw_5d = yf.download(yf_sym, period='5d', interval=interval_str,
                             auto_adjust=True, progress=False)
        df5 = _parse_yf_intraday(raw_5d)
        if not df5.empty:
            last_date = df5['datetime'].dt.normalize().max()
            df_last   = df5[df5['datetime'].dt.normalize() == last_date].copy()
            if not df_last.empty:
                print(f'  [intraday/yf] {yf_sym}: {len(df_last)} bars '
                      f'from last session {last_date.date()}')
                return df_last.reset_index(drop=True)

    except Exception as e:
        print(f'  [intraday/yf] {yf_sym}: {type(e).__name__}: {e}')

    # ── 2. Barchart fallback ──────────────────────────────────────────────────
    today_str = _date.today().strftime('%Y-%m-%d')
    _attempts = [
        dict(type='minutes', interval=str(interval_min)),
        dict(type='minute',  interval=str(interval_min)),
    ]
    try:
        session = build_barchart_session(ticker)
        headers = barchart_api_headers(session, ticker)
        for extra in _attempts:
            params = {
                'symbol': ticker, 'startDate': today_str, 'endDate': today_str,
                'fields': 'tradingDay,time,open,high,low,close,volume',
                'limit': 500, **extra,
            }
            try:
                resp = session.get(BARCHART_HIST_URL, headers=headers,
                                   params=params, timeout=12)
                if resp.status_code != 200:
                    continue
                items = (resp.json().get('data')
                         or resp.json().get('results') or [])
                rows = []
                for item in items:
                    raw = item.get('raw', item)
                    try:
                        day_s  = raw.get('tradingDay') or raw.get('date') or today_str
                        time_s = raw.get('time') or raw.get('tradeTime') or '00:00:00'
                        dt     = _dt.strptime(f'{day_s} {str(time_s)[:8]}',
                                              '%Y-%m-%d %H:%M:%S')
                        rows.append({
                            'datetime': dt,
                            'open':    float(raw.get('open',  0) or 0),
                            'high':    float(raw.get('high',  0) or 0),
                            'low':     float(raw.get('low',   0) or 0),
                            'close':   float(raw.get('close', 0) or 0),
                            'volume':  float(raw.get('volume', 0) or 0),
                        })
                    except (ValueError, TypeError):
                        continue
                if rows:
                    df = (pd.DataFrame(rows)
                            .query('close > 0')
                            .sort_values('datetime')
                            .reset_index(drop=True))
                    print(f'  [intraday/bc] {ticker}: {len(df)} bars')
                    return df
            except Exception as e:
                print(f'  [intraday/bc] {extra}: {e}')
    except Exception as exc:
        print(f'  [intraday] Barchart session error: {exc}')

    print(f'  [intraday] all sources failed — spot fallback will be used')
    return pd.DataFrame()


# Backward-compatibility alias
def fetch_price_history(ticker: str, n_days: int = 5) -> pd.DataFrame:
    """Deprecated alias — use fetch_intraday_history."""
    return fetch_intraday_history(ticker)



    """Fetch the last n_days of daily OHLC for ticker from Barchart.

    Tries several parameter variants of the Barchart historical endpoint
    because the exact accepted parameters vary by account tier.  Prints
    diagnostic output so stall/failure reasons are visible in the terminal.

    Returns a DataFrame(date, open, high, low, close, volume) sorted
    ascending, or an empty DataFrame if no attempt succeeds.
    """
    from datetime import date as _date, timedelta as _td
    end_dt   = _date.today()
    start_dt = end_dt - _td(days=n_days * 3 + 10)

    # Multiple parameter combinations to try — different Barchart tiers accept
    # different keys (interval vs type, dashed vs bare dates, etc.)
    _attempts = [
        dict(type='daily',    orderBy='tradingDay', orderDir='asc'),
        dict(interval='daily', orderBy='tradingDay', orderDir='asc'),
        dict(type='daily'),
        dict(interval='daily'),
    ]

    def _parse_items(items: list) -> pd.DataFrame:
        rows = []
        for item in items:
            raw = item.get('raw', item)
            # Field names vary: tradingDay / date / tradeDay / dateTime
            date_str = (raw.get('tradingDay') or raw.get('date')
                        or raw.get('tradeDay') or raw.get('dateTime') or '')
            if not date_str:
                continue
            try:
                rows.append({
                    'date':  pd.Timestamp(str(date_str)[:10]),
                    'open':  float(raw.get('open',  raw.get('dailyOpen',  0)) or 0),
                    'high':  float(raw.get('high',  raw.get('dailyHigh',  0)) or 0),
                    'low':   float(raw.get('low',   raw.get('dailyLow',   0)) or 0),
                    'close': float(raw.get('close', raw.get('lastPrice',
                                   raw.get('dailyClose', 0))) or 0),
                    'volume': float(raw.get('volume', 0) or 0),
                })
            except (ValueError, TypeError):
                continue
        if not rows:
            return pd.DataFrame()
        return (pd.DataFrame(rows)
                  .dropna(subset=['date'])
                  .query('close > 0')
                  .sort_values('date')
                  .tail(n_days)
                  .reset_index(drop=True))

    try:
        session = build_barchart_session(ticker)
        headers = barchart_api_headers(session, ticker)

        for extra in _attempts:
            params = {
                'symbol':    ticker,
                'startDate': start_dt.strftime('%Y-%m-%d'),
                'endDate':   end_dt.strftime('%Y-%m-%d'),
                'fields':    'tradingDay,open,high,low,close,volume',
                'meta':      'field.shortName',
                'limit':     n_days + 10,
                **extra,
            }
            try:
                resp = session.get(BARCHART_HIST_URL, headers=headers,
                                   params=params, timeout=15)
                print(f'  [hist] {extra} → HTTP {resp.status_code}')
                if resp.status_code != 200:
                    print(f'  [hist] body preview: {resp.text[:200]}')
                    continue

                payload = resp.json()
                # Response can nest data under 'data', 'results', or directly as list
                items = (payload.get('data')
                         or payload.get('results')
                         or (payload if isinstance(payload, list) else []))
                print(f'  [hist] keys={list(payload.keys()) if isinstance(payload,dict) else "list"}'
                      f'  items={len(items)}')
                if not items:
                    print(f'  [hist] empty payload preview: {str(payload)[:300]}')
                    continue

                df = _parse_items(items)
                if not df.empty:
                    print(f'  [hist] OK: {len(df)} sessions for {ticker}')
                    return df
                print(f'  [hist] items found but none parsed — sample: {items[0]}')

            except Exception as e:
                print(f'  [hist] attempt {extra} error: {type(e).__name__}: {e}')
                continue

        print(f'  [hist] all attempts exhausted — price chart will be empty')
        return pd.DataFrame()

    except Exception as exc:
        print(f'  fetch_price_history({ticker}) session error: {type(exc).__name__}: {exc}')
        return pd.DataFrame()



# ── Ticker mapping ─────────────────────────────────────────────────────────────
_YF_MAP = {
    '$SPX': '^GSPC', '$SPY': 'SPY', '$NDX': '^NDX',
    '$VIX': '^VIX',  '^VIX': '^VIX', 'VIX':  '^VIX',
    'SPY': 'SPY', 'QQQ': 'QQQ', 'IWM': 'IWM',
}

def _yf_ticker(bc: str) -> str:
    return _YF_MAP.get(bc.upper(), bc)


def _yf_download(yf_ticker: str, n_calendar_days: int) -> pd.DataFrame:
    """Download daily OHLC from Yahoo Finance via yfinance.

    Handles both flat and MultiIndex column layouts across yfinance versions.
    Returns DataFrame(date, open, high, low, close, volume) or empty.
    """
    from datetime import date as _d, timedelta as _td
    try:
        import yfinance as yf
    except ImportError:
        print(f'  [yfinance] not installed — add yfinance to requirements.txt')
        return pd.DataFrame()

    end_s   = _d.today().strftime('%Y-%m-%d')
    start_s = (_d.today() - _td(days=n_calendar_days)).strftime('%Y-%m-%d')

    try:
        raw = yf.download(
            yf_ticker, start=start_s, end=end_s,
            auto_adjust=True, progress=False,
        )
    except Exception as e:
        print(f'  [yfinance] {yf_ticker} download error: {e}')
        return pd.DataFrame()

    if raw is None or raw.empty:
        print(f'  [yfinance] {yf_ticker}: empty response')
        return pd.DataFrame()

    # Flatten MultiIndex columns (yfinance >= 0.2.54 multi-ticker mode)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]
    raw.columns = [str(c).lower() for c in raw.columns]

    raw = raw.reset_index()
    date_col = next((c for c in raw.columns if 'date' in c.lower()), None)
    if not date_col:
        print(f'  [yfinance] {yf_ticker}: no date column found')
        return pd.DataFrame()

    raw = raw.rename(columns={date_col: 'date'})
    raw['date'] = pd.to_datetime(raw['date']).dt.normalize()

    keep = [c for c in ['date', 'open', 'high', 'low', 'close', 'volume']
            if c in raw.columns]
    df = (raw[keep]
            .dropna(subset=['date', 'close'])
            .query('close > 0')
            .sort_values('date')
            .reset_index(drop=True))
    print(f'  [yfinance] {yf_ticker}: {len(df)} sessions '
          f'({df["date"].iloc[0].date()} → {df["date"].iloc[-1].date()})')
    return df


def fetch_ohlc_history(ticker: str = '$SPX',
                       n_calendar_days: int = 210) -> pd.DataFrame:
    """Fetch daily OHLC for the last n_calendar_days.

    Primary source: yfinance (free, no auth required).
    Fallback: Barchart historical endpoint (requires premium subscription).
    Returns DataFrame(date, open, high, low, close, volume).
    """
    from datetime import date as _date, timedelta as _td

    # ── 1. yfinance (primary — always free) ──────────────────────────────────
    yf_sym = _yf_ticker(ticker)
    df = _yf_download(yf_sym, n_calendar_days)
    if not df.empty:
        return df

    # ── 2. Barchart fallback (premium accounts only) ──────────────────────────
    print(f'  [ohlc] yfinance failed for {yf_sym} → trying Barchart')
    end_dt   = _date.today()
    start_dt = end_dt - _td(days=n_calendar_days)
    try:
        session = build_barchart_session(ticker)
        headers = barchart_api_headers(session, ticker)
        for extra in [dict(type='daily'), dict(interval='daily'), dict(type='eod')]:
            params = {
                'symbol': ticker,
                'startDate': start_dt.strftime('%Y-%m-%d'),
                'endDate':   end_dt.strftime('%Y-%m-%d'),
                'fields':    'tradingDay,open,high,low,close,volume',
                'limit':     300,
                **extra,
            }
            try:
                resp = session.get(BARCHART_HIST_URL, headers=headers,
                                   params=params, timeout=15)
                if resp.status_code != 200:
                    continue
                items = (resp.json().get('data')
                         or resp.json().get('results') or [])
                rows = []
                for item in items:
                    raw = item.get('raw', item)
                    ds  = raw.get('tradingDay') or raw.get('date') or ''
                    if not ds:
                        continue
                    try:
                        rows.append({
                            'date':   pd.Timestamp(str(ds)[:10]),
                            'open':   float(raw.get('open',  0) or 0),
                            'high':   float(raw.get('high',  0) or 0),
                            'low':    float(raw.get('low',   0) or 0),
                            'close':  float(raw.get('close', 0) or 0),
                            'volume': float(raw.get('volume', 0) or 0),
                        })
                    except (ValueError, TypeError):
                        continue
                if rows:
                    df = (pd.DataFrame(rows)
                            .query('close > 0')
                            .sort_values('date')
                            .reset_index(drop=True))
                    print(f'  [ohlc] Barchart OK: {len(df)} sessions for {ticker}')
                    return df
            except Exception as e:
                print(f'  [ohlc] Barchart attempt {extra}: {e}')
    except Exception as exc:
        print(f'  [ohlc] Barchart session error: {exc}')

    print(f'  [ohlc] all sources failed for {ticker}')
    return pd.DataFrame()


def fetch_vix_history(n_calendar_days: int = 210) -> pd.DataFrame:
    """Fetch VIX daily history.

    Primary: yfinance ^VIX.
    Returns DataFrame(date, vix) where vix is decimal (0.18 = 18%).
    """
    df = _yf_download('^VIX', n_calendar_days)
    if not df.empty:
        result = df[['date', 'close']].copy()
        result['vix'] = result['close'] / 100.0
        return result[['date', 'vix']]

    # Barchart fallback
    from datetime import date as _d, timedelta as _td
    end_dt   = _d.today()
    start_dt = end_dt - _td(days=n_calendar_days)
    for vix_sym in ('$VIX', '^VIX'):
        try:
            session = build_barchart_session(vix_sym)
            headers = barchart_api_headers(session, vix_sym)
            for extra in [dict(type='daily'), dict(interval='daily')]:
                params = {'symbol': vix_sym,
                          'startDate': start_dt.strftime('%Y-%m-%d'),
                          'endDate':   end_dt.strftime('%Y-%m-%d'),
                          'fields': 'tradingDay,close', 'limit': 300, **extra}
                resp = session.get(BARCHART_HIST_URL, headers=headers,
                                   params=params, timeout=12)
                if resp.status_code != 200:
                    continue
                items = resp.json().get('data') or resp.json().get('results') or []
                rows = [{'date': pd.Timestamp(str(
                            (i.get('raw', i).get('tradingDay') or
                             i.get('raw', i).get('date') or ''))[:10]),
                          'vix': float(
                            i.get('raw', i).get('close') or
                            i.get('raw', i).get('lastPrice') or 0) / 100.0}
                        for i in items
                        if (i.get('raw', i).get('tradingDay') or i.get('raw', i).get('date'))
                        and float(i.get('raw', i).get('close') or
                                  i.get('raw', i).get('lastPrice') or 0) > 0]
                if rows:
                    return (pd.DataFrame(rows)
                              .sort_values('date')
                              .reset_index(drop=True))
        except Exception:
            continue

    print('  fetch_vix_history: all sources failed')
    return pd.DataFrame()




def fetch_options_data(ticker: str, r: float = RISK_FREE_RATE,
                       progress_callback=None, force_refresh: bool = False):
    """
    Fetch the FULL option chain from Barchart (no DTE or OI filters).
    Computes per-contract Greeks, DEX and GEX and saves to CSV.
    Filters are applied later in the dashboard from the cached dataset.
    """

    def notify(progress: int, message: str):
        if progress_callback is not None:
            progress_callback(progress, message)

    key = _cache_key(ticker.upper().lstrip('$'), 0, 0, r)

    if not force_refresh:
        # 1. In-memory cache — instant
        cached_data, cached_spot = _cache_get(key)
        if cached_data is not None:
            notify(100, f'Using cached {ticker} data (< 10 min old)')
            return cached_data, cached_spot

        # 2. CSV file cache — survives restarts. Read on a background thread so a
        #    slow disk (OneDrive-synced home, network share, AV scan) never freezes
        #    the progress bar. If the read takes > 10 s, skip and do a live fetch.
        notify(5, f'Loading {ticker} from local CSV cache ...')
        _csv_result = [None, None]
        def _do_csv_load():
            _csv_result[0], _csv_result[1] = _csv_load(ticker)
        _t = threading.Thread(target=_do_csv_load, daemon=True)
        _t.start()
        _t.join(timeout=10)
        if _csv_result[0] is not None:
            notify(100, f'Loaded {ticker} from local CSV cache')
            _cache_set(key, _csv_result[0], _csv_result[1])
            return _csv_result[0], _csv_result[1]
        elif _t.is_alive():
            notify(5, f'CSV cache read too slow, switching to live feed ...')

    # ── Live fetch from Barchart (full dataset, no filters) ────────────
    print(f'Fetching full dataset for {ticker} ...')
    notify(5, f'Opening Barchart session for {ticker} ...')

    # Open session + read spot + expiry list, retrying with a fresh session on
    # transient failure — mirrors the standalone scraper, which retries every
    # request. Previously these three were single-shot, so one slow response or
    # hiccup on the heavy $SPX page killed the whole load with no recovery.
    last_exc = None
    for attempt in range(_FETCH_ATTEMPTS):
        try:
            session = build_barchart_session(ticker)
            notify(15, f'Fetching {ticker} spot quote ...')
            S = fetch_barchart_spot(session, ticker)
            notify(20, f'Fetching {ticker} expiration list ...')
            nearest_payload = fetch_barchart_payload(session, ticker, 'nearest')
            break
        except Exception as exc:
            last_exc = exc
            print(f'  Barchart open attempt {attempt+1}/{_FETCH_ATTEMPTS} failed: '
                  f'{type(exc).__name__}: {str(exc).splitlines()[0]}')
            if attempt < _FETCH_ATTEMPTS - 1:
                notify(5, f'Retry {attempt+2}/{_FETCH_ATTEMPTS}: reconnecting to Barchart ...')
                time.sleep(1)
    else:
        raise RuntimeError(
            f'Barchart connection failed after {_FETCH_ATTEMPTS} attempts: {last_exc}'
        )

    expiries = flatten_barchart_expirations(nearest_payload.get('meta', {}))

    if not expiries:
        raise ValueError('No expirations found.')

    print(f'  {len(expiries)} expirations to fetch. Spot: ${S:.2f}')

    rows = []
    loop_base = 25
    loop_span = 60

    for index, exp_str in enumerate(expiries, 1):
        pct = loop_base + int(loop_span * index / len(expiries))
        notify(pct, f'[{index}/{len(expiries)}] Fetching {exp_str} ...')
        frame = pd.DataFrame()

        # Near-term expirations (≤180 DTE) get full retries — they're the core
        # GEX/DEX data. Long-dated ones (>180 DTE, thin OI, IV-surface only)
        # get one attempt: if Barchart is slow for them, skip rather than stall.
        exp_dte  = (date.fromisoformat(exp_str) - date.today()).days
        attempts = _FETCH_ATTEMPTS if exp_dte <= 180 else 1

        for attempt in range(attempts):
            try:
                payload = fetch_barchart_payload(session, ticker, exp_str)
                frame = barchart_payload_to_frame(payload, exp_str)
                break
            except requests.exceptions.Timeout:
                print(f'  [{index}/{len(expiries)}] {exp_str} attempt {attempt+1} timed out')
                if attempt < attempts - 1:
                    notify(pct, f'[{index}/{len(expiries)}] Retry {attempt+2}: {exp_str} ...')
                    time.sleep(0.5)
            except Exception as exc:
                print(f'  [{index}/{len(expiries)}] {exp_str} attempt {attempt+1} failed: '
                      f'{type(exc).__name__}: {str(exc).splitlines()[0]}')
                if attempt < attempts - 1:
                    notify(pct, f'[{index}/{len(expiries)}] Reconnecting, retry {attempt+2}: {exp_str} ...')
                    try:
                        session = build_barchart_session(ticker)
                    except Exception:
                        pass
                    time.sleep(1)

        if not frame.empty:
            rows.append(frame)
        else:
            print(f'  [{index}/{len(expiries)}] {exp_str} — no data, skipping')

        time.sleep(_FETCH_DELAY)

    if not rows:
        raise ValueError('No options data returned from Barchart.')

    notify(87, f'Assembling {ticker} chain ({len(rows)} expiries) ...')
    data = pd.concat(rows, ignore_index=True)

    notify(90, f'Processing {ticker} option chain ({len(data):,} rows) ...')
    today = date.today()
    expiry_dt    = pd.to_datetime(data['expiry'])
    data['T_days']  = (expiry_dt - pd.Timestamp(today)).dt.days
    data = data[data['T_days'] >= 0].copy()     # keep 0DTE (today's expiry)
    # Clip T_years to a 1-hour floor so BS gamma stays finite for 0DTE options
    _T_MIN = 1 / (365 * 24)                     # ≈ 1 hour expressed in years
    data['T_years'] = (data['T_days'] / 365.0).clip(lower=_T_MIN)
    data['spot']    = S
    data['mid']     = data[['bid', 'ask']].mean(axis=1, skipna=True)
    data['flag']    = data['optionType'].map({'call': 'c', 'put': 'p'})
    data['iv']      = data['impliedVolatility'] / 100.0

    # No OI or DTE filter here — full dataset goes to CSV
    data = data[data['mid'].notna()].copy()
    data.dropna(subset=['iv'], inplace=True)
    data = data[data['iv'] > 0].copy()

    # ── IV consistency check (removes stale/erroneous quotes) ─────────────
    n_before = len(data)
    data = validate_iv(data)
    n_dropped = n_before - len(data)
    if n_dropped > 0:
        print(f'  IV filter: removed {n_dropped} anomalous contracts '
              f'({n_dropped/n_before*100:.1f}%)')

    # ── Dynamic dividend yield via put-call parity (per expiry) ───────────
    # Compute q for each expiry from ATM call-put pair before running BS.
    # Falls back to SPX_DIV_YIELD constant if parity fails.
    data['mid']  = data[['bid', 'ask']].mean(axis=1, skipna=True)   # ensure mid exists
    q_by_expiry  = compute_implied_div_yield(data, S, r)
    data['q_impl'] = data['expiry'].map(q_by_expiry).fillna(SPX_DIV_YIELD)
    q_implied_mean = data['q_impl'].mean()
    print(f'  Implied div yield: mean q = {q_implied_mean*100:.2f}% '
          f'(vs constant {SPX_DIV_YIELD*100:.1f}%)')

    notify(93, f'Computing Greeks for {ticker} ...')
    # Use per-contract implied dividend yield (fallback = SPX_DIV_YIELD)
    q_arr = data['q_impl'].to_numpy(dtype=float)

    delta_calc = np.zeros(len(data))
    gamma_calc = np.zeros(len(data))
    Kv  = data['strike'].to_numpy(dtype=float)
    Tv  = data['T_years'].to_numpy(dtype=float)
    sv  = data['iv'].to_numpy(dtype=float)
    flv = data['flag'].to_numpy()
    valid_bs = (Tv > 0) & (sv > 0) & np.isfinite(Kv) & np.isfinite(sv)
    if np.any(valid_bs):
        sqT_v  = np.sqrt(Tv[valid_bs])
        d1_v   = (np.log(S / Kv[valid_bs])
                  + (r - q_arr[valid_bs] + 0.5*sv[valid_bs]**2)*Tv[valid_bs]
                  ) / (sv[valid_bs] * sqT_v)
        disc_v = np.exp(-q_arr[valid_bs] * Tv[valid_bs])
        cdf_d1 = norm.cdf(d1_v)
        delta_calc[valid_bs] = np.where(
            flv[valid_bs] == 'c', disc_v * cdf_d1, disc_v * (cdf_d1 - 1.0))
        gamma_calc[valid_bs] = disc_v * norm.pdf(d1_v) / (S * sv[valid_bs] * sqT_v)

    delta_available = data['delta_barchart'].notna()
    data['delta'] = np.where(delta_available, data['delta_barchart'], delta_calc)
    data['gamma'] = gamma_calc
    data['dex']   = data['delta'] * data['openInterest'] * 100
    sign          = data['flag'].map({'c': 1, 'p': -1})
    data['gex']   = sign * data['gamma'] * data['openInterest'] * 100 * S

    # ── Charm, Vanna, Rho per contract ────────────────────────────────────
    charm_arr = np.zeros(len(data))
    vanna_arr = np.zeros(len(data))
    rho_arr   = np.zeros(len(data))
    if np.any(valid_bs):
        sqT   = np.sqrt(Tv[valid_bs])
        qv    = q_arr[valid_bs]
        d1v   = (np.log(S / Kv[valid_bs]) + (r - qv + 0.5*sv[valid_bs]**2)*Tv[valid_bs]) / (sv[valid_bs]*sqT)
        d2v   = d1v - sv[valid_bs] * sqT
        nd1   = norm.pdf(d1v)
        disc  = np.exp(-qv * Tv[valid_bs])
        # Charm
        inner = 2.0*(r - qv)*Tv[valid_bs] - d2v*sv[valid_bs]*sqT
        term  = disc * nd1 * inner / (2.0*Tv[valid_bs]*sv[valid_bs]*sqT)
        call_m = flv[valid_bs] == 'c'
        charm_arr[valid_bs] = np.where(call_m,
            -term + qv*disc*norm.cdf(d1v),
            -term - qv*disc*norm.cdf(-d1v))
        # Vanna
        vanna_arr[valid_bs] = -disc * nd1 * d2v / sv[valid_bs]
        # Rho (per 1bp)
        disc_r = np.exp(-r * Tv[valid_bs])
        rho_arr[valid_bs] = np.where(call_m,
            Kv[valid_bs]*Tv[valid_bs]*disc_r*norm.cdf(d2v)  / 100.0,
            -Kv[valid_bs]*Tv[valid_bs]*disc_r*norm.cdf(-d2v) / 100.0)

    data['charm']     = charm_arr
    data['vanna']     = vanna_arr
    data['rho']       = rho_arr
    data['charm_exp'] = data['charm'] * data['openInterest'] * 100 * S
    data['vanna_exp'] = data['vanna'] * data['openInterest'] * 100 * S
    data['rho_exp']   = data['rho']   * data['openInterest'] * 100   # $ per 1bp rate move
    data['gross_gex'] = data['gamma'] * data['openInterest'] * 100 * S

    notify(96, f'Computed Greeks — {len(data):,} contracts across {data["expiry"].nunique()} expiries')
    print(f'  Loaded {len(data):,} contracts across {data["expiry"].nunique()} expiries.')
    _cache_set(key, data, S)

    # Write the CSV cache OFF the critical path. A slow or locked disk
    # (OneDrive sync, antivirus scan, file open in Excel, network home dir)
    # must never freeze the dashboard — the CSV is only a cache for next time.
    notify(98, f'Caching {ticker} chain to disk in background ...')
    threading.Thread(
        target=_csv_save, args=(ticker, data.copy(), S), daemon=True
    ).start()
    notify(100, f'Ready — {len(data):,} contracts across {data["expiry"].nunique()} expiries')
    return data, S


def aggregate_by_strike(data: pd.DataFrame):
    """Roll up DEX, GEX, charm, vanna, gross_gex to per-strike totals (vectorised)."""
    calls  = (data[data['flag'] == 'c']
               .groupby('strike', sort=True)
               .agg(call_dex=('dex', 'sum'), call_gex=('gex', 'sum')))
    puts   = (data[data['flag'] == 'p']
               .groupby('strike', sort=True)
               .agg(put_dex=('dex', 'sum'), put_gex=('gex', 'sum')))
    # Extra greek exposures — guard if columns absent (old cached CSVs)
    agg_dict = dict(net_dex=('dex','sum'), net_gex=('gex','sum'),
                    total_oi=('openInterest','sum'))
    for col, alias in [('charm_exp','charm_exp'), ('vanna_exp','vanna_exp'),
                       ('gross_gex','gross_gex')]:
        if col in data.columns:
            agg_dict[alias] = (col, 'sum')
    totals = data.groupby('strike', sort=True).agg(**agg_dict)
    return (totals.join(calls, how='left')
                  .join(puts,  how='left')
                  .fillna(0)
                  .reset_index())


def aggregate_by_expiry(data: pd.DataFrame):
    """Roll up DEX and GEX to per-expiry totals."""
    return data.groupby(['expiry', 'T_days']).agg(
        net_dex = ('dex', 'sum'),
        net_gex = ('gex', 'sum'),
        total_oi = ('openInterest', 'sum'),
    ).reset_index().sort_values('T_days')


def apply_dashboard_filters(
    data: pd.DataFrame,
    delta_min=None,
    delta_max=None,
    dte_min=None,
    dte_max=None,
    oi_min=None,
    oi_max=None,
    option_flags=None,
):
    filtered = data.copy()

    if option_flags:
        filtered = filtered[filtered['flag'].isin(option_flags)]

    if delta_min is not None:
        filtered = filtered[filtered['delta'] >= delta_min]
    if delta_max is not None:
        filtered = filtered[filtered['delta'] <= delta_max]
    if dte_min is not None:
        filtered = filtered[filtered['T_days'] >= dte_min]
    if dte_max is not None:
        filtered = filtered[filtered['T_days'] <= dte_max]
    if oi_min is not None:
        filtered = filtered[filtered['openInterest'] >= oi_min]
    if oi_max is not None:
        filtered = filtered[filtered['openInterest'] <= oi_max]

    return filtered.copy()

print('Data functions ready.')

# ── Load initial data (lazy: do not fetch until run) ─────────────────────────────────

# Note: fetching at import time can be slow and requires network; keep commented by default.
# raw_data, spot_price = fetch_options_data(DEFAULT_TICKER)
# by_strike  = aggregate_by_strike(raw_data)
# by_expiry  = aggregate_by_expiry(raw_data)

# ── Range & Skew analytics ────────────────────────────────────────────────────

def compute_daily_range(raw_df: pd.DataFrame, spot: float) -> pd.DataFrame:
    """For each expiry, find options near Δ±0.30 and compute the implied
    1-day expected move: spot × IV_Δ30 × √(1/252).
    Returns one row per expiry sorted by DTE.
    """
    records = []
    for expiry, grp in raw_df.groupby('expiry'):
        T_days = int(grp['T_days'].iloc[0])
        calls = grp[(grp['flag'] == 'c')].dropna(subset=['delta', 'iv'])
        puts  = grp[(grp['flag'] == 'p')].dropna(subset=['delta', 'iv'])
        if calls.empty or puts.empty:
            continue
        best_call = calls.iloc[(calls['delta'] - 0.30).abs().argsort().iloc[:1]]
        best_put  = puts.iloc[(puts['delta']  + 0.30).abs().argsort().iloc[:1]]
        if best_call.empty or best_put.empty:
            continue
        iv_d30 = (float(best_call['iv'].iloc[0]) + float(best_put['iv'].iloc[0])) / 2
        if iv_d30 <= 0:
            continue
        daily_move = spot * iv_d30 * np.sqrt(1 / 252)
        records.append({
            'expiry':         expiry,
            'T_days':         T_days,
            'iv_d30':         iv_d30,
            'daily_move_pts': daily_move,
            'daily_move_pct': (daily_move / spot) * 100,
            'upper':          spot + daily_move,
            'lower':          spot - daily_move,
            'call_strike':    int(best_call['strike'].iloc[0]),
            'put_strike':     int(best_put['strike'].iloc[0]),
            'call_delta':     round(float(best_call['delta'].iloc[0]), 3),
            'put_delta':      round(float(best_put['delta'].iloc[0]),  3),
        })
    return pd.DataFrame(records).sort_values('T_days').reset_index(drop=True)


def get_put_skew_by_expiry(raw_df: pd.DataFrame) -> dict:
    """Return {expiry: DataFrame(strike, iv_pct, moneyness, delta, T_days)}
    for put options only, sorted by strike ascending.
    """
    spot = float(raw_df['spot'].iloc[0])
    result = {}
    puts = raw_df[raw_df['flag'] == 'p'].dropna(subset=['iv', 'strike'])
    for expiry, grp in puts.groupby('expiry'):
        df = (grp[['strike', 'iv', 'delta', 'T_days']]
              .sort_values('strike')
              .drop_duplicates('strike')
              .reset_index(drop=True))
        df['moneyness'] = (df['strike'] / spot - 1) * 100
        df['iv_pct']    = df['iv'] * 100
        result[expiry]  = df
    return result


def get_put_monitor(raw_df: pd.DataFrame, spot: float,
                    target_deltas: list = None) -> pd.DataFrame:
    """For monthly and quarterly expirations, find the put closest to each
    target delta and return its mid price (premium in pts and % of spot) and IV.

    Monthly:    expirationType == 'monthly'
    Quarterly:  monthly expiries in Mar / Jun / Sep / Dec
    """
    if target_deltas is None:
        target_deltas = [0.25, 0.10, 0.05]
    puts = raw_df[(raw_df['flag'] == 'p') &
                  (raw_df['expirationType'] == 'monthly')].copy()
    if puts.empty:
        return pd.DataFrame()

    puts['abs_delta'] = puts['delta'].abs()
    puts['exp_month'] = pd.to_datetime(puts['expiry']).dt.month
    puts['is_qtr']    = puts['exp_month'].isin([3, 6, 9, 12])

    records = []
    for expiry, grp in puts.groupby('expiry'):
        T_days  = int(grp['T_days'].iloc[0])
        is_qtr  = bool(grp['is_qtr'].iloc[0])
        exp_lbl = 'Quarterly' if is_qtr else 'Monthly'
        for delta_t in target_deltas:
            valid = grp.dropna(subset=['delta', 'mid', 'iv'])
            if valid.empty:
                continue
            idx = (valid['abs_delta'] - delta_t).abs().idxmin()
            row = valid.loc[idx]
            records.append({
                'expiry':       expiry,
                'T_days':       T_days,
                'type':         exp_lbl,
                'delta_target': delta_t,
                'delta_actual': round(float(row['abs_delta']), 3),
                'strike':       int(row['strike']),
                'mid':          round(float(row['mid']), 2),
                'iv_pct':       round(float(row['iv']) * 100, 1),
                'mid_pct_spot': round(float(row['mid']) / spot * 100, 3),
            })
    return (pd.DataFrame(records)
              .sort_values(['T_days', 'delta_target'])
              .reset_index(drop=True))




def backtest_har_oos(ohlc: pd.DataFrame,
                     train_frac: float = 0.60,
                     trading_periods: int = 252) -> dict:
    """HAR-RV expanding-window out-of-sample validation.

    Fits HAR on the first train_frac of data, then forecasts one-step-ahead
    on the remaining (1 - train_frac) observations.  Returns RMSE, MAE,
    directional accuracy, and R² oos.
    """
    if ohlc is None or ohlc.empty or len(ohlc) < 60:
        return {}
    work = ohlc.set_index('date') if 'date' in ohlc.columns else ohlc
    rv   = rvol_yang_zhang(work, window=5, trading_periods=trading_periods).dropna() * 100
    rv_d = rv
    rv_w = rv.rolling(5).mean()
    rv_m = rv.rolling(22).mean()
    df   = pd.DataFrame({'rv': rv, 'd': rv_d, 'w': rv_w, 'm': rv_m}).dropna()
    if len(df) < 40:
        return {}

    n_train = max(30, int(len(df) * train_frac))
    actuals, forecasts = [], []
    for i in range(n_train, len(df) - 1):
        sub = df.iloc[:i]
        X   = np.column_stack([np.ones(len(sub)-1),
                                sub['d'].values[:-1],
                                sub['w'].values[:-1],
                                sub['m'].values[:-1]])
        y   = sub['rv'].values[1:]
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            last = df.iloc[i]
            fct  = float(np.clip(beta[0] + beta[1]*last['d']
                                  + beta[2]*last['w'] + beta[3]*last['m'], 1.0, 150.0))
            actuals.append(float(df.iloc[i+1]['rv']))
            forecasts.append(fct)
        except Exception:
            continue

    if len(actuals) < 10:
        return {}
    actuals   = np.array(actuals)
    forecasts = np.array(forecasts)
    errors    = actuals - forecasts
    rmse      = float(np.sqrt(np.mean(errors**2)))
    mae       = float(np.mean(np.abs(errors)))
    dir_acc   = float(np.mean(np.sign(np.diff(actuals))
                               == np.sign(np.diff(forecasts))))
    ss_res    = float(np.sum(errors**2))
    ss_tot    = float(np.sum((actuals - actuals.mean())**2))
    r2_oos    = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    bench_rmse = float(np.sqrt(np.mean((actuals - actuals.mean())**2)))
    return {
        'n_oos':      len(actuals),
        'rmse':       rmse,
        'mae':        mae,
        'dir_acc':    dir_acc,
        'r2_oos':     r2_oos,
        'vs_mean':    rmse < bench_rmse,
        'bench_rmse': bench_rmse,
    }


def backtest_vol_premium(rvol_df: pd.DataFrame,
                          vix_df: pd.DataFrame,
                          threshold: float = 5.0,
                          horizons: list = None) -> dict:
    """Vol premium capture analysis.

    For each historical day where VIX - RVol_Yang_Zhang > threshold (%),
    compute the average forward RVol over the next N trading days.
    Tests whether elevated vol premium predicts subsequent mean-reversion.

    Returns dict with results per horizon.
    """
    if horizons is None:
        horizons = [5, 10, 22]
    if rvol_df is None or rvol_df.empty or vix_df is None or vix_df.empty:
        return {}

    # Align on common dates
    yz   = rvol_df['Yang-Zhang'] if 'Yang-Zhang' in rvol_df.columns else pd.Series(dtype=float)
    if yz.empty:
        return {}
    vix_s = vix_df.set_index('date')['vix'] * 100
    combined = pd.DataFrame({'rv': yz, 'vix': vix_s}).dropna()
    if len(combined) < 30:
        return {}
    combined['premium'] = combined['vix'] - combined['rv']
    signal_days = combined.index[combined['premium'] >= threshold]
    if len(signal_days) < 5:
        return {'signal_days': 0}

    results = {'signal_days': len(signal_days), 'threshold_pct': threshold}
    for h in horizons:
        forward_rv, forward_vix = [], []
        for d in signal_days:
            future = combined.loc[d:].iloc[1:h+1]
            if len(future) >= h // 2:
                forward_rv.append(float(future['rv'].mean()))
                forward_vix.append(float(future['vix'].mean()))
        if forward_rv:
            avg_rv  = float(np.mean(forward_rv))
            avg_vix = float(np.mean(forward_vix))
            results[f'h{h}_avg_rv']      = avg_rv
            results[f'h{h}_avg_vix']     = avg_vix
            results[f'h{h}_rv_gt_start'] = float(np.mean(
                [r > float(combined.loc[d, 'rv']) for r, d in zip(forward_rv, signal_days)
                 if d in combined.index]))
    return results



# ══════════════════════════════════════════════════════════════════════════════
# Snapshot persistence & day-over-day analytics
# ══════════════════════════════════════════════════════════════════════════════

SNAPSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'snapshots')


def save_daily_snapshot(raw_df: pd.DataFrame, spot: float,
                         ticker: str = 'SPX') -> str:
    """Persist a compact per-strike snapshot of today's chain.

    Saves aggregated per-strike data (not the full chain) to keep files small
    (~50 KB vs ~5 MB).  One file per ticker per day:
        snapshots/SPX_2026-06-08.csv
    If GitHub Actions commits the snapshots/ folder, history accumulates in
    the repo and survives Streamlit Cloud restarts.
    Returns the path written, or '' on failure.
    """
    try:
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)
        from datetime import date as _d
        bs   = aggregate_by_strike(raw_df)
        meta = pd.DataFrame([{
            'strike': -1,                       # sentinel row for metadata
            'net_gex': float(raw_df['gex'].sum()),
            'net_dex': float(raw_df['dex'].sum()),
            'total_oi': float(raw_df['openInterest'].sum()),
            'call_gex': spot,                   # reuse column: spot price
            'put_gex': len(raw_df),             # reuse column: n contracts
        }])
        out  = pd.concat([meta, bs], ignore_index=True)
        path = os.path.join(SNAPSHOT_DIR,
                            f'{ticker.replace("$","")}_{_d.today().isoformat()}.csv')
        out.to_csv(path, index=False)
        print(f'  [snapshot] saved {path} ({len(bs)} strikes)')
        return path
    except Exception as e:
        print(f'  [snapshot] save failed: {e}')
        return ''


def load_previous_snapshot(ticker: str = 'SPX') -> tuple:
    """Load the most recent snapshot older than today.

    Returns (by_strike_df, meta_dict, snapshot_date_str) or (None, {}, '').
    """
    try:
        from datetime import date as _d
        clean = ticker.replace('$', '')
        if not os.path.isdir(SNAPSHOT_DIR):
            return None, {}, ''
        files = sorted(f for f in os.listdir(SNAPSHOT_DIR)
                       if f.startswith(clean + '_') and f.endswith('.csv'))
        today_name = f'{clean}_{_d.today().isoformat()}.csv'
        prior = [f for f in files if f < today_name]
        if not prior:
            return None, {}, ''
        path = os.path.join(SNAPSHOT_DIR, prior[-1])
        df   = pd.read_csv(path)
        meta_row = df[df['strike'] == -1]
        meta = {}
        if not meta_row.empty:
            meta = {
                'net_gex':     float(meta_row['net_gex'].iloc[0]),
                'net_dex':     float(meta_row['net_dex'].iloc[0]),
                'total_oi':    float(meta_row['total_oi'].iloc[0]),
                'spot':        float(meta_row['call_gex'].iloc[0]),
                'n_contracts': int(meta_row['put_gex'].iloc[0]),
            }
        bs = df[df['strike'] != -1].reset_index(drop=True)
        snap_date = prior[-1].replace(clean + '_', '').replace('.csv', '')
        return bs, meta, snap_date
    except Exception as e:
        print(f'  [snapshot] load failed: {e}')
        return None, {}, ''


def compute_dod_changes(raw_df: pd.DataFrame, spot: float,
                         ticker: str = 'SPX') -> dict:
    """Day-over-day changes vs the last saved snapshot.

    Returns dict with prev_date, d_net_gex, d_net_dex, d_total_oi, d_spot
    (absolute and %), or {} if no prior snapshot exists.
    """
    prev_bs, prev_meta, prev_date = load_previous_snapshot(ticker)
    if prev_bs is None or not prev_meta:
        return {}
    cur_gex = float(raw_df['gex'].sum())
    cur_dex = float(raw_df['dex'].sum())
    cur_oi  = float(raw_df['openInterest'].sum())
    out = {
        'prev_date':  prev_date,
        'd_net_gex':  cur_gex - prev_meta['net_gex'],
        'd_net_dex':  cur_dex - prev_meta['net_dex'],
        'd_total_oi': cur_oi  - prev_meta['total_oi'],
        'd_spot':     spot    - prev_meta['spot'],
        'prev_gex':   prev_meta['net_gex'],
        'prev_spot':  prev_meta['spot'],
    }
    out['d_gex_pct'] = (out['d_net_gex'] / abs(prev_meta['net_gex']) * 100
                         if prev_meta['net_gex'] else None)
    out['d_oi_pct']  = (out['d_total_oi'] / prev_meta['total_oi'] * 100
                         if prev_meta['total_oi'] else None)
    return out


def compute_gex_percentile(raw_df: pd.DataFrame, ticker: str = 'SPX') -> dict:
    """Current net GEX vs its own snapshot history.

    Returns {'percentile': 0-100, 'n_history': N} or {} if < 5 snapshots.
    """
    try:
        clean = ticker.replace('$', '')
        if not os.path.isdir(SNAPSHOT_DIR):
            return {}
        hist = []
        for f in sorted(os.listdir(SNAPSHOT_DIR)):
            if f.startswith(clean + '_') and f.endswith('.csv'):
                df = pd.read_csv(os.path.join(SNAPSHOT_DIR, f), nrows=2)
                m  = df[df['strike'] == -1]
                if not m.empty:
                    hist.append(float(m['net_gex'].iloc[0]))
        if len(hist) < 5:
            return {'n_history': len(hist)}
        cur  = float(raw_df['gex'].sum())
        pct  = float(np.mean([h <= cur for h in hist]) * 100)
        return {'percentile': pct, 'n_history': len(hist)}
    except Exception:
        return {}


def compute_flip_by_expiry(raw_df: pd.DataFrame, spot: float,
                            max_expiries: int = 6) -> pd.DataFrame:
    """GEX Flip level computed separately for each of the nearest expiries.

    The aggregate flip can hide the fact that the 0DTE wall and the monthly
    wall sit at different levels.  Returns DataFrame(expiry, T_days, flip,
    net_gex, regime) sorted by T_days.
    """
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()
    rows = []
    expiries = (raw_df.groupby('expiry')['T_days'].first()
                       .sort_values().head(max_expiries))
    for exp, tdays in expiries.items():
        sub = raw_df[raw_df['expiry'] == exp]
        bs  = aggregate_by_strike(sub)
        if bs.empty or len(bs) < 3:
            continue
        srt  = bs.sort_values('strike').reset_index(drop=True)
        sgn  = np.sign(srt['net_gex'].values)
        flips = np.where(np.diff(sgn) != 0)[0]
        flip = None
        if len(flips):
            i = flips[0]
            s1, g1 = float(srt['strike'].iloc[i]),   float(srt['net_gex'].iloc[i])
            s2, g2 = float(srt['strike'].iloc[i+1]), float(srt['net_gex'].iloc[i+1])
            flip = s1 - g1*(s2-s1)/(g2-g1) if g2 != g1 else (s1+s2)/2
        net = float(sub['gex'].sum())
        rows.append({
            'expiry':  exp,
            'T_days':  int(tdays),
            'flip':    flip,
            'net_gex': net,
            'regime':  'LONG γ' if net >= 0 else 'SHORT γ',
            'spot_vs_flip': (spot - flip) if flip else None,
        })
    return pd.DataFrame(rows)



def generate_morning_brief(dte0_metrics: dict, gex_analytics: dict,
                            alert_flags: list, spot: float,
                            ticker: str = '$SPX',
                            dod: dict = None) -> bytes:
    """One-page PDF morning brief: alert flags, key 0DTE metrics, day levels.

    Returns the PDF as bytes (for st.download_button).
    """
    from io import BytesIO
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib.colors import HexColor, white
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                     Table, TableStyle, HRFlowable)
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from datetime import datetime as _dt

    VIOLET   = HexColor('#6C63FF'); VIOLET_D = HexColor('#1E1B4B')
    GRAY_D   = HexColor('#374151'); GRAY_L   = HexColor('#E5E7EB')
    GRAY_BG  = HexColor('#F8F9FD')
    STATUS_C = {'RED': HexColor('#EF4444'), 'AMBER': HexColor('#F59E0B'),
                'GREEN': HexColor('#10B981'), 'GREY': HexColor('#9CA3AF')}

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=1.8*cm, rightMargin=1.8*cm,
                            topMargin=1.5*cm, bottomMargin=1.2*cm)
    W = A4[0] - 3.6*cm
    sT  = ParagraphStyle('t', fontName='Helvetica-Bold', fontSize=18,
                          textColor=VIOLET_D, alignment=TA_LEFT, spaceAfter=2)
    sS  = ParagraphStyle('s', fontName='Helvetica', fontSize=9,
                          textColor=HexColor('#6B7280'), spaceAfter=10)
    sH  = ParagraphStyle('h', fontName='Helvetica-Bold', fontSize=11,
                          textColor=VIOLET, spaceBefore=10, spaceAfter=4)
    sB  = ParagraphStyle('b', fontName='Helvetica', fontSize=9,
                          textColor=GRAY_D, leading=13)
    sC  = ParagraphStyle('c', fontName='Helvetica-Bold', fontSize=9,
                          textColor=white, alignment=TA_CENTER)

    story = [
        Paragraph(f'Morning Brief — {ticker}', sT),
        Paragraph(_dt.now().strftime('%A %d %B %Y · %H:%M'), sS),
        HRFlowable(width='100%', thickness=1.5, color=VIOLET, spaceAfter=8),
    ]

    # ── Alert flags ──
    story.append(Paragraph('Alert Flags', sH))
    cells, colors_row = [], []
    for fl in (alert_flags or []):
        cells.append(Paragraph(
            f"<b>{fl['name']}</b><br/>{fl['value']}", ParagraphStyle(
                'fc', fontName='Helvetica', fontSize=8, textColor=white,
                alignment=TA_CENTER, leading=11)))
        colors_row.append(STATUS_C.get(fl['status'], GRAY_L))
    if cells:
        t = Table([cells], colWidths=[W/len(cells)]*len(cells))
        style = [('TOPPADDING',(0,0),(-1,-1),6), ('BOTTOMPADDING',(0,0),(-1,-1),6),
                 ('LEFTPADDING',(0,0),(-1,-1),4), ('RIGHTPADDING',(0,0),(-1,-1),4)]
        for i, c in enumerate(colors_row):
            style.append(('BACKGROUND', (i,0), (i,0), c))
        t.setStyle(TableStyle(style))
        story.append(t)

    # ── Key levels ──
    story.append(Paragraph('Livelli del giorno', sH))
    m  = dte0_metrics or {}
    ga = gex_analytics or {}
    rows = [
        ['Spot', f"${spot:,.1f}", 'GEX Flip (0DTE)',
         f"${m['gex_flip']:,.0f}" if m.get('gex_flip') else '—'],
        ['Max Pain', f"${m['max_pain']:,.0f}" if m.get('max_pain') else '—',
         'Max Gamma Strike',
         f"${m['max_gex_strike']:,.0f}" if m.get('max_gex_strike') else '—'],
        ['Expected Move',
         f"±{m['exp_move_pts']:.0f} pts ({m['exp_move_pct']:.2f}%)"
         if m.get('exp_move_pts') else '—',
         'GEX Center of Mass',
         f"${ga['center_of_mass']:,.0f}" if ga.get('center_of_mass') else '—'],
        ['ATM IV 0DTE',
         f"{m['atm_iv']*100:.1f}%" if m.get('atm_iv') else '—',
         'P/C OI Ratio',
         f"{m['pc_ratio']:.2f}" if m.get('pc_ratio') else '—'],
        ['Total 0DTE GEX',
         f"{m['total_gex']/1e9:+.2f}B" if m.get('total_gex') is not None else '—',
         'HHI Concentrazione',
         f"{ga['hhi']:.4f}" if ga.get('hhi') else '—'],
    ]
    body_rows = [[Paragraph(f'<b>{a}</b>', sB), Paragraph(b, sB),
                   Paragraph(f'<b>{c}</b>', sB), Paragraph(d, sB)]
                  for a,b,c,d in rows]
    lt = Table(body_rows, colWidths=[W*0.27, W*0.23, W*0.27, W*0.23])
    lt.setStyle(TableStyle([
        ('BOX',(0,0),(-1,-1),0.5,GRAY_L), ('INNERGRID',(0,0),(-1,-1),0.3,GRAY_L),
        ('ROWBACKGROUNDS',(0,0),(-1,-1),[GRAY_BG, white]),
        ('TOPPADDING',(0,0),(-1,-1),5), ('BOTTOMPADDING',(0,0),(-1,-1),5),
        ('LEFTPADDING',(0,0),(-1,-1),6),
    ]))
    story.append(lt)

    # ── Day-over-day ──
    if dod:
        story.append(Paragraph('Variazioni vs ' + dod.get('prev_date',''), sH))
        story.append(Paragraph(
            f"Δ Net GEX: <b>{dod['d_net_gex']/1e9:+.2f}B</b>"
            + (f" ({dod['d_gex_pct']:+.1f}%)" if dod.get('d_gex_pct') else '')
            + f" &nbsp;·&nbsp; Δ OI: <b>{dod['d_total_oi']/1e3:+.0f}k</b>"
            + f" &nbsp;·&nbsp; Δ Spot: <b>{dod['d_spot']:+.1f} pts</b>", sB))

    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width='100%', thickness=0.5, color=GRAY_L))
    story.append(Paragraph(
        'Generato da DEX/GEX Analytics — uso interno', ParagraphStyle(
            'f', fontName='Helvetica-Oblique', fontSize=7,
            textColor=HexColor('#9CA3AF'), alignment=TA_CENTER, spaceBefore=4)))

    doc.build(story)
    return buf.getvalue()


def compute_max_pain(raw_df: pd.DataFrame, expiry: str = None) -> dict:
    """Max Pain per scadenza: strike che minimizza il totale payout agli option buyer.

    Per ogni candidato strike K, somma:
      Σ_{K' < K} (K-K') * call_OI[K']   (ITM call payout)
    + Σ_{K' > K} (K'-K) * put_OI[K']    (ITM put payout)
    Restituisce il K con payout minimo = Max Pain (i dealer "vincono" di più lì).
    """
    df = raw_df[raw_df['expiry'] == expiry].copy() if expiry else raw_df.copy()
    if df.empty:
        return {}
    results = {}
    for exp in df['expiry'].unique():
        sub   = df[df['expiry'] == exp]
        calls = sub[sub['flag']=='c'].groupby('strike')['openInterest'].sum()
        puts  = sub[sub['flag']=='p'].groupby('strike')['openInterest'].sum()
        strikes = sorted(set(calls.index) | set(puts.index))
        if len(strikes) < 3:
            continue
        pain = {}
        for K in strikes:
            c_pain = sum(max(K - k, 0) * oi for k, oi in calls.items())
            p_pain = sum(max(k - K, 0) * oi for k, oi in puts.items())
            pain[K] = c_pain + p_pain
        results[exp] = float(min(pain, key=pain.get))
    return results


def compute_pc_ratio(raw_df: pd.DataFrame, expiry: str = None) -> dict:
    """Put/Call OI ratio — proxy del sentiment.

    > 1.0  →  più put aperte (lean ribassista)
    < 1.0  →  più call aperte (lean rialzista)
    ~1.0   →  posizionamento neutro
    """
    df = raw_df[raw_df['expiry'] == expiry].copy() if expiry else raw_df.copy()
    if df.empty:
        return {'ratio': None, 'put_oi': 0, 'call_oi': 0}
    call_oi = float(df[df['flag']=='c']['openInterest'].sum())
    put_oi  = float(df[df['flag']=='p']['openInterest'].sum())
    ratio   = (put_oi / call_oi) if call_oi > 0 else None
    return {'ratio': ratio, 'put_oi': put_oi, 'call_oi': call_oi}


def compute_har_rv(ohlc: pd.DataFrame,
                   trading_periods: int = 252) -> dict:
    """HAR-RV (Corsi 2009) — 1-day ahead realized vol forecast.

    RV_{t+1} = α + β_d·RV_d + β_w·RV_w + β_m·RV_m
    RV_d = daily Yang-Zhang RVol (window=1 proxy: abs log-return × √252)
    RV_w = 5-day mean of RV_d
    RV_m = 22-day mean of RV_d

    Returns dict with forecast (%), R², and coefficients.
    """
    if ohlc is None or ohlc.empty or len(ohlc) < 30:
        return {}
    # Use 1-day rolling Yang-Zhang as the base RV series
    rv = rvol_yang_zhang(ohlc, window=5, trading_periods=trading_periods).dropna() * 100
    if len(rv) < 30:
        return {}
    rv_d = rv
    rv_w = rv.rolling(5).mean()
    rv_m = rv.rolling(22).mean()
    data = pd.DataFrame({'rv': rv, 'rv_d': rv_d, 'rv_w': rv_w, 'rv_m': rv_m}).dropna()
    if len(data) < 25:
        return {}
    X = np.column_stack([np.ones(len(data)-1),
                          data['rv_d'].values[:-1],
                          data['rv_w'].values[:-1],
                          data['rv_m'].values[:-1]])
    y = data['rv'].values[1:]
    try:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        alpha, b_d, b_w, b_m = beta
        last        = data.iloc[-1]
        forecast    = float(np.clip(alpha + b_d*last['rv_d']
                                    + b_w*last['rv_w'] + b_m*last['rv_m'], 1.0, 150.0))
        y_hat       = X @ beta
        residuals   = y - y_hat
        sigma_resid = float(np.std(residuals, ddof=4))  # ddof=4 for 4 params
        ss_res      = float(np.sum(residuals**2))
        ss_tot      = float(np.sum((y - y.mean())**2))
        r2          = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        return {
            'forecast':  forecast,
            'ci_68_lo':  float(np.clip(forecast - sigma_resid,   1.0, 150.0)),
            'ci_68_hi':  float(np.clip(forecast + sigma_resid,   1.0, 150.0)),
            'ci_95_lo':  float(np.clip(forecast - 2*sigma_resid, 1.0, 150.0)),
            'ci_95_hi':  float(np.clip(forecast + 2*sigma_resid, 1.0, 150.0)),
            'sigma_resid': sigma_resid,
            'r2':        r2,
            'alpha':     float(alpha), 'b_d': float(b_d),
            'b_w':       float(b_w),  'b_m': float(b_m),
        }
    except Exception:
        return {}


def detect_vol_regime(rvol_series: pd.Series,
                       lookback: int = 252) -> tuple:
    """Volatility regime via rolling z-score of Yang-Zhang RVol.

    Returns (regime_str, z_score) where regime_str ∈ {'HIGH', 'NORMAL', 'LOW'}.
    Thresholds: |z| > 1.5  →  HIGH / LOW;  else NORMAL.
    """
    if rvol_series is None or len(rvol_series) < 10:
        return 'NORMAL', 0.0
    hist    = rvol_series.tail(lookback).dropna()
    current = float(hist.iloc[-1])
    mean    = float(hist.mean())
    std     = float(hist.std())
    if std == 0:
        return 'NORMAL', 0.0
    z = (current - mean) / std
    regime = 'HIGH' if z > 1.5 else ('LOW' if z < -1.5 else 'NORMAL')
    return regime, float(z)


def compute_0dte_metrics(raw_df: pd.DataFrame, spot: float) -> dict:
    """Compute key metrics for the 0DTE (today's expiry) chain.

    Returns a dict with:
      atm_iv, exp_move_pts, exp_move_pct  — intraday expected range
      gex_flip                             — strike where net GEX crosses zero
      max_gex_strike                       — strike with highest |GEX|
      total_gex, total_dex                 — aggregate exposures
      by_strike                            — aggregated DataFrame for 0DTE
      raw                                  — raw 0DTE rows
      n_contracts                          — number of 0DTE contracts
    Returns empty dict if no 0DTE data is present.
    """
    dte0 = raw_df[raw_df['T_days'] == 0].copy()
    if dte0.empty:
        return {}

    calls = dte0[dte0['flag'] == 'c'].dropna(subset=['delta', 'iv'])
    puts  = dte0[dte0['flag'] == 'p'].dropna(subset=['delta', 'iv'])

    # ATM IV: mean of the call and put closest to Δ ±0.50
    atm_iv = None
    if not calls.empty and not puts.empty:
        c = calls.iloc[(calls['delta'] - 0.50).abs().argsort().iloc[:1]]
        p = puts.iloc[(puts['delta']  + 0.50).abs().argsort().iloc[:1]]
        atm_iv = (float(c['iv'].iloc[0]) + float(p['iv'].iloc[0])) / 2

    bs_0dte = aggregate_by_strike(dte0)

    # GEX flip: interpolated strike where net_gex crosses zero
    gex_flip = None
    if not bs_0dte.empty and len(bs_0dte) > 1:
        srt = bs_0dte.sort_values('strike').reset_index(drop=True)
        sgn = np.sign(srt['net_gex'].values)
        flips = np.where(np.diff(sgn) != 0)[0]
        if len(flips):
            i  = flips[0]
            s1, g1 = float(srt['strike'].iloc[i]),   float(srt['net_gex'].iloc[i])
            s2, g2 = float(srt['strike'].iloc[i+1]), float(srt['net_gex'].iloc[i+1])
            gex_flip = s1 - g1 * (s2 - s1) / (g2 - g1) if g2 != g1 else (s1 + s2) / 2

    max_gex_strike = None
    if not bs_0dte.empty:
        max_gex_strike = float(
            bs_0dte.loc[bs_0dte['net_gex'].abs().idxmax(), 'strike']
        )

    exp_move_pts = exp_move_pct = None
    if atm_iv is not None and atm_iv > 0:
        exp_move_pts = spot * atm_iv * np.sqrt(1 / 252)
        exp_move_pct = atm_iv / np.sqrt(252) * 100

    max_pain_map = compute_max_pain(dte0)
    max_pain_0dte = max_pain_map.get(dte0['expiry'].iloc[0]) if not dte0.empty else None
    pc_0dte = compute_pc_ratio(dte0)

    # Charm and Vanna exposure (requires charm_exp / vanna_exp columns from fetch)
    charm_total = float(dte0['charm_exp'].sum()) if 'charm_exp' in dte0.columns else None
    vanna_total = float(dte0['vanna_exp'].sum()) if 'vanna_exp' in dte0.columns else None
    gross_gex_0 = float(dte0['gross_gex'].sum()) if 'gross_gex' in dte0.columns else None

    return {
        'atm_iv':         atm_iv,
        'exp_move_pts':   exp_move_pts,
        'exp_move_pct':   exp_move_pct,
        'gex_flip':       gex_flip,
        'max_gex_strike': max_gex_strike,
        'total_gex':      float(dte0['gex'].sum()),
        'total_dex':      float(dte0['dex'].sum()),
        'gross_gex':      gross_gex_0,
        'charm_exp':      charm_total,       # $·yr⁻¹ — daily delta bleed if /252
        'vanna_exp':      vanna_total,       # $ per 1pt vol move
        'max_pain':       max_pain_0dte,
        'pc_ratio':       pc_0dte.get('ratio'),
        'put_oi':         pc_0dte.get('put_oi', 0),
        'call_oi':        pc_0dte.get('call_oi', 0),
        'n_contracts':    len(dte0),
        'by_strike':      bs_0dte,
        'raw':            dte0,
    }


def delta_strike_bounds(raw_df: pd.DataFrame,
                        delta_lo: float = -0.20,
                        delta_hi: float = 0.20):
    """Derive (lo_strike, hi_strike) for the GEX/DEX chart window.

    Maps delta values to strikes using the nearest expiry:
    - delta_lo in (-0.50, 0) → OTM put at that delta → strike below spot
    - delta_lo ≤ -0.50       → ATM/ITM territory: use the minimum available
                                strike (widest range to the downside)
    - delta_hi in (0, 0.50)  → OTM call at that delta → strike above spot
    - delta_hi ≥  0.50       → ATM/ITM territory: use the maximum available
                                strike (widest range to the upside)

    This prevents the inversion bug where ITM deltas (|Δ|>0.5) would produce
    lo_strike > hi_strike and an empty chart.
    """
    if raw_df is None or raw_df.empty:
        return None, None
    try:
        min_dte = int(raw_df['T_days'].min())
        near    = raw_df[raw_df['T_days'] == min_dte].dropna(subset=['delta', 'strike'])
        if near.empty:
            return None, None

        puts  = near[near['flag'] == 'p']
        calls = near[near['flag'] == 'c']

        lo_strike = None
        if not puts.empty:
            if delta_lo <= -0.50:
                # At or beyond ATM: show all the way to the lowest available strike
                lo_strike = float(puts['strike'].min())
            else:
                # OTM region (-0.499 to 0): find the put with closest delta
                lo_strike = float(
                    puts.loc[(puts['delta'] - delta_lo).abs().idxmin(), 'strike']
                )

        hi_strike = None
        if not calls.empty:
            if delta_hi >= 0.50:
                # At or beyond ATM: show all the way to the highest available strike
                hi_strike = float(calls['strike'].max())
            else:
                # OTM region (0 to 0.499): find the call with closest delta
                hi_strike = float(
                    calls.loc[(calls['delta'] - delta_hi).abs().idxmin(), 'strike']
                )

        # Final safety: ensure lo ≤ hi regardless of edge cases
        if lo_strike is not None and hi_strike is not None:
            lo_strike, hi_strike = min(lo_strike, hi_strike), max(lo_strike, hi_strike)

        return lo_strike, hi_strike
    except Exception:
        return None, None

# ── Chart builders ────────────────────────────────────────────────────

DARK_BG    = '#F8F9FD'   # page background (very light gray)
CARD_BG    = '#FFFFFF'   # card background (white)
ACCENT_GRN = '#10B981'   # emerald green  (positive / calls)
ACCENT_RED = '#EF4444'   # red            (negative / puts)
ACCENT_BLU = '#6C63FF'   # violet         (primary accent / DEX)
ACCENT_YLW = '#F59E0B'   # amber          (spot / neutral)
ACCENT_PRP = '#8B5CF6'   # purple         (secondary / skew)
GRID_COL   = '#E5E7EB'   # light gray grid
TEXT_COL   = '#374151'   # dark gray text
TEXT_SEC   = '#9CA3AF'   # secondary gray


def base_layout(title='', height=420):
    return dict(
        title=dict(text=title,
                   font=dict(color=TEXT_COL, size=13,
                             family='Inter, system-ui, -apple-system, sans-serif')),
        paper_bgcolor='rgba(0,0,0,0)',   # transparent → CSS card shows through
        plot_bgcolor='#F9FAFB',
        font=dict(color=TEXT_COL,
                  family='Inter, system-ui, -apple-system, sans-serif'),
        height=height,
        xaxis=dict(gridcolor=GRID_COL, zerolinecolor=GRID_COL,
                   tickfont=dict(color=TEXT_SEC, size=11)),
        yaxis=dict(gridcolor=GRID_COL, zerolinecolor=GRID_COL,
                   tickfont=dict(color=TEXT_SEC, size=11)),
        margin=dict(l=50, r=20, t=45, b=40),
        legend=dict(bgcolor='rgba(0,0,0,0)',
                    font=dict(size=11, color=TEXT_COL)),
    )



# ── Alert flag levels ──────────────────────────────────────────────────────────
_AL_RED   = 'RED'
_AL_AMBER = 'AMBER'
_AL_GREEN = 'GREEN'
_AL_GREY  = 'GREY'

def build_alert_flags(raw_df, spot, dte0_metrics=None,
                      rvol_df=None, vix_hist=None) -> list:
    """Build a structured list of alert flags for the dashboard alert panel.

    Each alert is a dict:
      name, status (RED|AMBER|GREEN|GREY), value, threshold, detail
    """
    flags = []

    # 1. Gamma Regime — spot vs 0DTE GEX Flip (SOLO scadenza di oggi —
    #    diverso dal "GEX Regime" della tab GEX/DEX, che usa tutte le
    #    scadenze nel range selezionato dallo slider)
    flip = (dte0_metrics or {}).get('gex_flip')
    if flip and spot:
        dist_pct = (spot - flip) / spot * 100
        if spot < flip:
            flags.append(dict(name='Gamma Regime (0DTE)', status=_AL_RED,
                value=f'SHORT', threshold=f'Flip ${flip:,.0f}',
                detail=f'Spot {dist_pct:+.2f}% dal flip 0DTE → amplificante oggi'))
        elif abs(dist_pct) < 0.5:
            flags.append(dict(name='Gamma Regime (0DTE)', status=_AL_AMBER,
                value='NEAR FLIP', threshold=f'Flip ${flip:,.0f}',
                detail=f'Zona transizione (Δ={abs(spot-flip):.0f}pt)'))
        else:
            flags.append(dict(name='Gamma Regime (0DTE)', status=_AL_GREEN,
                value='LONG γ', threshold=f'Flip ${flip:,.0f}',
                detail=f'Spot {dist_pct:+.2f}% sopra flip 0DTE → stabilizzante oggi. '
                       f'Nota: misura solo la scadenza odierna — può differire dal '
                       f'"GEX Regime" in tab GEX/DEX, che aggrega tutte le scadenze.'))
    else:
        flags.append(dict(name='Gamma Regime (0DTE)', status=_AL_GREY,
            value='N/D', threshold='—', detail='Carica 0DTE'))

    # 2. P/C Ratio
    pc = (dte0_metrics or {}).get('pc_ratio')
    if pc is not None:
        if pc > 1.3:
            st_pc = _AL_RED
        elif pc > 1.1 or pc < 0.8:
            st_pc = _AL_AMBER
        else:
            st_pc = _AL_GREEN
        lean = 'PUT-heavy' if pc > 1.1 else ('CALL-heavy' if pc < 0.9 else 'Neutro')
        flags.append(dict(name='P/C OI Ratio', status=st_pc,
            value=f'{pc:.2f}', threshold='1.0 = neutro',
            detail=lean))
    else:
        flags.append(dict(name='P/C OI Ratio', status=_AL_GREY,
            value='N/D', threshold='—', detail='Carica 0DTE'))

    # 3. Vol Premium (VIX − RVol)
    last_vix = None
    last_rv  = None
    if vix_hist is not None and isinstance(vix_hist, pd.DataFrame) and not vix_hist.empty:
        last_vix = float(vix_hist['vix'].iloc[-1] * 100)
    if rvol_df is not None and isinstance(rvol_df, pd.DataFrame) and not rvol_df.empty:
        col = 'Yang-Zhang' if 'Yang-Zhang' in rvol_df.columns else rvol_df.columns[0]
        last_rv = float(rvol_df[col].iloc[-1])
    if last_vix and last_rv:
        prem = last_vix - last_rv
        if prem > 8:
            st_vp = _AL_GREEN   # verde = buono per vol seller
        elif prem > 3:
            st_vp = _AL_AMBER
        elif prem < -3:
            st_vp = _AL_RED     # vol selling in perdita
        else:
            st_vp = _AL_AMBER
        flags.append(dict(name='Vol Premium', status=st_vp,
            value=f'{prem:+.1f}%',
            threshold='VIX − RVol(126d)',
            detail=f'VIX {last_vix:.1f}% / RVol {last_rv:.1f}%'))
    else:
        flags.append(dict(name='Vol Premium', status=_AL_GREY,
            value='N/D', threshold='—', detail='Carica RVol'))

    # 4. ATM IV 0DTE
    atm_iv = (dte0_metrics or {}).get('atm_iv')
    if atm_iv:
        iv_pct = atm_iv * 100
        if iv_pct > 30:
            st_iv = _AL_RED
        elif iv_pct > 20:
            st_iv = _AL_AMBER
        else:
            st_iv = _AL_GREEN
        flags.append(dict(name='0DTE ATM IV', status=st_iv,
            value=f'{iv_pct:.1f}%',
            threshold='<20% normale',
            detail='IV implicita scadenza odierna'))
    else:
        flags.append(dict(name='0DTE ATM IV', status=_AL_GREY,
            value='N/D', threshold='—', detail='Nessuna 0DTE'))

    # 5. Expected Move vs 0DTE IV
    em = (dte0_metrics or {}).get('exp_move_pct')
    if em:
        if em > 2.0:
            st_em = _AL_RED
        elif em > 1.2:
            st_em = _AL_AMBER
        else:
            st_em = _AL_GREEN
        flags.append(dict(name='Expected Move', status=st_em,
            value=f'±{em:.2f}%',
            threshold='<1.2% bassa', detail='Movimento giornaliero atteso'))
    else:
        flags.append(dict(name='Expected Move', status=_AL_GREY,
            value='N/D', threshold='—', detail='Nessuna 0DTE'))

    # 6. Vanna Exposure (systemic vol→spot risk)
    vanna_exp = (dte0_metrics or {}).get('vanna_exp')
    if vanna_exp is not None:
        vanna_m = vanna_exp / 1e9
        if abs(vanna_m) > 5:
            st_va = _AL_RED
        elif abs(vanna_m) > 2:
            st_va = _AL_AMBER
        else:
            st_va = _AL_GREEN
        direction = 'Dealer COMPRA su spike vol' if vanna_m < 0 else 'Dealer VENDE su spike vol'
        flags.append(dict(name='Vanna Exp (0DTE)', status=st_va,
            value=f'${vanna_m:+.1f}B',
            threshold='|>$2B| = rilevante',
            detail=direction))
    else:
        flags.append(dict(name='Vanna Exp (0DTE)', status=_AL_GREY,
            value='N/D', threshold='—', detail='Nessuna 0DTE'))

    return flags



def bs_speed(S, K, T, r, sigma, q=SPX_DIV_YIELD):
    """Speed = dGamma/dSpot = third derivative of option value.

    Measures how quickly gamma changes as the underlying moves.
    High speed near a strike = GEX profile will shift rapidly with spot.
    Units: gamma per unit of spot movement.
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, _ = _bs_d1d2(S, K, T, r, sigma, q)
    gamma  = bs_gamma(S, K, T, r, sigma, q)
    return float(-gamma / S * (d1 / (sigma * np.sqrt(T)) + 1.0))


def compute_gex_profile(raw_df: pd.DataFrame,
                         spot: float,
                         shifts: list = None,
                         r: float = RISK_FREE_RATE) -> pd.DataFrame:
    """Conditional GEX at different spot levels (keeping IV and OI fixed).

    Shows how the net GEX regime changes as the underlying moves.
    Essential for knowing where the next flip will occur if market moves.

    Parameters
    ----------
    raw_df  : full options chain DataFrame
    spot    : current spot price
    shifts  : list of fractional spot shifts, e.g. [-0.03, -0.02, ..., 0.03]
    r       : risk-free rate

    Returns
    -------
    DataFrame with columns: spot_level, net_gex, gross_gex, regime
    """
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()
    if shifts is None:
        shifts = [-0.05, -0.04, -0.03, -0.02, -0.01, 0.0,
                   0.01,  0.02,  0.03,  0.04,  0.05]

    Kv   = raw_df['strike'].to_numpy(dtype=float)
    Tv   = raw_df['T_years'].to_numpy(dtype=float)
    sv   = raw_df['iv'].to_numpy(dtype=float)
    flv  = raw_df['flag'].to_numpy()
    oiv  = raw_df['openInterest'].to_numpy(dtype=float)
    qv   = raw_df.get('q_impl', pd.Series(SPX_DIV_YIELD, index=raw_df.index)
                       ).to_numpy(dtype=float)
    sign = np.where(flv == 'c', 1.0, -1.0)

    valid = (Tv > 0) & (sv > 0) & np.isfinite(Kv) & np.isfinite(sv)
    rows  = []
    for sh in shifts:
        S_new  = spot * (1.0 + sh)
        gamma_arr = np.zeros(len(raw_df))
        if np.any(valid):
            sqT  = np.sqrt(Tv[valid])
            d1   = (np.log(S_new / Kv[valid])
                    + (r - qv[valid] + 0.5*sv[valid]**2)*Tv[valid]
                    ) / (sv[valid]*sqT)
            disc = np.exp(-qv[valid]*Tv[valid])
            gamma_arr[valid] = disc * norm.pdf(d1) / (S_new * sv[valid] * sqT)

        gex_arr   = sign * gamma_arr * oiv * 100 * S_new
        net_gex   = float(gex_arr.sum())
        gross_gex = float(np.abs(gex_arr).sum())
        rows.append({
            'shift_pct':  sh * 100,
            'spot_level': S_new,
            'net_gex':    net_gex,
            'gross_gex':  gross_gex,
            'regime':     'LONG γ' if net_gex >= 0 else 'SHORT γ',
        })
    return pd.DataFrame(rows)


def compute_gex_analytics(raw_df: pd.DataFrame,
                            by_strike_df: pd.DataFrame,
                            spot: float) -> dict:
    """Advanced GEX metrics beyond net/gross totals.

    Returns
    -------
    dict with:
      center_of_mass   : strike where |GEX| is gravitationally centred ($ weighted)
      hhi              : Herfindahl-Hirschman Index of GEX concentration [0,1]
                         0 = perfectly distributed, 1 = all GEX at one strike
      flip_zone_lo     : lower bound of the flip uncertainty zone (±sigma of zero-crossing)
      flip_zone_hi     : upper bound
      impact_1pct      : $ delta hedging required for a 1% spot move
      impact_5pct      : $ delta hedging required for a 5% spot move
      top3_strikes     : the three strikes with highest |GEX| (pinning candidates)
    """
    if by_strike_df is None or by_strike_df.empty:
        return {}

    bs = by_strike_df.copy()
    total_gross = float(bs['gross_gex'].sum()) if 'gross_gex' in bs.columns                   else float(bs['net_gex'].abs().sum())
    if total_gross == 0:
        return {}

    # ── Centre of mass (GEX-weighted average strike) ──────────────────────
    weights = bs['net_gex'].abs()
    com     = float((bs['strike'] * weights).sum() / weights.sum()) if weights.sum() > 0 else spot

    # ── HHI — concentration index ─────────────────────────────────────────
    shares = (bs['net_gex'].abs() / total_gross) ** 2
    hhi    = float(shares.sum())    # 0 = distributed, 1 = concentrated

    # ── Flip zone — strikes where |net_gex| < 10% of peak (ambiguous zone) ─
    peak   = float(bs['net_gex'].abs().max())
    fuzzy  = bs[bs['net_gex'].abs() < 0.10 * peak]['strike']
    flip_zone_lo = float(fuzzy.min()) if not fuzzy.empty else spot
    flip_zone_hi = float(fuzzy.max()) if not fuzzy.empty else spot

    # ── Dollar impact per move ────────────────────────────────────────────
    net_gex_total = float(bs['net_gex'].sum())
    impact_1pct   = abs(net_gex_total * 0.01)   # $ of delta to hedge per 1%
    impact_5pct   = abs(net_gex_total * 0.05)   # $ of delta to hedge per 5%

    # ── Gamma flip (full chain) + flip-based regime ───────────────────────
    # The regime is set by where SPOT sits relative to the gamma flip, BUT a
    # genuine flip only exists if net GEX is meaningfully positive on one side
    # and negative on the other. In strong SHORT-gamma markets the per-strike
    # GEX is negative at essentially every strike (no real flip) — in that
    # case any "zero crossing" is spurious noise far from spot, so we must NOT
    # invent a flip. We therefore decide the regime primarily from the sign of
    # net GEX in the NEIGHBOURHOOD of spot (the strikes that actually matter
    # for dealer hedging right now), and only report a gamma_flip when there
    # is a real sign change near that neighbourhood.
    gamma_flip = None
    regime     = None
    bs_sorted  = bs.sort_values('strike').reset_index(drop=True)

    near_gex_sign = None
    if spot and 'net_gex' in bs_sorted.columns and not bs_sorted.empty:
        # window: ±3% around spot (the operative zone)
        band = bs_sorted[(bs_sorted['strike'] >= spot * 0.97) &
                         (bs_sorted['strike'] <= spot * 1.03)]
        if band.empty:
            band = bs_sorted
        near_sum = float(band['net_gex'].sum())
        near_gex_sign = 1.0 if near_sum >= 0 else -1.0

    # Look for a genuine flip: a sign change in per-strike net GEX where BOTH
    # sides carry real magnitude (not isolated noise). Require the positive
    # side to be a non-trivial fraction of total gross GEX.
    if 'net_gex' in bs_sorted.columns and len(bs_sorted) > 1 and spot:
        strikes = bs_sorted['strike'].to_numpy(dtype=float)
        ng      = bs_sorted['net_gex'].to_numpy(dtype=float)
        gross   = float(np.abs(ng).sum()) or 1.0
        pos_frac = float(ng[ng > 0].sum()) / gross    # share that is positive
        neg_frac = float(-ng[ng < 0].sum()) / gross   # share that is negative
        # A real flip needs both sides to carry weight (≥15% each); otherwise
        # the market is one-sided (pure LONG or pure SHORT gamma) and no flip.
        if pos_frac >= 0.15 and neg_frac >= 0.15:
            sgn = np.sign(ng)
            cross_idx = np.where(np.diff(sgn) != 0)[0]
            crossings = []
            for i in cross_idx:
                if ng[i+1] != ng[i]:
                    x = strikes[i] - ng[i] * (strikes[i+1] - strikes[i]) / (ng[i+1] - ng[i])
                else:
                    x = (strikes[i] + strikes[i+1]) / 2
                crossings.append(x)
            if crossings:
                gamma_flip = min(crossings, key=lambda x: abs(x - spot))

    if gamma_flip is not None and spot:
        regime = 'LONG γ' if spot >= gamma_flip else 'SHORT γ'
    elif near_gex_sign is not None:
        # No genuine flip → one-sided market → regime = sign of near-spot GEX
        regime = 'LONG γ' if near_gex_sign > 0 else 'SHORT γ'
    else:
        regime = 'LONG γ' if net_gex_total >= 0 else 'SHORT γ'

    # ── Top 3 pinning candidates ──────────────────────────────────────────
    top3 = (bs.nlargest(3, 'net_gex', keep='all')['strike']
              .tolist() if len(bs) >= 3 else bs['strike'].tolist())

    return {
        'center_of_mass': com,
        'hhi':            hhi,
        'flip_zone_lo':   flip_zone_lo,
        'flip_zone_hi':   flip_zone_hi,
        'impact_1pct':    impact_1pct,
        'impact_5pct':    impact_5pct,
        'top3_strikes':   top3,
        'net_gex_total':  net_gex_total,
        'gamma_flip':     gamma_flip,
        'regime':         regime,
    }


def compute_0dte_gamma_schedule(dte0_raw: pd.DataFrame,
                                 spot: float,
                                 r: float = RISK_FREE_RATE,
                                 session_hours: float = 6.5) -> pd.DataFrame:
    """Intraday gamma decay schedule for 0DTE options.

    Computes the net GEX at each hour of the trading session, showing how
    0DTE gamma accelerates toward expiration (the "gamma burn" curve).
    The gamma explosion in the last 1-2 hours is the core mechanic of
    0DTE intraday squeezes and pinning.

    Returns DataFrame with columns: time_label, T_hours, net_gex, gross_gex
    """
    if dte0_raw is None or dte0_raw.empty:
        return pd.DataFrame()

    Kv  = dte0_raw['strike'].to_numpy(dtype=float)
    sv  = dte0_raw['iv'].to_numpy(dtype=float)
    oiv = dte0_raw['openInterest'].to_numpy(dtype=float)
    flv = dte0_raw['flag'].to_numpy()
    qv  = dte0_raw.get('q_impl',
            pd.Series(SPX_DIV_YIELD, index=dte0_raw.index)).to_numpy(dtype=float)
    sign = np.where(flv == 'c', 1.0, -1.0)

    # Hours remaining in session from 09:30 to 16:00
    checkpoints = np.arange(session_hours, 0.0, -1.0).tolist() + [0.25]
    labels      = [f'{int(9.5 + session_hours - h):02d}:00'
                    if h >= 1 else '15:45' for h in checkpoints]

    rows = []
    for h, lbl in zip(checkpoints, labels):
        T_yr   = max(h / 8760.0, 1e-6)   # hours → years
        valid  = (sv > 0) & np.isfinite(Kv) & np.isfinite(sv)
        g_arr  = np.zeros(len(dte0_raw))
        if np.any(valid):
            sqT   = np.sqrt(T_yr)
            d1    = (np.log(spot / Kv[valid])
                     + (r - qv[valid] + 0.5*sv[valid]**2)*T_yr
                     ) / (sv[valid]*sqT)
            disc  = np.exp(-qv[valid]*T_yr)
            g_arr[valid] = disc * norm.pdf(d1) / (spot * sv[valid] * sqT)
        gex_arr = sign * g_arr * oiv * 100 * spot
        rows.append({
            'time':      lbl,
            'T_hours':   h,
            'net_gex':   float(gex_arr.sum()),
            'gross_gex': float(np.abs(gex_arr).sum()),
        })
    return pd.DataFrame(rows)


def gex_bar_chart(by_strike_df, spot, ticker, window_pct=0.10,
                  strike_lo=None, strike_hi=None, key_levels=None):
    """GEX by strike bar chart.

    Renders ALL strikes in by_strike_df and uses xaxis.range to set the
    initial zoom window.  This way the chart is never empty regardless of
    the delta range selected: the user can always zoom out to see the full
    picture.

    key_levels (optional): dict of {label: strike} drawn as horizontal-ish
    reference lines with labels — e.g. {'Call Resistance': 7800,
    'Put Support': 7300, 'Gamma Flip': 7445}. Turns the chart into an
    annotated map of the operative levels.
    """
    import plotly.graph_objects as go

    # Determine the initial view window from delta bounds (or ±window_pct fallback)
    lo = strike_lo if (strike_lo is not None and np.isfinite(strike_lo)) else spot * (1 - window_pct)
    hi = strike_hi if (strike_hi is not None and np.isfinite(strike_hi)) else spot * (1 + window_pct)
    if lo >= hi:
        lo, hi = spot * (1 - window_pct), spot * (1 + window_pct)

    df     = by_strike_df.copy()
    colors = [ACCENT_GRN if v >= 0 else ACCENT_RED for v in df['net_gex']]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df['strike'], y=df['net_gex'] / 1e6,
        marker_color=colors,
        name='Net GEX',
    ))
    fig.add_vline(x=spot, line_color=ACCENT_YLW, line_dash='dash',
                  annotation_text=f' Spot ${spot:.1f}',
                  annotation_font_color=ACCENT_YLW)

    # Key levels as labelled vertical lines (strikes are on the x-axis here)
    if key_levels:
        _lvl_clr = {'Call Resistance': '#EF4444', 'Put Support': '#10B981',
                    'Gamma Flip': '#A78BFA', 'Pivot': '#A78BFA'}
        for _name, _lvl in key_levels.items():
            if _lvl is None:
                continue
            try:
                _lv = float(_lvl)
            except Exception:
                continue
            fig.add_vline(x=_lv, line_color=_lvl_clr.get(_name, '#9CA3AF'),
                          line_dash='dot', line_width=1.4,
                          annotation_text=f' {_name} {_lv:.0f}',
                          annotation_font_color=_lvl_clr.get(_name, '#9CA3AF'),
                          annotation_font_size=9)

    layout = base_layout(f'GEX by Strike — {ticker}')
    layout['xaxis'].update(range=[lo, hi])
    layout.update(yaxis_title='GEX ($M)', xaxis_title='Strike')
    fig.update_layout(**layout)
    return fig


def dex_bar_chart(by_strike_df, spot, ticker, window_pct=0.10,
                  strike_lo=None, strike_hi=None):
    """DEX by strike chart (call / put / net).

    Same approach as gex_bar_chart: all data rendered, xaxis.range for zoom.
    Net DEX line is always yellow (#EAB308) for easy visual separation from
    the green/red call-put bars and the blue/violet spot vline.
    """
    import plotly.graph_objects as go

    _NET_DEX_CLR = '#EAB308'   # clear yellow — distinct from amber spot line

    lo = strike_lo if (strike_lo is not None and np.isfinite(strike_lo)) else spot * (1 - window_pct)
    hi = strike_hi if (strike_hi is not None and np.isfinite(strike_hi)) else spot * (1 + window_pct)
    if lo >= hi:
        lo, hi = spot * (1 - window_pct), spot * (1 + window_pct)

    df = by_strike_df.copy()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df['strike'], y=df['call_dex'] / 1e6,
                          name='Call DEX', marker_color=ACCENT_GRN, opacity=0.80))
    fig.add_trace(go.Bar(x=df['strike'], y=df['put_dex'] / 1e6,
                          name='Put DEX', marker_color=ACCENT_RED, opacity=0.80))
    fig.add_trace(go.Scatter(
        x=df['strike'], y=df['net_dex'] / 1e6,
        mode='lines+markers', name='Net DEX',
        line=dict(color=_NET_DEX_CLR, width=2.5),
        marker=dict(size=4, color=_NET_DEX_CLR,
                    line=dict(color='white', width=1)),
    ))
    fig.add_vline(x=spot, line_color=ACCENT_BLU, line_dash='dash',
                  annotation_text=f' Spot ${spot:.1f}',
                  annotation_font_color=ACCENT_BLU)
    layout = base_layout(f'DEX by Strike — {ticker}')
    layout['xaxis'].update(range=[lo, hi])
    layout.update(barmode='relative', yaxis_title='DEX ($M)', xaxis_title='Strike',
                  legend=dict(orientation='h', y=1.08,
                              font=dict(color=TEXT_COL, size=11),
                              bgcolor='rgba(0,0,0,0)'))
    fig.update_layout(**layout)
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


def oi_heatmap(raw_df, spot, ticker, window_pct=0.12,
               strike_lo=None, strike_hi=None):
    import plotly.graph_objects as go

    lo = strike_lo if strike_lo is not None else spot * (1 - window_pct)
    hi = strike_hi if strike_hi is not None else spot * (1 + window_pct)
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
            title=dict(text='IV (%)', font=dict(color=TEXT_COL, family='Courier New')),
            tickfont=dict(color=TEXT_COL, family='Courier New'),
        ),
        opacity=0.92,
        # 2) Contour lines sovrapposte
        contours=dict(
            z=dict(
                show=True,
                usecolormap=True,
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
        x=[grid_strikes[atm_col]] * n_expiry,
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
                title=dict(text='Strike', font=dict(color=TEXT_COL)),
                gridcolor=GRID_COL,
                backgroundcolor=CARD_BG,
                tickfont=dict(color=TEXT_COL),
            ),
            yaxis=dict(
                title=dict(text='Expiry', font=dict(color=TEXT_COL)),
                gridcolor=GRID_COL,
                backgroundcolor=CARD_BG,
                tickfont=dict(color=TEXT_COL, size=9),
            ),
            zaxis=dict(
                title=dict(text='IV (%)', font=dict(color=TEXT_COL)),
                gridcolor=GRID_COL,
                backgroundcolor=CARD_BG,
                tickfont=dict(color=TEXT_COL),
            ),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.8)),
        ),
    )

    return fig



def vega_heatmap(raw_df, spot, ticker, window_pct=0.12):
    import plotly.graph_objects as go

    if 'vega_exposure' not in raw_df.columns:
        fig = go.Figure()
        fig.update_layout(**base_layout(f'Vega Exposure — {ticker} (vega not computed)'))
        return fig

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


def daily_range_chart(raw_df: pd.DataFrame, spot: float, ticker: str):
    """Bar chart of implied 1-day expected move (Δ0.30 IV) per expiry."""
    import plotly.graph_objects as go
    dr = compute_daily_range(raw_df, spot)
    fig = go.Figure()
    if dr.empty:
        fig.update_layout(**base_layout(f'Daily Range — {ticker} (no data)'))
        return fig

    fig.add_trace(go.Bar(
        x=dr['expiry'],
        y=dr['daily_move_pts'],
        name='Expected daily move (pts)',
        marker_color=ACCENT_BLU,
        opacity=0.85,
        text=[f'±{v:.1f} ({p:.2f}%)'
              for v, p in zip(dr['daily_move_pts'], dr['daily_move_pct'])],
        textposition='outside',
        textfont=dict(color=TEXT_COL, size=9),
        hovertemplate=(
            '<b>%{x}</b><br>'
            'Daily move: ±%{y:.1f} pts<br>'
            'IV Δ30: %{customdata[0]:.1f}%<br>'
            'Call strike: %{customdata[1]}<br>'
            'Put strike: %{customdata[2]}<br>'
            '<extra></extra>'
        ),
        customdata=list(zip(
            (dr['iv_d30'] * 100).round(1),
            dr['call_strike'],
            dr['put_strike'],
        )),
    ))

    # Horizontal reference from nearest expiry
    primary = float(dr.iloc[0]['daily_move_pts'])
    fig.add_hline(y=primary, line_dash='dot', line_color=ACCENT_YLW,
                  annotation_text=f'Near-term ±{primary:.1f}',
                  annotation_font_color=ACCENT_YLW, annotation_font_size=11)

    layout = base_layout(f'Implied Daily Range (Δ0.30) — {ticker}')
    layout.update(yaxis_title='Expected daily move (pts)', xaxis_title='Expiry',
                  showlegend=False)
    fig.update_layout(**layout)
    return fig


def iv_skew_overlay_chart(raw_df: pd.DataFrame, spot: float, ticker: str,
                           selected_expiries: list = None,
                           iv_levels: list = None):
    """Overlaid put IV skew curves per expiry.
    Horizontal dashed lines at iv_levels (e.g. [0.16, 0.18, 0.20]).
    x = moneyness (% from spot), y = IV %.
    """
    import plotly.graph_objects as go
    if iv_levels is None:
        iv_levels = [0.16, 0.18, 0.20]

    skew_data = get_put_skew_by_expiry(raw_df)
    fig = go.Figure()
    if not skew_data:
        fig.update_layout(**base_layout(f'IV Skew Overlay — {ticker} (no data)'))
        return fig

    if selected_expiries:
        skew_data = {k: v for k, v in skew_data.items() if k in selected_expiries}

    expiries_sorted = sorted(skew_data,
                             key=lambda e: int(skew_data[e]['T_days'].iloc[0]))
    n = max(len(expiries_sorted), 1)
    # Blue → green gradient along the term structure
    hues = [int(200 + 60 * i / max(n - 1, 1)) for i in range(n)]
    colors = [f'hsl({h},70%,55%)' for h in hues]

    for i, expiry in enumerate(expiries_sorted):
        df = skew_data[expiry]
        T  = int(df['T_days'].iloc[0])
        df = df[(df['moneyness'] >= -20) & (df['moneyness'] <= 5)]
        if df.empty:
            continue
        fig.add_trace(go.Scatter(
            x=df['moneyness'], y=df['iv_pct'],
            mode='lines', name=f'{expiry} ({T}d)',
            line=dict(color=colors[i], width=2),
            hovertemplate=(
                f'<b>{expiry} ({T}d)</b><br>'
                'Moneyness: %{x:.1f}%<br>'
                'IV: %{y:.1f}%<extra></extra>'
            ),
        ))

    lv_colors = {0.16: ACCENT_GRN, 0.18: ACCENT_YLW, 0.20: ACCENT_RED}
    for lv in iv_levels:
        lv_pct = lv * 100
        col = lv_colors.get(lv, '#aaaaaa')
        fig.add_hline(y=lv_pct, line_dash='dash', line_color=col, line_width=1.5,
                      annotation_text=f'IV {lv_pct:.0f}%', annotation_position='left',
                      annotation_font_color=col, annotation_font_size=11)

    fig.add_vline(x=0, line_dash='dot', line_color=TEXT_COL, line_width=1, opacity=0.4)

    layout = base_layout(f'Put IV Skew Overlay — {ticker}')
    layout.update(
        xaxis_title='Moneyness (% from spot)',
        yaxis_title='Implied Volatility (%)',
        legend=dict(orientation='v', x=1.01, y=1,
                    font=dict(color=TEXT_COL, size=10),
                    bgcolor='rgba(0,0,0,0)'),
    )
    fig.update_layout(**layout)
    return fig


def put_monitor_chart(raw_df: pd.DataFrame, spot: float, ticker: str):
    """Grouped bar chart of put mid price (% of spot) at Δ0.25/0.10/0.05
    across monthly and quarterly expirations.
    """
    import plotly.graph_objects as go
    pm = get_put_monitor(raw_df, spot)
    fig = go.Figure()
    if pm.empty:
        fig.update_layout(**base_layout(
            f'Put Monitor — {ticker} (no monthly expirations in loaded chain)'))
        return fig

    delta_styles = {
        0.25: (ACCENT_BLU,  'Δ 0.25'),
        0.10: (ACCENT_YLW,  'Δ 0.10'),
        0.05: (ACCENT_RED,  'Δ 0.05'),
    }
    for delta_t, (color, label) in delta_styles.items():
        sub = pm[pm['delta_target'] == delta_t]
        if sub.empty:
            continue
        fig.add_trace(go.Bar(
            x=sub.apply(lambda r: f"{r['expiry']}\n({r['type']})", axis=1),
            y=sub['mid_pct_spot'],
            name=label,
            marker_color=color,
            opacity=0.85,
            text=[f"{v:.3f}%" for v in sub['mid_pct_spot']],
            textposition='outside',
            textfont=dict(color=TEXT_COL, size=9),
            hovertemplate=(
                '<b>%{x}</b><br>'
                f'Target delta: {delta_t:.2f}<br>'
                'Premium: %{customdata[0]:.2f} pts (%{y:.3f}% of spot)<br>'
                'Strike: %{customdata[1]}<br>'
                'IV: %{customdata[2]:.1f}%<br>'
                'Actual Δ: %{customdata[3]:.3f}<br>'
                '<extra></extra>'
            ),
            customdata=list(zip(
                sub['mid'],
                sub['strike'],
                sub['iv_pct'],
                sub['delta_actual'],
            )),
        ))

    layout = base_layout(f'Put Premium Monitor — {ticker}  (Monthly / Quarterly)')
    layout.update(barmode='group', yaxis_title='Premium (% of spot)',
                  xaxis_title='Expiry',
                  legend=dict(orientation='h', y=1.08,
                              font=dict(color=TEXT_COL, size=11)))
    fig.update_layout(**layout)
    return fig


def price_vs_dex_chart(by_strike_df: pd.DataFrame, intraday_df,
                       spot: float, ticker: str,
                       strike_lo=None, strike_hi=None,
                       key_levels: dict = None):
    """Dual-panel chart with shared Y-axis (price / strike level).

    Left  — DEX exposure as horizontal bars per strike.
    Right — Today's intraday price (5-min bars as a line + fill) with
            light volume bars on a secondary Y-axis.
            Falls back to the spot price as a horizontal reference line
            when intraday data is unavailable (market closed, weekend,
            or Barchart endpoint restricted).

    key_levels (optional): dict like {'Gamma Flip': 7365, 'Peak GEX': 7300}
    drawn as horizontal dotted reference lines on the price panel — turns
    scattered numbers into an operational map.
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    from datetime import datetime as _dt

    has_intra = (intraday_df is not None
                 and isinstance(intraday_df, pd.DataFrame)
                 and not intraday_df.empty
                 and 'datetime' in intraday_df.columns)

    # Y-range: use delta bounds with ±4% fallback
    lo = strike_lo if (strike_lo is not None and np.isfinite(strike_lo)) else spot * 0.96
    hi = strike_hi if (strike_hi is not None and np.isfinite(strike_hi)) else spot * 1.04
    if lo >= hi:
        lo, hi = spot * 0.96, spot * 1.04

    # Extend to cover today's intraday range if available
    if has_intra:
        lo = min(lo, float(intraday_df['low'].min()))
        hi = max(hi, float(intraday_df['high'].max()))
    y_pad = (hi - lo) * 0.04

    # DEX panel data
    buf = (hi - lo) * 0.10
    dex = by_strike_df[
        (by_strike_df['strike'] >= lo - buf) &
        (by_strike_df['strike'] <= hi + buf)
    ].copy()
    if dex.empty:
        dex = by_strike_df.copy()

    if has_intra:
        # Detect if data is from today or a previous session
        _last_dt   = intraday_df['datetime'].iloc[-1]
        _first_dt  = intraday_df['datetime'].iloc[0]
        _is_today  = pd.Timestamp(_last_dt).date() == pd.Timestamp.now().date()
        _date_lbl  = 'Intraday' if _is_today else f'Last session {pd.Timestamp(_first_dt).strftime("%d %b")}'
        _n_bars    = len(intraday_df)

    fig = make_subplots(
        rows=1, cols=2,
        shared_yaxes=True,
        column_widths=[0.28, 0.72],
        horizontal_spacing=0.02,
        specs=[[{}, {'secondary_y': True}]],
        subplot_titles=['DEX by Strike',
                        f'{ticker}  ·  {_date_lbl if has_intra else "Intraday"}'],
    )

    # ── DEX horizontal bars ──────────────────────────────────────────────
    colors_dex = [ACCENT_GRN if v >= 0 else ACCENT_RED for v in dex['net_dex']]
    fig.add_trace(go.Bar(
        y=dex['strike'], x=dex['net_dex'] / 1e6,
        orientation='h', marker_color=colors_dex, marker_line_width=0,
        name='Net DEX', opacity=0.85,
        hovertemplate='Strike %{y}<br>Net DEX: %{x:.1f}M<extra></extra>',
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        y=dex['strike'], x=dex['call_dex'] / 1e6,
        orientation='h', marker_color=ACCENT_GRN, opacity=0.35,
        name='Call DEX', visible='legendonly',
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        y=dex['strike'], x=dex['put_dex'] / 1e6,
        orientation='h', marker_color=ACCENT_RED, opacity=0.35,
        name='Put DEX', visible='legendonly',
    ), row=1, col=1)
    fig.add_vline(x=0, line_color=GRID_COL, line_width=1, row=1, col=1)
    fig.add_hline(y=spot, line_color=ACCENT_YLW, line_dash='dash',
                  line_width=1.5, row=1, col=1)

    # Key levels overlay — operational map (gamma flip, peak GEX, max pain...)
    if key_levels:
        _lvl_colors = {'Gamma Flip': '#10B981', 'Peak GEX': '#A78BFA',
                       'Max Pain': '#F472B6'}
        for _name, _lvl in key_levels.items():
            if _lvl is None:
                continue
            try:
                _lv = float(_lvl)
            except Exception:
                continue
            fig.add_hline(y=_lv, line_color=_lvl_colors.get(_name, '#9CA3AF'),
                          line_dash='dot', line_width=1.3,
                          annotation_text=f"{_name} {_lv:,.0f}",
                          annotation_position='left',
                          annotation_font_size=9, row=1, col=1)

    # ── Intraday price panel ─────────────────────────────────────────────
    if has_intra:
        x_times    = intraday_df['datetime']
        close      = intraday_df['close']
        volume     = intraday_df['volume']
        day_hi     = float(intraday_df['high'].max())
        day_lo     = float(intraday_df['low'].min())
        last_close = float(close.iloc[-1])
        last_time  = x_times.iloc[-1]

        # Volume bars (secondary Y, very light)
        fig.add_trace(go.Bar(
            x=x_times, y=volume / 1e6,
            name='Volume (M)', marker_color='rgba(108,99,255,0.10)',
            marker_line_width=0, showlegend=False,
        ), row=1, col=2, secondary_y=True)

        # Price line + fill
        fig.add_trace(go.Scatter(
            x=x_times, y=close, mode='lines',
            name=ticker, line=dict(color=ACCENT_BLU, width=2.5),
            fill='tozeroy', fillcolor='rgba(108,99,255,0.07)',
            hovertemplate='%{x|%H:%M}  $%{y:,.2f}<extra></extra>',
        ), row=1, col=2, secondary_y=False)

        # Current price marker
        fig.add_trace(go.Scatter(
            x=[last_time], y=[last_close],
            mode='markers+text',
            marker=dict(color=ACCENT_BLU, size=10,
                        line=dict(color='white', width=2)),
            text=[f'  ${last_close:,.2f}'],
            textposition='middle right',
            textfont=dict(color=TEXT_COL, size=11),
            name='Last', showlegend=False,
        ), row=1, col=2, secondary_y=False)

        # Day high / low
        fig.add_hline(y=day_hi, line_dash='dot', line_color=ACCENT_GRN,
                      line_width=1, row=1, col=2,
                      annotation_text=f' Hi ${day_hi:,.1f}',
                      annotation_font_color=ACCENT_GRN, annotation_font_size=10)
        fig.add_hline(y=day_lo, line_dash='dot', line_color=ACCENT_RED,
                      line_width=1, row=1, col=2,
                      annotation_text=f' Lo ${day_lo:,.1f}',
                      annotation_font_color=ACCENT_RED, annotation_font_size=10)

        fig.update_xaxes(title_text='Time (ET)', row=1, col=2,
                         rangeslider_visible=False)
        fig.update_yaxes(title_text='Volume (M)', secondary_y=True,
                         showgrid=False, row=1, col=2)

    else:
        # ── Fallback: spot price as horizontal reference ───────────────
        ts = _dt.now().strftime('%H:%M')
        fig.add_hline(
            y=spot, line_color=ACCENT_BLU, line_dash='dash', line_width=2,
            row=1, col=2,
            annotation_text=f'  Last  ${spot:,.2f}  ({ts})',
            annotation_font_color=ACCENT_BLU, annotation_font_size=12,
        )
        fig.add_annotation(
            x=0.65, y=0.38, xref='paper', yref='paper',
            text='Intraday data unavailable<br>'
                 '<span style="color:#9CA3AF; font-size:10px">'
                 'Market closed or Barchart endpoint restricted</span>',
            showarrow=False,
            font=dict(size=11, color=TEXT_SEC, family='Inter, sans-serif'),
            align='center',
        )

    # ── Styling ──────────────────────────────────────────────────────────
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#F9FAFB',
        font=dict(color=TEXT_COL, family='Inter, system-ui, sans-serif'),
        height=520,
        title=dict(text=f'Price vs DEX  ·  {ticker}',
                   font=dict(color=TEXT_COL, size=13,
                             family='Inter, sans-serif')),
        barmode='overlay', bargap=0.08,
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=10, color=TEXT_COL),
                    orientation='h', y=-0.12),
        margin=dict(l=50, r=20, t=50, b=50),
    )
    fig.update_yaxes(range=[lo - y_pad, hi + y_pad],
                     gridcolor=GRID_COL, zerolinecolor=GRID_COL)
    fig.update_xaxes(gridcolor=GRID_COL, zerolinecolor=GRID_COL)
    fig.update_xaxes(title_text='DEX ($M)', row=1, col=1)
    fig.update_yaxes(title_text=f'{ticker} Level', row=1, col=1)
    return fig


def gex_dex_0dte_chart(dte0_metrics: dict, spot: float, ticker: str):
    """Combined GEX + net-DEX chart for 0DTE, zoomed to the relevant strike range.

    Uses a tight delta-derived window so the chart shows only the strikes where
    0DTE gamma is actually meaningful — typically ±30–35 delta from spot.
    """
    import plotly.graph_objects as go
    bs = dte0_metrics.get('by_strike', pd.DataFrame())
    if bs.empty:
        fig = go.Figure()
        fig.update_layout(**base_layout(f'0DTE GEX/DEX — {ticker} (no data)'))
        return fig

    # Default strike window: ±35-delta strikes from the nearest expiry data
    raw0 = dte0_metrics.get('raw', pd.DataFrame())
    lo, hi = delta_strike_bounds(raw0, -0.35, 0.35) if not raw0.empty else (None, None)
    lo = lo if (lo is not None and np.isfinite(lo)) else spot * 0.975
    hi = hi if (hi is not None and np.isfinite(hi)) else spot * 1.025
    if lo >= hi:
        lo, hi = spot * 0.975, spot * 1.025
    pad = (hi - lo) * 0.05

    fig = go.Figure()

    # GEX bars (primary axis)
    colors = [ACCENT_GRN if v >= 0 else ACCENT_RED for v in bs['net_gex']]
    fig.add_trace(go.Bar(
        x=bs['strike'], y=bs['net_gex'] / 1e6,
        marker_color=colors, marker_line_width=0,
        name='Net GEX ($M)', opacity=0.85,
        hovertemplate='Strike %{x}<br>Net GEX: %{y:.1f}M<extra></extra>',
    ))

    # Net DEX overlay (secondary axis)
    fig.add_trace(go.Scatter(
        x=bs['strike'], y=bs['net_dex'] / 1e6,
        mode='lines+markers', name='Net DEX ($M)',
        line=dict(color=ACCENT_BLU, width=2),
        marker=dict(size=5),
        yaxis='y2',
        hovertemplate='Strike %{x}<br>Net DEX: %{y:.1f}M<extra></extra>',
    ))

    # Spot, GEX flip, max-gamma annotations
    fig.add_vline(x=spot, line_color=ACCENT_YLW, line_dash='dash', line_width=1.5,
                  annotation_text=f' Spot', annotation_font_color=ACCENT_YLW,
                  annotation_font_size=10)
    if dte0_metrics.get('gex_flip') and np.isfinite(dte0_metrics['gex_flip']):
        fig.add_vline(x=dte0_metrics['gex_flip'], line_color=ACCENT_RED,
                      line_dash='dot', line_width=1.5,
                      annotation_text=' GEX Flip', annotation_font_color=ACCENT_RED,
                      annotation_font_size=10)
    if dte0_metrics.get('max_gex_strike'):
        fig.add_vline(x=dte0_metrics['max_gex_strike'], line_color='#cc99ff',
                      line_dash='dot', line_width=1,
                      annotation_text=' Max Γ', annotation_font_color='#cc99ff',
                      annotation_font_size=10)

    layout = base_layout(f'0DTE  GEX & DEX by Strike — {ticker}')
    layout['xaxis'].update(range=[lo - pad, hi + pad])
    layout.update(
        barmode='overlay',
        yaxis=dict(title='GEX ($M)', gridcolor=GRID_COL, zerolinecolor=GRID_COL,
                   color=TEXT_COL),
        yaxis2=dict(title='DEX ($M)', overlaying='y', side='right',
                    gridcolor='rgba(0,0,0,0)', color=ACCENT_BLU),
        legend=dict(orientation='h', y=1.08, font=dict(color=TEXT_COL, size=11)),
    )
    fig.update_layout(**layout)
    return fig


def oi_0dte_chart(dte0_metrics: dict, spot: float, ticker: str):
    """Call vs Put open interest per strike for the 0DTE chain."""
    import plotly.graph_objects as go
    raw0 = dte0_metrics.get('raw', pd.DataFrame())
    if raw0.empty:
        fig = go.Figure()
        fig.update_layout(**base_layout(f'0DTE OI — {ticker} (no data)'))
        return fig

    lo, hi = delta_strike_bounds(raw0, -0.35, 0.35) if not raw0.empty else (None, None)
    lo = lo if (lo is not None and np.isfinite(lo)) else spot * 0.975
    hi = hi if (hi is not None and np.isfinite(hi)) else spot * 1.025
    if lo >= hi:
        lo, hi = spot * 0.975, spot * 1.025
    pad = (hi - lo) * 0.05

    calls = raw0[raw0['flag'] == 'c'].groupby('strike')['openInterest'].sum().reset_index()
    puts  = raw0[raw0['flag'] == 'p'].groupby('strike')['openInterest'].sum().reset_index()
    puts['openInterest'] = -puts['openInterest']    # flip puts below zero

    fig = go.Figure()
    fig.add_trace(go.Bar(x=calls['strike'], y=calls['openInterest'] / 1e3,
                          name='Call OI (k)', marker_color=ACCENT_GRN, opacity=0.85))
    fig.add_trace(go.Bar(x=puts['strike'], y=puts['openInterest'] / 1e3,
                          name='Put OI (k)', marker_color=ACCENT_RED, opacity=0.85))
    fig.add_vline(x=spot, line_color=ACCENT_YLW, line_dash='dash', line_width=1.5)

    layout = base_layout(f'0DTE  Call vs Put OI — {ticker}')
    layout['xaxis'].update(range=[lo - pad, hi + pad])
    layout.update(barmode='overlay', yaxis_title='Open Interest (k)',
                  legend=dict(orientation='h', y=1.08, font=dict(color=TEXT_COL, size=11)))
    fig.update_layout(**layout)
    return fig


def smile_0dte_chart(dte0_metrics: dict, spot: float, ticker: str):
    """IV smile (call and put) for the 0DTE chain."""
    import plotly.graph_objects as go
    raw0 = dte0_metrics.get('raw', pd.DataFrame())
    if raw0.empty:
        fig = go.Figure()
        fig.update_layout(**base_layout(f'0DTE Vol Smile — {ticker} (no data)'))
        return fig

    fig = go.Figure()
    for flag, color, label in [('c', ACCENT_GRN, 'Call IV'), ('p', ACCENT_RED, 'Put IV')]:
        side = raw0[raw0['flag'] == flag].dropna(subset=['strike', 'iv'])
        side = side.sort_values('strike')
        moneyness = (side['strike'] / spot - 1) * 100
        mask = (moneyness >= -15) & (moneyness <= 5)
        fig.add_trace(go.Scatter(
            x=moneyness[mask], y=side['iv'][mask] * 100,
            mode='lines+markers', name=label,
            line=dict(color=color, width=2), marker=dict(size=4),
            hovertemplate='Moneyness: %{x:.1f}%<br>IV: %{y:.1f}%<extra></extra>',
        ))

    fig.add_vline(x=0, line_color=ACCENT_YLW, line_dash='dot', line_width=1,
                  annotation_text=' ATM', annotation_font_color=ACCENT_YLW,
                  annotation_font_size=10)
    layout = base_layout(f'0DTE  Vol Smile — {ticker}')
    layout.update(xaxis_title='Moneyness (% from spot)', yaxis_title='IV (%)',
                  legend=dict(orientation='h', y=1.08, font=dict(color=TEXT_COL, size=11)))
    fig.update_layout(**layout)
    return fig



# ══════════════════════════════════════════════════════════════════════════════
# Realized Volatility Models (pure numpy/pandas — no OpenBB required)
# ══════════════════════════════════════════════════════════════════════════════

def _rvol_ann(series: pd.Series, window: int, tp: int) -> pd.Series:
    """Annualize a rolling variance series."""
    return np.sqrt(series.rolling(window).mean() * tp)


def rvol_std(ohlc: pd.DataFrame, window: int = 30,
             trading_periods: int = 252) -> pd.Series:
    """Standard Deviation model — annualized log-return std."""
    log_ret = np.log(ohlc['close'] / ohlc['close'].shift(1))
    return (log_ret.rolling(window).std() * np.sqrt(trading_periods)
            ).rename('Std Dev')


def rvol_parkinson(ohlc: pd.DataFrame, window: int = 30,
                   trading_periods: int = 252) -> pd.Series:
    """Parkinson (1980) — High-Low range estimator."""
    hl2 = np.log(ohlc['high'] / ohlc['low']) ** 2
    k   = 1.0 / (4.0 * np.log(2.0))
    return (np.sqrt(trading_periods * k * hl2.rolling(window).mean())
            ).rename('Parkinson')


def rvol_garman_klass(ohlc: pd.DataFrame, window: int = 30,
                      trading_periods: int = 252) -> pd.Series:
    """Garman-Klass (1980) — OHLC estimator, ~8× more efficient than Std Dev."""
    hl2 = 0.5 * np.log(ohlc['high'] / ohlc['low']) ** 2
    co2 = (2.0 * np.log(2.0) - 1.0) * np.log(ohlc['close'] / ohlc['open']) ** 2
    return (np.sqrt(trading_periods * (hl2 - co2).rolling(window).mean())
            ).rename('Garman-Klass')


def rvol_rogers_satchell(ohlc: pd.DataFrame, window: int = 30,
                          trading_periods: int = 252) -> pd.Series:
    """Rogers-Satchell (1991) — non-zero drift, no overnight gap."""
    h_o = np.log(ohlc['high']  / ohlc['open'])
    l_o = np.log(ohlc['low']   / ohlc['open'])
    h_c = np.log(ohlc['high']  / ohlc['close'])
    l_c = np.log(ohlc['low']   / ohlc['close'])
    rs  = h_o * h_c + l_o * l_c
    return (np.sqrt(trading_periods * rs.rolling(window).mean())
            ).rename('Rogers-Satchell')


def rvol_hodges_tompkins(ohlc: pd.DataFrame, window: int = 30,
                          trading_periods: int = 252) -> pd.Series:
    """Hodges-Tompkins (1992) — bias-corrected Std Dev for overlapping windows."""
    σ_std = rvol_std(ohlc, window, trading_periods)
    # Correction: sqrt(window / (window - 1))
    corr  = np.sqrt(window / max(window - 1, 1))
    return (σ_std * corr).rename('Hodges-Tompkins')


def rvol_yang_zhang(ohlc: pd.DataFrame, window: int = 30,
                    trading_periods: int = 252) -> pd.Series:
    """Yang-Zhang (2000) — minimum-variance, handles drift and overnight gaps."""
    log_oc  = np.log(ohlc['close'] / ohlc['open'])       # open-to-close
    log_on  = np.log(ohlc['open']  / ohlc['close'].shift(1))  # overnight
    h_o = np.log(ohlc['high']  / ohlc['open'])
    l_o = np.log(ohlc['low']   / ohlc['open'])
    h_c = np.log(ohlc['high']  / ohlc['close'])
    l_c = np.log(ohlc['low']   / ohlc['close'])
    σ_rs_sq  = (h_o * h_c + l_o * l_c).rolling(window).mean()
    σ_on_sq  = log_on.rolling(window).var()
    σ_oc_sq  = log_oc.rolling(window).var()
    k = 0.34 / (1.34 + (window + 1) / max(window - 1, 1))
    yz = np.sqrt(trading_periods * (σ_on_sq + k * σ_oc_sq + (1 - k) * σ_rs_sq))
    return yz.rename('Yang-Zhang')


def rvol_ewma(ohlc: pd.DataFrame,
              lambda_: float = 0.94,
              trading_periods: int = 252) -> pd.Series:
    """EWMA Realized Volatility — RiskMetrics model (J.P. Morgan, 1994).

    σ²_t = λ·σ²_{t-1} + (1−λ)·r²_t

    λ = 0.94 is the daily RiskMetrics standard.  Compared to rolling
    windows, EWMA weights recent observations exponentially more, making
    it the most reactive purely-returns-based estimator.  Half-life:
    ln(0.5)/ln(λ) ≈ 11 days for λ=0.94.

    Useful as the realized-vol counterpart to VIX: when the market
    spikes, EWMA jumps within 1-3 days vs 2-4 weeks for a 30-day window.
    """
    log_ret  = np.log(ohlc['close'] / ohlc['close'].shift(1)).dropna()
    r2       = log_ret ** 2
    ewma_var = r2.ewm(alpha=1 - lambda_, adjust=False).mean()
    return (np.sqrt(ewma_var * trading_periods)).rename(f'EWMA λ={lambda_}')


def compute_rvol_all(ohlc: pd.DataFrame, window: int = 126,
                     trading_periods: int = 252) -> pd.DataFrame:
    """All realized vol models in a single DataFrame, values in %.

    Includes the 6 OHLC models at the specified window PLUS reactive estimators:
    - EWMA (λ=0.94): RiskMetrics — most reactive, half-life ~11 days
    - Yang-Zhang 5d: short-window, tracks weekly realized vol
    - Yang-Zhang 21d: one-month, fair comparison to VIX 30d IV

    Output index is DatetimeIndex (from ohlc['date'] column if present).
    """
    if ohlc is None or ohlc.empty or len(ohlc) < max(window, 21) + 5:
        return pd.DataFrame()
    work = ohlc.set_index('date') if 'date' in ohlc.columns else ohlc
    parts = [f(work, window, trading_periods)
             for f in [rvol_std, rvol_parkinson, rvol_garman_klass,
                       rvol_hodges_tompkins, rvol_rogers_satchell, rvol_yang_zhang]]
    # Reactive estimators — always computed regardless of main window
    parts += [
        rvol_ewma(work, lambda_=0.94,  trading_periods=trading_periods),
        rvol_yang_zhang(work, window=5,  trading_periods=trading_periods
                        ).rename('Yang-Zhang 5d'),
        rvol_yang_zhang(work, window=21, trading_periods=trading_periods
                        ).rename('Yang-Zhang 21d'),
    ]
    return pd.concat(parts, axis=1).dropna().mul(100)


def compute_rvol_cones(ohlc: pd.DataFrame,
                       windows=None,
                       trading_periods: int = 252) -> pd.DataFrame:
    """Volatility cones: realized vol quantile distribution at each window.

    Returns DataFrame indexed by window (days) with columns:
    min, q10, q25, q50, q75, q90, max, current.  Values in %.
    """
    if windows is None:
        windows = [3, 5, 10, 21, 30, 45, 63, 90]
    records = []
    for w in windows:
        if ohlc is None or ohlc.empty or len(ohlc) < w + 5:
            continue
        roll = (rvol_yang_zhang(ohlc, w, trading_periods) * 100).dropna()
        if roll.empty:
            continue
        records.append({
            'window':  w,
            'min':     float(roll.min()),
            'q10':     float(roll.quantile(0.10)),
            'q25':     float(roll.quantile(0.25)),
            'q50':     float(roll.quantile(0.50)),
            'q75':     float(roll.quantile(0.75)),
            'q90':     float(roll.quantile(0.90)),
            'max':     float(roll.max()),
            'current': float(roll.iloc[-1]),
        })
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).set_index('window')


def compute_intraday_rvol(intraday_df: pd.DataFrame,
                           trading_periods: int = 252,
                           bars_per_day: int = 78) -> float:
    """Current-session realized vol from 5-min intraday bars, annualized.

    bars_per_day = 78 (6.5 h × 12 bars/h).
    Returns annualized float (e.g. 0.142 = 14.2%) or None.
    """
    if intraday_df is None or intraday_df.empty:
        return None
    closes = intraday_df['close'].dropna()
    if len(closes) < 2:
        return None
    log_ret = np.log(closes / closes.shift(1)).dropna()
    σ = float(log_ret.std() * np.sqrt(trading_periods * bars_per_day))
    return σ if np.isfinite(σ) else None


# ── Chart builders ─────────────────────────────────────────────────────────
_RVOL_COLORS = {
    # 6 standard models (muted — background reference)
    'Std Dev':           '#6C63FF',
    'Parkinson':         '#10B981',
    'Garman-Klass':      '#3B82F6',
    'Hodges-Tompkins':   '#F59E0B',
    'Rogers-Satchell':   '#EC4899',
    'Yang-Zhang':        '#EAB308',
    # Reactive estimators (bold — foreground, key comparison vs VIX)
    'EWMA λ=0.94':       '#FF6B35',   # vivid orange — most reactive
    'Yang-Zhang 5d':     '#00C9A7',   # teal — short-window
    'Yang-Zhang 21d':    '#C77DFF',   # violet — VIX-comparable window
}

_RVOL_DASH = {
    'EWMA λ=0.94':     'solid',
    'Yang-Zhang 5d':   'solid',
    'Yang-Zhang 21d':  'solid',
}
_RVOL_WIDTH = {
    'EWMA λ=0.94':    2.8,
    'Yang-Zhang 5d':  1.8,
    'Yang-Zhang 21d': 2.2,
}


def rvol_models_chart(rvol_df: pd.DataFrame, vix_df: pd.DataFrame,
                      ticker: str = '$SPX'):
    """RVol models vs VIX.  Reactive estimators (EWMA, short-window YZ) are
    drawn bold in the foreground; the 6 standard models are muted background lines.
    """
    import plotly.graph_objects as go
    fig = go.Figure()

    _reactive = {'EWMA λ=0.94', 'Yang-Zhang 5d', 'Yang-Zhang 21d'}

    if rvol_df is not None and not rvol_df.empty:
        # Draw standard models first (muted, thin)
        for col in [c for c in rvol_df.columns if c not in _reactive]:
            color = _RVOL_COLORS.get(col, ACCENT_BLU)
            fig.add_trace(go.Scatter(
                x=rvol_df.index, y=rvol_df[col],
                mode='lines', name=col,
                line=dict(color=color, width=1.2),
                opacity=0.45,
                hovertemplate=f'{col}: %{{y:.1f}}%<extra></extra>',
            ))
        # Draw reactive estimators bold on top
        for col in [c for c in rvol_df.columns if c in _reactive]:
            color = _RVOL_COLORS.get(col, ACCENT_BLU)
            fig.add_trace(go.Scatter(
                x=rvol_df.index, y=rvol_df[col],
                mode='lines', name=col,
                line=dict(color=color,
                          width=_RVOL_WIDTH.get(col, 2.0),
                          dash=_RVOL_DASH.get(col, 'solid')),
                hovertemplate=f'{col}: %{{y:.1f}}%<extra></extra>',
            ))

    # VIX — thickest, most prominent (the benchmark)
    if vix_df is not None and not vix_df.empty:
        vix_pct = vix_df.set_index('date')['vix'] * 100
        fig.add_trace(go.Scatter(
            x=vix_pct.index, y=vix_pct,
            mode='lines', name='VIX (Implied)',
            line=dict(color=ACCENT_RED, width=3.0, dash='dot'),
            hovertemplate='VIX: %{y:.1f}%<extra></extra>',
        ))

    layout = base_layout(
        f'Realized Vol vs VIX — {ticker}  '
        f'(bold = reactive: EWMA / YZ-5d / YZ-21d)', height=420)
    layout.update(
        yaxis_title='Annualized Vol (%)',
        xaxis_title='',
        legend=dict(orientation='h', y=-0.22, font=dict(size=10, color=TEXT_COL),
                    bgcolor='rgba(0,0,0,0)'),
    )
    fig.update_layout(**layout)
    return fig


def rvol_cones_chart(cones_df: pd.DataFrame,
                      ticker: str = '$SPX'):
    """Volatility cones: Yang-Zhang RVol quantile bands at each window length."""
    import plotly.graph_objects as go
    if cones_df is None or cones_df.empty:
        fig = go.Figure()
        fig.update_layout(**base_layout(f'Vol Cones — {ticker} (no data)'))
        return fig

    windows = cones_df.index.tolist()
    fig = go.Figure()

    # Outer band: q10 → q90
    fig.add_trace(go.Scatter(
        x=windows + windows[::-1],
        y=cones_df['q90'].tolist() + cones_df['q10'].tolist()[::-1],
        fill='toself', fillcolor='rgba(108,99,255,0.08)',
        line=dict(color='rgba(0,0,0,0)'),
        name='10–90th pct', hoverinfo='skip',
    ))

    # Inner band: q25 → q75
    fig.add_trace(go.Scatter(
        x=windows + windows[::-1],
        y=cones_df['q75'].tolist() + cones_df['q25'].tolist()[::-1],
        fill='toself', fillcolor='rgba(108,99,255,0.18)',
        line=dict(color='rgba(0,0,0,0)'),
        name='25–75th pct', hoverinfo='skip',
    ))

    # Median line
    fig.add_trace(go.Scatter(
        x=windows, y=cones_df['q50'],
        mode='lines', name='Median',
        line=dict(color=ACCENT_BLU, width=2, dash='dash'),
        hovertemplate='Window %{x}d — Median: %{y:.1f}%<extra></extra>',
    ))

    # Current value markers
    fig.add_trace(go.Scatter(
        x=windows, y=cones_df['current'],
        mode='lines+markers', name='Current',
        line=dict(color=ACCENT_RED, width=2),
        marker=dict(size=9, color=ACCENT_RED,
                    line=dict(color='white', width=2)),
        hovertemplate='Window %{x}d — Current: %{y:.1f}%<extra></extra>',
    ))

    layout = base_layout(f'Volatility Cones (Yang-Zhang) — {ticker}', height=380)
    layout.update(
        xaxis_title='Rolling Window (trading days)',
        yaxis_title='Annualized RVol (%)',
        legend=dict(orientation='h', y=-0.20, font=dict(size=10, color=TEXT_COL),
                    bgcolor='rgba(0,0,0,0)'),
        xaxis=dict(tickmode='array', tickvals=windows,
                   ticktext=[f'{w}d' for w in windows],
                   gridcolor=GRID_COL, zerolinecolor=GRID_COL),
        yaxis=dict(gridcolor=GRID_COL, zerolinecolor=GRID_COL),
    )
    fig.update_layout(**layout)
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
        'updated_at': None,
        'ohlc':      None,   # kept for compatibility
        'intraday':  None,   # today's 5-min bars — fetched alongside the chain
    }
    store_lock = threading.Lock()
    loader_lock = threading.Lock()
    loader_state = {
        'request_id': 0,
        'status': 'idle',
        'ticker': DEFAULT_TICKER,
        'progress': 0,
        'message': 'Enter a ticker and press  ⬇ LOAD DATA  to begin.',
        'error': None,
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

    def filter_input(input_id, placeholder, value=None, width='96px'):
        return dcc.Input(
            id=input_id,
            type='number',
            value=value,
            placeholder=placeholder,
            debounce=True,
            style={
                'background': '#1a1d27',
                'border': f'1px solid {GRID_COL}',
                'color': TEXT_COL,
                'padding': '8px 10px',
                'borderRadius': '4px',
                'fontFamily': 'Courier New',
                'fontSize': '13px',
                'width': width,
            },
        )

    def filter_group(label, min_id, max_id, min_placeholder, max_placeholder, min_value=None, max_value=None):
        return html.Div([
            html.P(label, style={
                'color': '#7a8399',
                'fontSize': '11px',
                'margin': '0 0 6px 0',
                'letterSpacing': '1.5px',
                'textTransform': 'uppercase',
                'fontFamily': 'Courier New',
            }),
            html.Div([
                filter_input(min_id, min_placeholder, min_value),
                filter_input(max_id, max_placeholder, max_value),
            ], style={'display': 'flex', 'gap': '8px'}),
        ], style={'display': 'flex', 'flexDirection': 'column'})

    def option_type_filter():
        return html.Div([
            html.P('Option Type', style={
                'color': '#7a8399',
                'fontSize': '11px',
                'margin': '0 0 6px 0',
                'letterSpacing': '1.5px',
                'textTransform': 'uppercase',
                'fontFamily': 'Courier New',
            }),
            dcc.Checklist(
                id='option-type-input',
                options=[
                    {'label': 'Call', 'value': 'c'},
                    {'label': 'Put', 'value': 'p'},
                ],
                value=['c', 'p'],
                inline=True,
                inputStyle={'marginRight': '6px', 'marginLeft': '0'},
                labelStyle={
                    'display': 'inline-flex',
                    'alignItems': 'center',
                    'marginRight': '14px',
                    'color': TEXT_COL,
                    'fontFamily': 'Courier New',
                    'fontSize': '13px',
                },
                style={
                    'background': '#13161e',
                    'border': f'1px solid {GRID_COL}',
                    'borderRadius': '4px',
                    'padding': '8px 12px',
                    'minHeight': '38px',
                },
            ),
        ], style={'display': 'flex', 'flexDirection': 'column', 'minWidth': '210px'})

    def empty_figure():
        return go.Figure().update_layout(**base_layout())

    def snapshot_store():
        with store_lock:
            return dict(store)

    def snapshot_loader():
        with loader_lock:
            return dict(loader_state)

    def set_loader_state(**updates):
        with loader_lock:
            loader_state.update(updates)
            return dict(loader_state)

    def report_progress(request_id: int, progress: int, message: str) -> bool:
        with loader_lock:
            if loader_state['request_id'] != request_id:
                return False
            loader_state.update({
                'status': 'loading',
                'progress': max(0, min(100, int(progress))),
                'message': message,
                'error': None,
            })
            return True

    def load_dashboard_data(request_id: int, ticker: str, force_refresh: bool = False):
        try:
            report_progress(request_id, 2, f'Starting {ticker} data feed ...')
            report_progress(request_id, 3, f'Resolving {ticker} symbol on Barchart ...')
            resolved_barchart_ticker = resolve_barchart_symbol(ticker)
            report_progress(request_id, 4, f'Connecting to Barchart for {ticker} ...')
            raw_df, spot = fetch_options_data(
                resolved_barchart_ticker,
                progress_callback=lambda pct, msg: report_progress(request_id, pct, msg),
                force_refresh=force_refresh,
            )
            report_progress(request_id, 97, f'Finalizing {ticker} dashboard feed ...')

            # Fetch today's intraday 5-min bars — lightweight, non-fatal
            try:
                intraday = fetch_intraday_history(resolved_barchart_ticker, interval_min=5)
            except Exception:
                intraday = pd.DataFrame()

            with store_lock:
                store['raw'] = raw_df
                store['by_strike'] = None
                store['by_expiry'] = None
                store['spot'] = spot
                store['ticker'] = resolved_barchart_ticker
                store['updated_at'] = datetime.now()
                store['intraday'] = intraday

            set_loader_state(
                status='ready',
                ticker=ticker,
                progress=100,
                message=f'{ticker} feed loaded',
                error=None,
            )
        except Exception as exc:
            set_loader_state(
                status='error',
                ticker=ticker,
                progress=100,
                message=f'Error loading {ticker}',
                error=str(exc),
            )

    def start_background_load(ticker: str, force_refresh: bool = False) -> int:
        with loader_lock:
            loader_state['request_id'] += 1
            request_id = loader_state['request_id']
            loader_state.update({
                'status': 'loading',
                'ticker': ticker,
                'progress': 0,
                'message': f'Queued {ticker} load ...',
                'error': None,
            })

        worker = threading.Thread(target=load_dashboard_data, args=(request_id, ticker, force_refresh), daemon=True)
        worker.start()
        return request_id

    def resolve_ticker(manual_value):
        manual = (manual_value or '').strip().upper()
        return manual or DEFAULT_TICKER

    def progress_style(loader):
        if loader['status'] == 'error':
            return 'danger', False, False
        if loader['status'] == 'loading':
            return 'info', True, True
        if loader['status'] == 'ready':
            return 'success', False, False
        return 'secondary', False, False

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
                      placeholder='Type ticker ...',
                      style={'background': '#1a1d27', 'border': f'1px solid {GRID_COL}',
                             'color': TEXT_COL, 'padding': '8px 12px', 'borderRadius': '4px',
                             'fontFamily': 'Courier New', 'fontSize': '14px',
                             'width': '180px', 'textTransform': 'uppercase'}),
            html.Div([
                    html.Label('Δ Filter',
                               style={'color': '#7a8399', 'fontFamily': 'Courier New',
                                      'fontSize': '10px', 'letterSpacing': '1px',
                                      'display': 'block', 'marginBottom': '2px'}),
                    dcc.RangeSlider(
                        id='delta-range-slider',
                        min=-1, max=1, step=0.05,
                        value=[-0.20, 0.20],
                        marks={v: f'{v:+.2f}' for v in [-1, -0.5, -0.20, 0, 0.20, 0.5, 1]},
                        tooltip={'always_visible': False, 'placement': 'bottom'},
                        className='mx-2',
                    ),
                ], style={'width': '260px'}),
                   html.Button('⬇ CARICA FULL CHAIN', id='refresh-btn',
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

        html.Div([
            option_type_filter(),
            filter_group('Delta', 'delta-min-input', 'delta-max-input', 'Min <=', 'Max <=', -1, 1),
            filter_group('Expiry Days', 'dte-min-input', 'dte-max-input', 'Min', 'Max', 0, 90),
            filter_group('Open Interest', 'oi-min-input', 'oi-max-input', 'Min <=', 'Max <=', OI_THRESHOLD, None),
            html.Button('⊞ APPLICA FILTRI', id='apply-filters-btn',
                style={'background': 'transparent', 'border': f'1px solid {ACCENT_BLU}',
                       'color': ACCENT_BLU, 'padding': '8px 20px',
                       'borderRadius': '4px', 'cursor': 'pointer',
                       'fontFamily': 'Courier New', 'fontSize': '13px',
                       'letterSpacing': '1px', 'alignSelf': 'flex-end'}),
        ], style={
            'display': 'flex',
            'gap': '18px',
            'padding': '12px 28px 0',
            'flexWrap': 'wrap',
            'alignItems': 'flex-end',
        }),

        html.Div(id='status-bar', style={'padding': '6px 28px', 'fontSize': '12px',
                                         'color': '#7a8399', 'fontFamily': 'Courier New',
                                         'borderBottom': f'1px solid {GRID_COL}'}),

        html.Div([
            dbc.Progress(
                id='load-progress',
                value=0,
                label='0%',
                striped=True,
                animated=True,
                color='info',
                style={'height': '22px', 'fontSize': '13px'}
            ),
            html.Div(id='load-message', style={
                'textAlign': 'center', 'fontFamily': 'Courier New',
                'fontSize': '14px', 'color': ACCENT_BLU, 'marginTop': '8px',
                'minHeight': '20px',
            }),
        ], style={'padding': '10px 28px 4px'}),

        html.Div(id='stat-cards',
                 style={'display': 'flex', 'gap': '12px', 'padding': '16px 28px',
                        'flexWrap': 'wrap'}),

        # Price vs DEX Levels — full width, the new centrepiece chart
        html.Div(dcc.Graph(id='price-dex-chart'),
                 style={'padding': '0 28px 12px'}),

        html.Div([
            html.Div(dcc.Graph(id='gex-bar'), style={'flex': '1', 'minWidth': '400px'}),
            html.Div(dcc.Graph(id='dex-bar'), style={'flex': '1', 'minWidth': '400px'}),
        ], style={'display': 'flex', 'gap': '12px', 'padding': '0 28px 12px'}),

        html.Div([
            html.Div(dcc.Graph(id='gex-expiry'), style={'flex': '1', 'minWidth': '300px'}),
            html.Div(dcc.Graph(id='oi-heatmap'), style={'flex': '1.3', 'minWidth': '380px'}),
            html.Div(dcc.Graph(id='vol-smile'), style={'flex': '1', 'minWidth': '300px'}),
        ], style={'display': 'flex', 'gap': '12px', 'padding': '0 28px 28px'}),

        dcc.Store(id='load-request-store'),
        dcc.Store(id='filter-store', data={
            'option_flags': ['c', 'p'],
            'delta_min': -1,
            'delta_max': 1,
            'dte_min': 0,
            'dte_max': 90,
            'oi_min': OI_THRESHOLD,
            'oi_max': None,
            'timestamp': datetime.now().isoformat(),
        }),
        dcc.Store(id='skew-expiry-store'),
        dcc.Interval(id='loading-interval', interval=1500, n_intervals=0),

        # ── Section: Range & Skew ───────────────────────────────────────────
        html.Div([
            html.Span('📐  DAILY RANGE  &  IV SKEW',
                      style={'color': ACCENT_BLU, 'fontFamily': 'Courier New',
                             'fontSize': '13px', 'letterSpacing': '3px'}),
        ], style={'padding': '18px 28px 4px',
                  'borderTop': f'1px solid {GRID_COL}',
                  'marginTop': '12px'}),

        # Controls: IV level checkboxes + expiry selector
        html.Div([
            html.Div([
                html.Label('IV Reference Levels',
                           style={'color': '#7a8399', 'fontSize': '11px',
                                  'letterSpacing': '1px', 'display': 'block',
                                  'marginBottom': '6px'}),
                dcc.Checklist(
                    id='iv-levels-checklist',
                    options=[
                        {'label': ' 16%', 'value': 0.16},
                        {'label': ' 18%', 'value': 0.18},
                        {'label': ' 20%', 'value': 0.20},
                    ],
                    value=[0.16, 0.18, 0.20],
                    inline=True,
                    style={'color': TEXT_COL, 'fontFamily': 'Courier New',
                           'fontSize': '13px', 'gap': '12px'},
                    inputStyle={'marginRight': '4px'},
                ),
            ], style={'flex': '0 0 auto'}),

            html.Div([
                html.Label('Expiries on Skew Chart (leave blank = all ≤90d)',
                           style={'color': '#7a8399', 'fontSize': '11px',
                                  'letterSpacing': '1px', 'display': 'block',
                                  'marginBottom': '4px'}),
                dcc.Dropdown(
                    id='skew-expiry-dropdown',
                    options=[],
                    value=[],
                    multi=True,
                    placeholder='All near-term expiries…',
                    style={'background': '#1a1d27', 'color': TEXT_COL,
                           'fontFamily': 'Courier New', 'fontSize': '12px',
                           'border': f'1px solid {GRID_COL}', 'minWidth': '420px'},
                ),
            ], style={'flex': '1'}),
        ], style={'display': 'flex', 'gap': '28px', 'padding': '6px 28px 10px',
                  'alignItems': 'flex-end', 'flexWrap': 'wrap'}),

        html.Div([
            html.Div(dcc.Graph(id='daily-range-chart'), style={'flex': '1', 'minWidth': '400px'}),
            html.Div(dcc.Graph(id='skew-overlay-chart'), style={'flex': '1.2', 'minWidth': '440px'}),
        ], style={'display': 'flex', 'gap': '12px', 'padding': '0 28px 28px'}),

        # ── Section: Put Monitor ────────────────────────────────────────────
        html.Div([
            html.Span('💰  PUT PREMIUM MONITOR  —  Monthly / Quarterly',
                      style={'color': ACCENT_GRN, 'fontFamily': 'Courier New',
                             'fontSize': '13px', 'letterSpacing': '3px'}),
        ], style={'padding': '18px 28px 4px',
                  'borderTop': f'1px solid {GRID_COL}'}),

        html.Div([
            html.Div(dcc.Graph(id='put-monitor-chart'),
                     style={'flex': '1.2', 'minWidth': '440px'}),
            html.Div(id='put-monitor-table',
                     style={'flex': '1', 'minWidth': '360px',
                            'overflowX': 'auto', 'padding': '8px 0'}),
        ], style={'display': 'flex', 'gap': '12px', 'padding': '0 28px 36px',
                  'flexWrap': 'wrap'}),

    ], style={'background': DARK_BG, 'minHeight': '100vh', 'fontFamily': 'Courier New'})

    # Callbacks
    @app.callback(
        Output('load-request-store', 'data'),
        Output('filter-store', 'data'),
        Output('ticker-input', 'value'),
        Input('refresh-btn', 'n_clicks'),
        Input('apply-filters-btn', 'n_clicks'),
        State('option-type-input', 'value'),
        State('ticker-input', 'value'),
        State('delta-min-input', 'value'),
        State('delta-max-input', 'value'),
        State('dte-min-input', 'value'),
        State('dte-max-input', 'value'),
        State('oi-min-input', 'value'),
        State('oi-max-input', 'value'),
        prevent_initial_call=False,
     )
    def update_request(
        n_clicks,
        n_clicks_filter,
        option_flags,
        ticker_val,
        delta_min,
        delta_max,
        dte_min,
        dte_max,
        oi_min,
        oi_max,
    ):
        triggered = ctx.triggered_id
        ticker = resolve_ticker(ticker_val)
        current_store = snapshot_store()
        loader = snapshot_loader()
        current_ticker = resolve_ticker(current_store.get('ticker'))
        has_cached_data = current_store.get('raw') is not None and current_store.get('spot') is not None
        load_clicked = triggered == 'refresh-btn'
        filter_clicked = triggered == 'apply-filters-btn'

        if loader['status'] == 'loading' and resolve_ticker(loader.get('ticker')) == ticker:
            # Already loading this ticker — don't start a second thread
            request_id = loader['request_id']
        elif load_clicked:
            # Explicit LOAD DATA press always goes straight to Barchart.
            # force_refresh=True skips both the in-memory and CSV cache so the
            # button is never blocked by a slow or stale cached file.
            request_id = start_background_load(ticker, force_refresh=True)
        elif filter_clicked and has_cached_data:
            # Apply filters only — reuse in-memory data, no reload
            request_id = loader['request_id']
        else:
            # Initial page load (or any non-load trigger before data exists):
            # do NOT auto-fetch. Stay idle until the user presses LOAD DATA.
            request_id = loader['request_id']

        load_request = {
            'request_id': request_id,
            'ticker': ticker,
            'triggered_by': triggered or 'initial-load',
            'timestamp': datetime.now().isoformat(),
        }
        filter_request = {
            'option_flags': option_flags or [],
            'delta_min': delta_min,
            'delta_max': delta_max,
            'dte_min': dte_min,
            'dte_max': dte_max,
            'oi_min': oi_min,
            'oi_max': oi_max,
            'timestamp': datetime.now().isoformat(),
        }
        return load_request, filter_request, ticker


    @app.callback(
        Output('stat-cards', 'children'),
        Output('price-dex-chart', 'figure'),
        Output('gex-bar', 'figure'),
        Output('dex-bar', 'figure'),
        Output('gex-expiry', 'figure'),
        Output('oi-heatmap', 'figure'),
        Output('vol-smile', 'figure'),
        Output('status-bar', 'children'),
        Output('load-progress', 'value'),
        Output('load-progress', 'label'),
        Output('load-progress', 'color'),
        Output('load-progress', 'animated'),
        Output('load-progress', 'striped'),
        Output('load-message', 'children'),
        Input('loading-interval', 'n_intervals'),
        Input('delta-range-slider', 'value'),
        Input('load-request-store', 'data'),
        Input('filter-store', 'data'),
        prevent_initial_call=False,
     )
    def update_dashboard(
        n_intervals,
        delta_range,
        load_request,
        filter_store,
    ):
        try:
            return _update_dashboard_inner(n_intervals, delta_range, load_request, filter_store)
        except Exception as _cb_exc:
            import traceback, tempfile, os as _os
            _tb = traceback.format_exc()
            print(f'[CALLBACK ERROR] {_tb}')
            try:
                _logp = _os.path.join(tempfile.gettempdir(), 'dash_cb_error.txt')
                with open(_logp, 'a') as _f:
                    _f.write(_tb + '\n---\n')
            except Exception:
                pass
            empty = empty_figure()
            err_msg = str(_cb_exc)[:200]
            return ([], empty, empty, empty, empty, empty, empty,
                    f'Callback error: {err_msg}',
                    0, '0%', 'danger', False, False, err_msg)

    def _update_dashboard_inner(
        n_intervals,
        delta_range,
        load_request,
        filter_store,
    ):
        dr      = delta_range or [-0.5, 0.5]
        delta_lo, delta_hi = float(dr[0]), float(dr[1])
        current_store = snapshot_store()
        loader = snapshot_loader()
        progress_color, progress_animated, progress_striped = progress_style(loader)
        active_filters = filter_store or {}

        raw_df = current_store['raw']
        spot = current_store['spot']
        ticker = current_store['ticker']

        if raw_df is None or spot is None:
            if loader['error']:
                msg = f"ERROR: {loader['error']}"
                status = f"Feed error: {loader['error']}"
            elif loader['status'] == 'idle':
                msg = loader['message']
                status = loader['message']
            else:
                msg = loader['message']
                status = f"{loader['progress']}%  |  {loader['message']}"
            empty = empty_figure()
            return (
                [], empty, empty, empty, empty, empty, empty,
                status,
                loader['progress'], f"{loader['progress']}%",
                progress_color, progress_animated, progress_striped,
                msg,
            )

        updated_at = current_store['updated_at']
        ts = updated_at.strftime('%H:%M:%S') if updated_at else datetime.now().strftime('%H:%M:%S')

        filtered_df = apply_dashboard_filters(
            raw_df,
            option_flags=active_filters.get('option_flags'),
            delta_min=active_filters.get('delta_min'),
            delta_max=active_filters.get('delta_max'),
            dte_min=active_filters.get('dte_min'),
            dte_max=active_filters.get('dte_max'),
            oi_min=active_filters.get('oi_min'),
            oi_max=active_filters.get('oi_max'),
        )

        if filtered_df.empty:
            empty = empty_figure()
            status = (f'Last updated: {ts}  |  {ticker}  |  Spot ${spot:.2f}  |  '
                      'No contracts match the active filters')
            return (
                [], empty, empty, empty, empty, empty, empty,
                status,
                loader['progress'], f"{loader['progress']}%",
                progress_color, progress_animated, progress_striped,
                '',
            )

        bs_df = aggregate_by_strike(filtered_df)
        be_df = aggregate_by_expiry(filtered_df)
        ohlc     = current_store.get('intraday')   # today's intraday bars

        lo, hi = delta_strike_bounds(raw_df, delta_lo, delta_hi)
        stats        = build_stats(filtered_df, bs_df, spot)
        fig_pricedex = price_vs_dex_chart(bs_df, ohlc, spot, ticker,
                                           strike_lo=lo, strike_hi=hi)
        fig_gex = gex_bar_chart(bs_df, spot, ticker, strike_lo=lo, strike_hi=hi)
        fig_dex = dex_bar_chart(bs_df, spot, ticker, strike_lo=lo, strike_hi=hi)
        fig_exp = gex_expiry_chart(be_df, ticker)
        fig_oi  = oi_heatmap(filtered_df, spot, ticker, strike_lo=lo, strike_hi=hi)
        fig_vol = vol_smile_chart(filtered_df, spot, ticker)

        status = (f'Updated: {ts}  |  {ticker}  |  {len(filtered_df):,} contracts  |  '
                  f'{filtered_df["expiry"].nunique()} expiries  |  Spot ${spot:.2f}')
        load_msg = ''
        if loader['status'] == 'loading':
            load_msg = f"{loader['progress']}%  |  {loader['message']}"
        elif loader['status'] == 'error':
            load_msg = f"Feed error: {loader['error']}"
        elif loader['status'] == 'ready':
            load_msg = loader['message']

        return (
            stats, fig_pricedex, fig_gex, fig_dex, fig_exp, fig_oi, fig_vol,
            status,
            loader['progress'], f"{loader['progress']}%",
            progress_color, progress_animated, progress_striped,
            load_msg,
        )


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


    # ── Callback: Range & Skew ──────────────────────────────────────────────
    @app.callback(
        Output('daily-range-chart',   'figure'),
        Output('skew-overlay-chart',  'figure'),
        Output('skew-expiry-dropdown', 'options'),
        Input('loading-interval',     'n_intervals'),
        Input('load-request-store',   'data'),
        Input('iv-levels-checklist',  'value'),
        Input('skew-expiry-dropdown', 'value'),
        prevent_initial_call=False,
    )
    def update_range_skew(n_intervals, load_request, iv_levels, selected_expiries):
        empty = empty_figure()
        raw_df = snapshot_store().get('raw')
        spot   = snapshot_store().get('spot')
        if raw_df is None or spot is None:
            return empty, empty, []

        try:
            fig_range = daily_range_chart(raw_df, spot,
                                          snapshot_store().get('ticker', DEFAULT_TICKER))
        except Exception:
            fig_range = empty

        # Dropdown options: expiries within 90 days (most relevant for skew)
        near = raw_df[raw_df['T_days'] <= MAX_EXPIRY_DAYS]['expiry'].unique()
        options = [{'label': e, 'value': e} for e in sorted(near)]

        try:
            # If user hasn't picked any, default to all ≤90d expiries
            exp_for_chart = selected_expiries if selected_expiries else list(sorted(near))
            fig_skew = iv_skew_overlay_chart(
                raw_df, spot,
                snapshot_store().get('ticker', DEFAULT_TICKER),
                selected_expiries=exp_for_chart,
                iv_levels=iv_levels or [0.16, 0.18, 0.20],
            )
        except Exception:
            fig_skew = empty

        return fig_range, fig_skew, options


    # ── Callback: Put Monitor ───────────────────────────────────────────────
    @app.callback(
        Output('put-monitor-chart', 'figure'),
        Output('put-monitor-table', 'children'),
        Input('loading-interval',   'n_intervals'),
        Input('load-request-store', 'data'),
        prevent_initial_call=False,
    )
    def update_put_monitor(n_intervals, load_request):
        empty = empty_figure()
        raw_df = snapshot_store().get('raw')
        spot   = snapshot_store().get('spot')
        ticker = snapshot_store().get('ticker', DEFAULT_TICKER)
        if raw_df is None or spot is None:
            return empty, html.Div()

        try:
            fig_pm = put_monitor_chart(raw_df, spot, ticker)
        except Exception:
            fig_pm = empty

        # DataTable-style HTML table for the put monitor
        try:
            from dash import dash_table
            pm = get_put_monitor(raw_df, spot)
            if pm.empty:
                tbl = html.Div('No monthly/quarterly expirations in loaded chain.',
                               style={'color': '#7a8399', 'fontFamily': 'Courier New',
                                      'fontSize': '12px', 'padding': '16px'})
            else:
                disp = pm[['expiry', 'type', 'T_days', 'delta_target',
                            'strike', 'mid', 'iv_pct', 'mid_pct_spot', 'delta_actual']].copy()
                disp.columns = ['Expiry', 'Type', 'DTE', 'Δ Target',
                                'Strike', 'Mid (pts)', 'IV %', 'Mid % Spot', 'Δ Actual']
                tbl = dash_table.DataTable(
                    data=disp.to_dict('records'),
                    columns=[{'name': c, 'id': c} for c in disp.columns],
                    style_table={'overflowX': 'auto'},
                    style_header={
                        'backgroundColor': CARD_BG, 'color': '#7a8399',
                        'fontFamily': 'Courier New', 'fontSize': '11px',
                        'border': f'1px solid {GRID_COL}', 'letterSpacing': '1px',
                    },
                    style_cell={
                        'backgroundColor': DARK_BG, 'color': TEXT_COL,
                        'fontFamily': 'Courier New', 'fontSize': '12px',
                        'border': f'1px solid {GRID_COL}', 'padding': '6px 10px',
                        'textAlign': 'center',
                    },
                    style_data_conditional=[
                        {'if': {'filter_query': '{Type} = "Quarterly"'},
                         'color': ACCENT_YLW},
                        {'if': {'column_id': 'Δ Target', 'filter_query': '{Δ Target} = 0.05'},
                         'color': ACCENT_RED},
                    ],
                    page_size=20,
                    sort_action='native',
                )
        except Exception:
            tbl = html.Div()

        return fig_pm, tbl



    preferred_ports = [PORT, 8051, 8052]
    selected_port = _pick_port(preferred_ports)
    print('Dash app built.')
    # Note: do NOT pre-load here. The initial UI callback (prevent_initial_call=False)
    # triggers the first load once the browser is connected and polling, so the
    # progress bar can actually show the fetch / Greeks / CSV stages live.
    print(f"\n>>  Starting dashboard on  http://127.0.0.1:{selected_port}")
    if selected_port != PORT:
        print(f"  Note: preferred port {PORT} was unavailable; using {selected_port} instead.")
    print('   Press  Ctrl+C  in this terminal to stop.\n')
    app.run(debug=False, use_reloader=False, threaded=True,
            port=selected_port, host='127.0.0.1')


# ══════════════════════════════════════════════════════════════════════════════
# VolDex-style ATM Implied Volatility Index — independent replica
# ══════════════════════════════════════════════════════════════════════════════
# Re-implements the PUBLISHED Nations VolDex® / Nasdaq VOLQ® methodology
# (Brenner & Subrahmanyam closed-form implied volatility on at-the-money
# options, interpolated across expiries to a constant horizon) on the SPX
# chain already loaded by this dashboard.
#
# IMPORTANT: this is NOT the licensed Nations Indexes product. VolDex®,
# VOLQ®, CallDex®, PutDex® and TailDex® are registered trademarks of their
# respective owners. This is an original implementation of the publicly
# documented mathematics, computed on a different underlying (SPX, not
# NDX) using mid-quotes from this dashboard's own data feed (not real-time
# NBBO). Absolute values will differ from the official tickers — treat
# this as an independent, same-methodology measure for SPX.

VOLDEX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'voldex_history')
VOLDEX_CSV = os.path.join(VOLDEX_DIR, 'voldex_history.csv')


def _brenner_subrahmanyam_iv(price: float, F: float, T: float) -> float:
    """Closed-form implied volatility (Brenner & Subrahmanyam, 1988).

    CFIV = sqrt(2π / T) × price / F

    A fast, exact approximation valid for near at-the-money options —
    the same formula used by VOLQ/VolDex. price and F must be in the
    same units (index points).
    """
    if T <= 0 or F <= 0 or price <= 0:
        return 0.0
    return float(np.sqrt(2.0 * np.pi / T) * (price / F))


def _triangular_weights(distances: list) -> list:
    """Normalised triangular-kernel weights from a list of |distance| values.

    Bandwidth = max(distances) × 1.01, so every input gets a strictly
    positive raw weight (closer points weighted more). Mirrors the kernel
    used by VolDex/VOLQ for both strike-weighting and term-weighting.
    """
    if not distances:
        return []
    n = len(distances)
    bw = max(distances) * 1.01
    if bw <= 0:
        return [1.0 / n] * n
    raw = [max(0.0, 1.0 - d / bw) for d in distances]
    total = sum(raw)
    if total <= 0:
        return [1.0 / n] * n
    return [w / total for w in raw]


def _voldex_term_metrics(grp: pd.DataFrame, spot: float,
                         r: float = RISK_FREE_RATE) -> dict:
    """VolDex-style ATM forward, ATM call/put price and total variance for
    ONE expiry ("term"). Returns None if the expiry lacks enough strikes.

    Steps (mirrors the published VolDex/VOLQ methodology):
      1. Find K* minimising |Call(K) − Put(K)| → forward price via put-call parity
      2. Select the 2 strikes immediately above and 2 immediately below F
      3. Weight them with a triangular kernel on |K − F|
      4. ATM call/put price = weighted average across those 4 strikes
      5. Closed-form IV (Brenner-Subrahmanyam) for call and put separately
      6. Total variance = T × average(CFIV_call², CFIV_put²)
    """
    T = float(grp['T_years'].iloc[0]) if 'T_years' in grp.columns else 0.0
    if T <= 0:
        return None

    calls = grp[grp['flag'] == 'c'].dropna(subset=['strike', 'mid'])
    puts  = grp[grp['flag'] == 'p'].dropna(subset=['strike', 'mid'])
    if calls.empty or puts.empty:
        return None

    common_strikes = sorted(set(calls['strike']) & set(puts['strike']))
    if len(common_strikes) < 4:
        return None

    # 1. Forward price via put-call parity at min |C-P|
    best = None
    for K in common_strikes:
        c = float(calls.loc[calls['strike'] == K, 'mid'].iloc[0])
        p = float(puts.loc[puts['strike'] == K, 'mid'].iloc[0])
        d = abs(c - p)
        if best is None or d < best[0]:
            best = (d, K, c, p)
    _, k_star, c_star, p_star = best
    F = k_star + np.exp(r * T) * (c_star - p_star)
    if F <= 0:
        return None

    # 2. Two strikes immediately above F, two immediately at/below F
    above = sorted([k for k in common_strikes if k > F])[:2]
    below = sorted([k for k in common_strikes if k <= F], reverse=True)[:2]
    if len(above) < 2 or len(below) < 2:
        return None
    sel_strikes = sorted(below + above)

    # 3. Triangular kernel weights on |K - F|
    dists   = [abs(k - F) for k in sel_strikes]
    weights = _triangular_weights(dists)

    # 4. Weighted ATM call / put price
    atm_call = atm_put = 0.0
    for K, w in zip(sel_strikes, weights):
        c = float(calls.loc[calls['strike'] == K, 'mid'].iloc[0])
        p = float(puts.loc[puts['strike'] == K, 'mid'].iloc[0])
        atm_call += w * c
        atm_put  += w * p

    # 5. Brenner-Subrahmanyam closed-form IV
    cfiv_call = _brenner_subrahmanyam_iv(atm_call, F, T)
    cfiv_put  = _brenner_subrahmanyam_iv(atm_put, F, T)
    if cfiv_call <= 0 and cfiv_put <= 0:
        return None

    # 6. Total variance for this term
    variance = T * (cfiv_call**2 + cfiv_put**2) / 2.0

    return {
        'T_years': T, 'T_days': T * 365.0, 'forward': F, 'k_star': k_star,
        'strikes': sel_strikes, 'weights': [round(w, 3) for w in weights],
        'atm_call': round(atm_call, 2), 'atm_put': round(atm_put, 2),
        'cfiv_call': round(cfiv_call, 4), 'cfiv_put': round(cfiv_put, 4),
        'variance': variance,
    }


def compute_voldex(raw_df: pd.DataFrame, spot: float,
                   r: float = RISK_FREE_RATE,
                   target_days: float = 30.0) -> dict:
    """Independent replica of the Nations VolDex® / Nasdaq VOLQ® methodology
    on the SPX chain. See module docstring above for the trademark/scope note.

    Returns a dict:
      voldex   — headline ATM 30-day implied volatility (annualised %)
      calldex  — IV of the ~16-delta (≈1 std-dev) OTM call, interpolated to 30d
      putdex   — IV of the ~16-delta OTM put, interpolated to 30d
      taildex  — IV of the ~10-delta (deeper OTM) put — tail-risk proxy
      terms    — per-expiry diagnostic table (forward, strikes, weights, CFIV)
      error    — set if the computation could not be completed
    """
    out = {'voldex': None, 'calldex': None, 'putdex': None, 'taildex': None,
          'terms': [], 'error': None}

    if raw_df is None or raw_df.empty or not spot or spot <= 0:
        out['error'] = 'Chain non disponibile — carica la chain prima.'
        return out

    data = raw_df.dropna(subset=['T_years', 'strike', 'mid', 'flag']).copy()
    if data.empty:
        out['error'] = 'Dati insufficienti nella chain (mancano mid/strike/iv).'
        return out

    term_results = []
    for exp, grp in data.groupby('expiry'):
        m = _voldex_term_metrics(grp, spot, r)
        if m:
            m['expiry'] = exp
            term_results.append(m)

    if len(term_results) < 2:
        out['error'] = ('Servono almeno 2 scadenze valide per interpolare a '
                        f'{target_days:.0f} giorni — solo '
                        f'{len(term_results)} disponibile/i.')
        return out

    term_results.sort(key=lambda x: x['T_days'])
    out['terms'] = term_results

    # ── Headline VolDex: interpolate total variance to the target horizon ──────
    closest = sorted(term_results, key=lambda t: abs(t['T_days'] - target_days))[:4]
    dists   = [abs(t['T_days'] - target_days) for t in closest]
    weights = _triangular_weights(dists)
    variance_30 = sum(w * t['variance'] for w, t in zip(weights, closest))
    T_30   = target_days / 365.0
    cfiv_30 = np.sqrt(max(variance_30, 0.0) / T_30)
    out['voldex'] = round(100.0 * cfiv_30, 2)

    # ── Companion measures: CallDex / PutDex / TailDex ─────────────────────────
    # Use the chain's own per-contract IV at specific delta targets, then
    # apply the same triangular time-interpolation to 30 days.
    def _iv_at_delta(flag, target_delta):
        rows = []
        for exp, grp in data.groupby('expiry'):
            T = float(grp['T_years'].iloc[0]) if 'T_years' in grp.columns else 0.0
            if T <= 0:
                continue
            sub = grp[grp['flag'] == flag].dropna(subset=['iv'])
            sub = sub[sub['iv'] > 0]
            if sub.empty:
                continue
            q = float(sub['q_impl'].iloc[0]) if 'q_impl' in sub.columns else SPX_DIV_YIELD
            try:
                ds = sub.apply(lambda row: bs_delta(spot, row['strike'], T, r,
                                                    max(float(row['iv']), 1e-4),
                                                    flag, q), axis=1)
            except Exception:
                continue
            idx = (ds - target_delta).abs().idxmin()
            rows.append((T * 365.0, float(sub.loc[idx, 'iv'])))
        return rows

    def _interp_to_30d(rows):
        if not rows or len(rows) < 2:
            return None
        rs = sorted(rows, key=lambda x: abs(x[0] - target_days))[:4]
        dd = [abs(d - target_days) for d, _ in rs]
        ww = _triangular_weights(dd)
        return round(100.0 * sum(w * iv for w, (_, iv) in zip(ww, rs)), 2)

    try:
        out['calldex']  = _interp_to_30d(_iv_at_delta('c', 0.16))
        out['putdex']   = _interp_to_30d(_iv_at_delta('p', -0.16))
        out['taildex']  = _interp_to_30d(_iv_at_delta('p', -0.10))
    except Exception:
        pass

    return out


def save_voldex_snapshot(values: dict) -> str:
    """Append today's VolDex/CallDex/PutDex/TailDex to a small CSV history
    file — same persistence pattern as the chain snapshots: a daily row,
    committed by GitHub Actions, so the series survives Streamlit Cloud
    restarts and grows day by day.
    """
    os.makedirs(VOLDEX_DIR, exist_ok=True)
    row = {
        'date':    date.today().isoformat(),
        'voldex':  values.get('voldex'),
        'calldex': values.get('calldex'),
        'putdex':  values.get('putdex'),
        'taildex': values.get('taildex'),
    }
    df_new = pd.DataFrame([row])
    if os.path.exists(VOLDEX_CSV):
        try:
            df_old = pd.read_csv(VOLDEX_CSV)
            df_old = df_old[df_old['date'] != row['date']]   # replace same-day row
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        except Exception:
            df_all = df_new
    else:
        df_all = df_new
    df_all.to_csv(VOLDEX_CSV, index=False)
    return VOLDEX_CSV


def load_voldex_history() -> pd.DataFrame:
    """Load the persisted VolDex history (empty DataFrame if none yet)."""
    if os.path.exists(VOLDEX_CSV):
        try:
            df = pd.read_csv(VOLDEX_CSV)
            df['date'] = pd.to_datetime(df['date'])
            return df.sort_values('date').reset_index(drop=True)
        except Exception:
            pass
    return pd.DataFrame(columns=['date', 'voldex', 'calldex', 'putdex', 'taildex'])


def voldex_history_chart(hist: pd.DataFrame):
    """Multi-line chart of the VolDex suite history."""
    import plotly.graph_objects as go
    if hist is None or hist.empty or len(hist) < 2:
        fig = go.Figure()
        fig.update_layout(**base_layout(
            "VolDex Suite — storico (servono ≥ 2 giorni di dati)", height=360))
        return fig

    fig = go.Figure()
    colors = {'voldex': '#6C63FF', 'calldex': '#10B981',
              'putdex': '#EF4444', 'taildex': '#F59E0B'}
    names  = {'voldex': 'VolDex (ATM 30d)', 'calldex': 'CallDex (call ~16Δ)',
              'putdex': 'PutDex (put ~16Δ)', 'taildex': 'TailDex (put ~10Δ)'}
    for col in ['voldex', 'calldex', 'putdex', 'taildex']:
        if col in hist.columns:
            fig.add_scatter(x=hist['date'], y=hist[col], mode='lines+markers',
                            name=names[col],
                            line=dict(color=colors[col], width=2.2),
                            marker=dict(size=5))
    layout = base_layout("VolDex Suite (replica SPX) — storico", height=380)
    layout.update(yaxis_title='Volatilità implicita ATM (%, annualizzata)',
                  legend=dict(orientation='h', y=-0.18))
    fig.update_layout(**layout)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# GEX/DEX metrics history — persist the DERIVED metrics, not just prices
# ══════════════════════════════════════════════════════════════════════════════
# Stores a daily row of the key derived metrics (Net GEX, gamma flip, regime,
# HHI, P/C ratio, spot) so the dashboard can show the TREND, not only today's
# value. For a long-vol strategy the *change* in regime/GEX is often more
# informative than the absolute level. Same persistence pattern as VolDex
# history: a small CSV committed by GitHub Actions, surviving Cloud restarts.

GEXHIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gex_history')
GEXHIST_CSV = os.path.join(GEXHIST_DIR, 'gex_metrics_history.csv')


def save_gex_metrics_snapshot(raw_df: pd.DataFrame, by_strike_df: pd.DataFrame,
                              spot: float) -> str:
    """Append today's derived GEX/DEX metrics to the history CSV.

    One row per day (same-day re-runs replace the row). Computes the metrics
    via compute_gex_analytics so the stored values match exactly what the
    dashboard shows.
    """
    os.makedirs(GEXHIST_DIR, exist_ok=True)
    ga = compute_gex_analytics(raw_df, by_strike_df, spot) or {}
    try:
        total_gex = float(raw_df['gex'].sum())
    except Exception:
        total_gex = None
    try:
        total_dex = float(raw_df['dex'].sum())
    except Exception:
        total_dex = None
    try:
        call_oi = int(raw_df[raw_df['flag'] == 'c']['openInterest'].sum())
        put_oi  = int(raw_df[raw_df['flag'] == 'p']['openInterest'].sum())
        pcr = put_oi / call_oi if call_oi else None
    except Exception:
        pcr = None

    row = {
        'date':       date.today().isoformat(),
        'spot':       round(spot, 2) if spot else None,
        'total_gex':  round(total_gex, 0) if total_gex is not None else None,
        'total_dex':  round(total_dex, 0) if total_dex is not None else None,
        'net_gex':    round(ga.get('net_gex_total'), 0) if ga.get('net_gex_total') is not None else None,
        'gamma_flip': round(ga.get('gamma_flip'), 0) if ga.get('gamma_flip') is not None else None,
        'regime':     ga.get('regime'),
        'hhi':        round(ga.get('hhi'), 4) if ga.get('hhi') is not None else None,
        'pc_ratio':   round(pcr, 3) if pcr is not None else None,
    }
    df_new = pd.DataFrame([row])
    if os.path.exists(GEXHIST_CSV):
        try:
            old = pd.read_csv(GEXHIST_CSV)
            old = old[old['date'] != row['date']]
            allrows = pd.concat([old, df_new], ignore_index=True)
        except Exception:
            allrows = df_new
    else:
        allrows = df_new
    allrows.to_csv(GEXHIST_CSV, index=False)
    return GEXHIST_CSV


def load_gex_metrics_history() -> pd.DataFrame:
    """Load the persisted GEX/DEX metrics history (empty if none yet)."""
    if os.path.exists(GEXHIST_CSV):
        try:
            df = pd.read_csv(GEXHIST_CSV)
            df['date'] = pd.to_datetime(df['date'])
            return df.sort_values('date').reset_index(drop=True)
        except Exception:
            pass
    return pd.DataFrame(columns=['date', 'spot', 'total_gex', 'total_dex',
                                 'net_gex', 'gamma_flip', 'regime', 'hhi', 'pc_ratio'])


def gex_metrics_history_chart(hist: pd.DataFrame, metric: str = 'net_gex'):
    """Line chart of one stored GEX metric over time, with spot/flip overlay
    for the gamma_flip view."""
    import plotly.graph_objects as go
    titles = {
        'net_gex':   'Net GEX ($) — storico',
        'gamma_flip':'Gamma Flip vs Spot — storico',
        'hhi':       'HHI (concentrazione GEX) — storico',
        'pc_ratio':  'Put/Call OI ratio — storico',
        'total_dex': 'Total DEX ($) — storico',
    }
    if hist is None or hist.empty or len(hist) < 2:
        fig = go.Figure()
        fig.update_layout(**base_layout(
            f"{titles.get(metric, metric)} (servono ≥ 2 giorni)", height=320))
        return fig

    fig = go.Figure()
    if metric == 'gamma_flip':
        # overlay spot and flip to show their relationship (regime)
        fig.add_scatter(x=hist['date'], y=hist['spot'], mode='lines+markers',
                        name='Spot', line=dict(color='#6C63FF', width=2.2))
        fig.add_scatter(x=hist['date'], y=hist['gamma_flip'], mode='lines+markers',
                        name='Gamma Flip', line=dict(color='#F59E0B', width=2.2, dash='dash'))
    else:
        colors = {'net_gex': '#EF4444', 'hhi': '#10B981',
                  'pc_ratio': '#F59E0B', 'total_dex': '#3B82F6'}
        fig.add_scatter(x=hist['date'], y=hist[metric], mode='lines+markers',
                        name=metric, line=dict(color=colors.get(metric, '#6C63FF'), width=2.2))
        if metric == 'net_gex':
            fig.add_hline(y=0, line_dash='dot', line_color='#9CA3AF')
    layout = base_layout(titles.get(metric, metric), height=340)
    layout.update(legend=dict(orientation='h', y=-0.18))
    fig.update_layout(**layout)
    return fig


def cumulative_gex_chart(by_strike_df: pd.DataFrame, spot: float,
                         gamma_flip=None, ticker: str = 'SPX'):
    """Cumulative net-GEX profile across strikes — the curve whose zero-crossing
    IS the gamma flip. Makes the flip (or its absence) visible at a glance:
    if the curve never crosses zero near spot, there is no real flip and the
    market is one-sided. This is the visual complement of the Gamma Flip card.
    """
    import plotly.graph_objects as go
    if by_strike_df is None or by_strike_df.empty or 'net_gex' not in by_strike_df.columns:
        fig = go.Figure()
        fig.update_layout(**base_layout("GEX Cumulato — dati non disponibili", height=360))
        return fig

    d = by_strike_df.sort_values('strike').reset_index(drop=True)
    cum = d['net_gex'].cumsum() / 1e6   # in $M

    fig = go.Figure()
    fig.add_scatter(x=d['strike'], y=cum, mode='lines',
                    name='GEX cumulato', line=dict(color='#6C63FF', width=2.5),
                    fill='tozeroy', fillcolor='rgba(108,99,255,0.12)')
    fig.add_hline(y=0, line_dash='dot', line_color='#9CA3AF')
    if spot:
        fig.add_vline(x=spot, line_dash='dash', line_color='#F59E0B',
                      annotation_text=f'Spot ${spot:,.0f}', annotation_position='top')
    if gamma_flip:
        fig.add_vline(x=gamma_flip, line_dash='dash', line_color='#10B981',
                      annotation_text=f'Flip ${gamma_flip:,.0f}',
                      annotation_position='bottom')
    layout = base_layout(f"GEX Cumulato per Strike — {ticker}", height=380)
    layout.update(xaxis_title='Strike', yaxis_title='GEX cumulato ($M)')
    fig.update_layout(**layout)
    return fig


def signal_health_check(raw_df: pd.DataFrame, by_strike_df: pd.DataFrame,
                        spot: float, dte0_metrics: dict = None) -> list:
    """Internal consistency checks across correlated metrics — surfaces
    suspicious divergences before the user stumbles on them.

    Returns a list of {level, message} dicts. level ∈ {ok, warn, info}.
    This does NOT hide divergences; it flags them so they can be investigated.
    """
    out = []
    try:
        ga = compute_gex_analytics(raw_df, by_strike_df, spot) or {}
    except Exception:
        return [{'level': 'info', 'message': 'Analytics non disponibili per il check.'}]

    net_gex = ga.get('net_gex_total')
    regime  = ga.get('regime')
    flip    = ga.get('gamma_flip')

    # Check 1: regime vs net GEX sign coherence (flip-based may legitimately
    # differ, but a mismatch is worth explaining)
    if regime and net_gex is not None:
        sign_regime = 'LONG' if 'LONG' in regime else 'SHORT'
        sign_netgex = 'LONG' if net_gex >= 0 else 'SHORT'
        if sign_regime != sign_netgex:
            out.append({'level': 'info', 'message':
                f"Regime ({sign_regime} γ) diverge dal segno del Net GEX grezzo "
                f"({sign_netgex}). Normale: il regime usa la posizione spot/flip, "
                f"non la somma grezza. Nessun errore."})

    # Check 2: full-chain regime vs 0DTE gamma regime
    if dte0_metrics and regime:
        flip0 = dte0_metrics.get('gex_flip')
        if flip0 and spot:
            reg0 = 'LONG' if spot >= flip0 else 'SHORT'
            regF = 'LONG' if 'LONG' in regime else 'SHORT'
            if reg0 != regF:
                out.append({'level': 'warn', 'message':
                    f"Regime 0DTE ({reg0} γ) ≠ regime full chain ({regF} γ). "
                    f"Orizzonti diversi: l'intraday può divergere dal medio termine. "
                    f"Da tenere d'occhio se prendi decisioni multi-giorno."})

    # Check 3: gamma flip plausibility (should be within a reasonable band of spot)
    if flip and spot:
        dist_pct = abs(flip - spot) / spot * 100
        if dist_pct > 8:
            out.append({'level': 'warn', 'message':
                f"Gamma flip (${flip:,.0f}) è {dist_pct:.0f}% lontano dallo spot — "
                f"insolito. Verifica il grafico GEX cumulato: potrebbe non esserci "
                f"un flip reale (mercato a senso unico)."})

    # Check 4: spot near flip → unstable regime
    if flip and spot and abs(flip - spot) / spot * 100 < 0.5:
        out.append({'level': 'info', 'message':
            f"Spot molto vicino al flip (${flip:,.0f}) — regime instabile, "
            f"piccoli movimenti possono ribaltarlo."})

    if not out:
        out.append({'level': 'ok', 'message':
            'Segnali coerenti: nessuna divergenza sospetta rilevata.'})
    return out


def compute_term_structure_slope(voldex_result: dict) -> dict:
    """Derive the volatility term-structure slope from a compute_voldex() result.

    Compares short-dated implied vol (~7-12d) with longer-dated (~25-45d).
    - Contango  (short < long): normal, calm regime → safe to sell vol
    - Backwardation (short > long): acute stress / panic in progress → do NOT
      sell vol (a crash may be underway or imminent)

    Returns {'slope': long-short in vol pts, 'state': 'contango'|'backwardation'
    |'flat'|None, 'short_iv', 'long_iv'}. Robust to sparse term tables.
    """
    out = {'slope': None, 'state': None, 'short_iv': None, 'long_iv': None}
    if not voldex_result or not voldex_result.get('terms'):
        return out
    terms = [t for t in voldex_result['terms']
             if t.get('T_days') and t.get('variance') is not None and t['T_days'] > 0]
    if len(terms) < 2:
        return out

    def _iv_pct(t):
        # annualised vol from total variance: sqrt(variance / T_years)
        T = t['T_days'] / 365.0
        return 100.0 * np.sqrt(max(t["variance"], 0.0) / T) if T > 0 else None

    # nearest term to ~9 days = short; nearest to ~30 days = long
    short_t = min(terms, key=lambda t: abs(t['T_days'] - 9))
    long_t  = min(terms, key=lambda t: abs(t['T_days'] - 30))
    if short_t['T_days'] == long_t['T_days']:
        # fall back to the two extremes if they collapse to the same term
        terms_sorted = sorted(terms, key=lambda t: t['T_days'])
        short_t, long_t = terms_sorted[0], terms_sorted[-1]
        if short_t['T_days'] == long_t['T_days']:
            return out

    short_iv = _iv_pct(short_t)
    long_iv  = _iv_pct(long_t)
    if short_iv is None or long_iv is None:
        return out
    slope = long_iv - short_iv      # positive = contango, negative = backwardation
    if slope > 0.5:
        state = 'contango'
    elif slope < -0.5:
        state = 'backwardation'
    else:
        state = 'flat'
    out.update({'slope': round(slope, 2), 'state': state,
                'short_iv': round(short_iv, 2), 'long_iv': round(long_iv, 2)})
    return out


def compute_key_levels(by_strike_df: pd.DataFrame, spot: float,
                       gamma_flip=None) -> dict:
    """Identify the operative GEX levels for an annotated map (MenthorQ-style,
    own nomenclature):

      - Call Resistance: strike of the largest POSITIVE net-GEX above spot
        (the main upside gamma wall — dealers sell into it, capping rallies).
      - Put Support: strike of the largest |net-GEX| on the negative side
        below spot (the main downside gamma floor).
      - Gamma Flip: passed through (spot vs flip defines the regime).

    All derived from data already computed — no new inputs.
    """
    out = {'call_resistance': None, 'put_support': None, 'gamma_flip': gamma_flip}
    if by_strike_df is None or by_strike_df.empty or 'net_gex' not in by_strike_df.columns:
        return out
    df = by_strike_df.copy()
    above = df[df['strike'] > spot]
    below = df[df['strike'] < spot]
    # Call resistance: biggest positive GEX wall above spot
    pos_above = above[above['net_gex'] > 0]
    if not pos_above.empty:
        out['call_resistance'] = float(pos_above.loc[pos_above['net_gex'].idxmax(), 'strike'])
    elif not above.empty:
        out['call_resistance'] = float(above.loc[above['net_gex'].abs().idxmax(), 'strike'])
    # Put support: biggest |GEX| below spot
    if not below.empty:
        out['put_support'] = float(below.loc[below['net_gex'].abs().idxmax(), 'strike'])
    return out


def gex_regime_narrative(spot: float, gamma_flip, regime: str,
                         net_gex_total: float, key_levels: dict) -> str:
    """Generate a plain-language one-paragraph summary of where price sits in
    the GEX structure — the auto-commentary that turns numbers into a read.

    Uses only computed values; states the regime, the operative levels, and
    the stabilisation threshold. Deliberately factual, no predictions.
    """
    parts = []
    net_b = net_gex_total / 1e9 if net_gex_total is not None else None
    reg_txt = ('regime SHORT γ (dealer che amplificano i movimenti — volatilità in '
               'aumento)' if regime and 'SHORT' in regime else
               'regime LONG γ (dealer che smorzano i movimenti — volatilità contenuta)'
               if regime and 'LONG' in regime else 'regime indeterminato')
    if net_b is not None:
        parts.append(f"SPX a ${spot:,.0f}, Net GEX {net_b:+.2f}B — {reg_txt}.")
    else:
        parts.append(f"SPX a ${spot:,.0f} — {reg_txt}.")

    cr = key_levels.get('call_resistance')
    ps = key_levels.get('put_support')
    if ps:
        parts.append(f"Supporto put (pavimento gamma) verso {ps:,.0f}: una rottura "
                     f"sotto accelererebbe la volatilità al ribasso.")
    if cr:
        parts.append(f"Resistenza call (muro gamma) verso {cr:,.0f}.")

    if gamma_flip:
        if spot < gamma_flip:
            parts.append(f"Soglia di stabilizzazione: sopra {gamma_flip:,.0f} (gamma "
                         f"flip) il regime tornerebbe positivo e i movimenti si "
                         f"smorzerebbero. Ora il prezzo è sotto, quindi fragile.")
        else:
            parts.append(f"Il prezzo è sopra il gamma flip ({gamma_flip:,.0f}): finché "
                         f"resta sopra, i dealer tendono a stabilizzare. Sotto quel "
                         f"livello il regime diventerebbe fragile.")
    else:
        parts.append("Nessun gamma flip netto individuato: il mercato è interamente "
                     "in un solo regime sugli strike vicini al prezzo.")
    return " ".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# Premium / IV snapshot history — "specchietto premi" (premi + vol per scadenza)
# ══════════════════════════════════════════════════════════════════════════════
PREMIUM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'premium_history')

PREMIUM_MAX_MONTHS  = 6       # salva solo scadenze entro ~6 mesi
PREMIUM_BAND_DOWN   = 0.25    # salva strike da -25% dallo spot (copre skew ribassista)
PREMIUM_BAND_UP     = 0.15    # ... a +15% sopra lo spot


def _premium_csv_for(day: str) -> str:
    """One CSV per day keeps each snapshot small and the date-picker simple:
    the set of available comparison dates is just the set of files present."""
    return os.path.join(PREMIUM_DIR, f'premium_{day}.csv')


def save_premium_snapshot(raw_df: pd.DataFrame, spot: float,
                          atm_iv: float = None) -> str:
    """Persist today's premium + IV grid for SPX: for each expiry within
    PREMIUM_MAX_MONTHS, the strikes within ±PREMIUM_SD_RANGE standard
    deviations of spot, with mid premium and implied vol for both call and
    put. One row per (expiry, strike). Written as a dated CSV so the compare
    date-picker only ever offers days that actually have data.

    Robust by design: any failure returns '' and the caller carries on — a
    missing premium snapshot must never break the daily run.
    """
    try:
        if raw_df is None or raw_df.empty or not spot:
            return ''
        df = raw_df.copy()
        # mid price (fallback to lastPrice if bid/ask missing)
        if 'mid' not in df.columns:
            if 'bid' in df.columns and 'ask' in df.columns:
                df['mid'] = (df['bid'].fillna(0) + df['ask'].fillna(0)) / 2.0
                # where mid is 0/NaN, fall back to lastPrice
                _bad = (df['mid'].isna()) | (df['mid'] <= 0)
                if 'lastPrice' in df.columns:
                    df.loc[_bad, 'mid'] = df.loc[_bad, 'lastPrice']
            elif 'lastPrice' in df.columns:
                df['mid'] = df['lastPrice']
            else:
                return ''

        # T_days per expiry (add if missing)
        if 'T_days' not in df.columns:
            _today = pd.Timestamp(date.today())
            df['T_days'] = (pd.to_datetime(df['expiry']) - _today).dt.days

        # keep expiries within the horizon window
        max_days = int(PREMIUM_MAX_MONTHS * 30.4)
        df = df[(df['T_days'] > 0) & (df['T_days'] <= max_days)]
        if df.empty:
            return ''

        rows = []
        # fixed percentage band around spot (covers the downside skew wider)
        lo = spot * (1.0 - PREMIUM_BAND_DOWN)
        hi = spot * (1.0 + PREMIUM_BAND_UP)
        for expiry, grp in df.groupby('expiry'):
            T_days = float(grp['T_days'].iloc[0])
            band = grp[(grp['strike'] >= lo) & (grp['strike'] <= hi)]
            if band.empty:
                continue
            for strike, sgrp in band.groupby('strike'):
                calls = sgrp[sgrp['optionType'] == 'call']
                puts  = sgrp[sgrp['optionType'] == 'put']
                row = {
                    'expiry': expiry, 'T_days': round(T_days, 1),
                    'strike': float(strike),
                    'call_mid': round(float(calls['mid'].iloc[0]), 2) if not calls.empty else None,
                    'call_iv':  round(float(calls['impliedVolatility'].iloc[0]), 4) if not calls.empty and pd.notna(calls['impliedVolatility'].iloc[0]) else None,
                    'put_mid':  round(float(puts['mid'].iloc[0]), 2) if not puts.empty else None,
                    'put_iv':   round(float(puts['impliedVolatility'].iloc[0]), 4) if not puts.empty and pd.notna(puts['impliedVolatility'].iloc[0]) else None,
                }
                rows.append(row)

        if not rows:
            return ''
        out = pd.DataFrame(rows)
        out.insert(0, 'spot', round(spot, 2))
        out.insert(0, 'date', date.today().isoformat())
        os.makedirs(PREMIUM_DIR, exist_ok=True)
        path = _premium_csv_for(date.today().isoformat())
        out.to_csv(path, index=False)
        return path
    except Exception:
        return ''


def list_premium_dates() -> list:
    """Return the sorted list of dates that have a saved premium snapshot —
    exactly the days the compare date-picker should offer."""
    if not os.path.isdir(PREMIUM_DIR):
        return []
    days = []
    for fn in os.listdir(PREMIUM_DIR):
        if fn.startswith('premium_') and fn.endswith('.csv'):
            days.append(fn[len('premium_'):-len('.csv')])
    return sorted(days)


def load_premium_snapshot(day: str) -> pd.DataFrame:
    """Load one day's premium/IV grid (empty DataFrame if absent)."""
    path = _premium_csv_for(day)
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            pass
    return pd.DataFrame(columns=['date', 'spot', 'expiry', 'T_days', 'strike',
                                 'call_mid', 'call_iv', 'put_mid', 'put_iv'])


def build_premium_comparison(day_today: str, day_ref: str) -> pd.DataFrame:
    """Join two snapshots (today vs a chosen reference date) on (expiry, strike)
    and compute the deltas — the heart of the specchietto: for each cell, the
    premium and IV now vs then, and whether they rose or fell.

    Returns a tidy DataFrame ready to display, sorted by expiry then strike.
    Only rows present in BOTH snapshots are kept (so a change can be computed).
    """
    a = load_premium_snapshot(day_today)
    b = load_premium_snapshot(day_ref)
    if a.empty or b.empty:
        return pd.DataFrame()
    keys = ['expiry', 'strike']
    cols = ['call_mid', 'call_iv', 'put_mid', 'put_iv']
    merged = a.merge(b[keys + cols], on=keys, suffixes=('', '_ref'), how='inner')
    if merged.empty:
        return merged
    for c in cols:
        merged[f'{c}_chg'] = merged[c] - merged[f'{c}_ref']
    merged = merged.sort_values(['T_days', 'strike']).reset_index(drop=True)
    return merged


def sunny_money_chart(snapshot_df: pd.DataFrame, expiry: str, mode: str,
                      compare_df: pd.DataFrame = None, spot: float = None,
                      highlight_strikes: list = None):
    """Sunny-Money-style bar chart: one bar per strike for a single expiry.

    mode:
      'vol_call' / 'vol_put'   — implied vol per strike (the skew/smile)
      'prem_call' / 'prem_put' — premium per strike
      'diff_vol_call' / 'diff_vol_put'   — IV change vs compare_df (yellow bars)
      'diff_prem_call' / 'diff_prem_put' — premium % change vs compare_df

    compare_df is required only for the 'diff_*' modes. highlight_strikes are
    drawn red/green (like the coloured bars in Sunny Money) to mark levels of
    interest. Mirrors the reference tool's four view modes.
    """
    import plotly.graph_objects as go
    sub = snapshot_df[snapshot_df['expiry'] == expiry].copy().sort_values('strike')
    if sub.empty:
        return empty_fig("Nessun dato per questa scadenza")

    is_diff = mode.startswith('diff_')
    is_vol = 'vol' in mode
    is_call = mode.endswith('_call')
    side = 'call' if is_call else 'put'

    # Build y-values per mode
    if not is_diff:
        col = f'{side}_iv' if is_vol else f'{side}_mid'
        y = sub[col].astype(float)
        if is_vol:
            y = y * 100.0            # IV in %
        ylab = 'Volatilità implicita (%)' if is_vol else 'Premio (punti)'
        bar_color = ACCENT_YLW if is_vol else ('#3B82F6' if is_call else '#8B5CF6')
    else:
        if compare_df is None or compare_df.empty:
            return empty_fig("Serve una data di confronto")
        merged = sub.merge(
            compare_df[compare_df['expiry'] == expiry][['strike', f'{side}_iv', f'{side}_mid']],
            on='strike', suffixes=('', '_ref'), how='inner').sort_values('strike')
        if merged.empty:
            return empty_fig("Nessuno strike in comune con la data di confronto")
        sub = merged
        if is_vol:
            y = (merged[f'{side}_iv'].astype(float) - merged[f'{side}_iv_ref'].astype(float)) * 100.0
            ylab = 'Δ Volatilità (punti %)'
        else:
            base = merged[f'{side}_mid_ref'].astype(float).replace(0, np.nan)
            y = (merged[f'{side}_mid'].astype(float) - merged[f'{side}_mid_ref'].astype(float)) / base * 100.0
            ylab = 'Δ Premio (%)'
        bar_color = '#EAB308'       # Sunny-Money yellow for differentials

    strikes = sub['strike'].astype(float).tolist()
    yv = y.tolist()

    # Per-bar colours: highlights override, else base colour
    hl = set(highlight_strikes or [])
    colors = []
    for i, k in enumerate(strikes):
        if k in hl:
            colors.append(ACCENT_GRN)
        else:
            colors.append(bar_color)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=strikes, y=yv, marker_color=colors,
                         text=[f"{v:.2f}" if abs(v) >= 0.01 else "" for v in yv],
                         textposition='outside', textfont=dict(size=8),
                         name=ylab))
    if spot:
        fig.add_vline(x=spot, line_color='#111827', line_dash='dash',
                      annotation_text=f' spot {spot:.0f}', annotation_font_size=9)
    _mode_names = {
        'vol_call': 'Volatilità CALL', 'vol_put': 'Volatilità PUT',
        'prem_call': 'Premi CALL', 'prem_put': 'Premi PUT',
        'diff_vol_call': 'Diff. Vol CALL', 'diff_vol_put': 'Diff. Vol PUT',
        'diff_prem_call': 'Diff. Premi CALL', 'diff_prem_put': 'Diff. Premi PUT',
    }
    layout = base_layout(f"{_mode_names.get(mode, mode)} — {expiry}")
    layout.update(yaxis_title=ylab, xaxis_title='Strike', showlegend=False)
    fig.update_layout(**layout)
    return fig
