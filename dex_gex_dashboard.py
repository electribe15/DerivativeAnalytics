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
OI_THRESHOLD      = 100     # Minimum open interest to keep a strike
FETCH_EXPIRY_DAYS = 270     # Max DTE when downloading the chain.
                            # 270d (9 months) gives a full IV term structure while
                            # staying clear of LEAP expirations (>9m) where Barchart's
                            # API becomes unreliable for $SPX (slow response, partial
                            # JSON, or server-side hangs that bypass read timeouts).
MAX_EXPIRY_DAYS   = 90      # Default display window (sidebar DTE filter, post-fetch)
PORT              = 8051    # Browser port (safe default)

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


def bs_greeks_vectorized(S, K, T, r, sigma, flag):
    """Vectorized Black-Scholes delta and gamma."""
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    flag = np.asarray(flag)

    delta = np.zeros_like(K, dtype=float)
    gamma = np.zeros_like(K, dtype=float)

    valid = (T > 0) & (sigma > 0) & np.isfinite(K) & np.isfinite(T) & np.isfinite(sigma)
    if not np.any(valid):
        return delta, gamma

    sqrt_T = np.sqrt(T[valid])
    d1 = (np.log(S / K[valid]) + (r + 0.5 * sigma[valid] ** 2) * T[valid]) / (sigma[valid] * sqrt_T)
    cdf_d1 = norm.cdf(d1)

    delta_valid = np.where(flag[valid] == 'c', cdf_d1, cdf_d1 - 1.0)
    gamma_valid = norm.pdf(d1) / (S * sigma[valid] * sqrt_T)

    delta[valid] = delta_valid
    gamma[valid] = gamma_valid
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
BARCHART_API_URL = 'https://www.barchart.com/proxies/core-api/v1/options/get'
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
        if 1 <= dte <= max_dte:
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
_FETCH_TIMEOUT    = 15    # seconds per request (reduced: API calls should be fast;
                          # timeouts mean Barchart is slow, not that session expired)


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
    data = data[data['T_days'] >= 1].copy()          # drop expired only
    data['T_years'] = data['T_days'] / 365.0
    data['spot']    = S
    data['mid']     = data[['bid', 'ask']].mean(axis=1, skipna=True)
    data['flag']    = data['optionType'].map({'call': 'c', 'put': 'p'})
    data['iv']      = data['impliedVolatility'] / 100.0

    # No OI or DTE filter here — full dataset goes to CSV
    data = data[data['mid'].notna()].copy()
    data.dropna(subset=['iv'], inplace=True)
    data = data[data['iv'] > 0].copy()

    notify(93, f'Computing Greeks for {ticker} ...')
    delta_calc, gamma_calc = bs_greeks_vectorized(
        S,
        data['strike'].to_numpy(),
        data['T_years'].to_numpy(),
        r,
        data['iv'].to_numpy(),
        data['flag'].to_numpy(),
    )
    delta_available = data['delta_barchart'].notna()
    data['delta'] = np.where(delta_available, data['delta_barchart'], delta_calc)
    data['gamma'] = gamma_calc
    data['dex']   = data['delta'] * data['openInterest'] * 100
    sign          = data['flag'].map({'c': 1, 'p': -1})
    data['gex']   = sign * data['gamma'] * data['openInterest'] * 100 * S

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
    """Roll up DEX and GEX to per-strike totals — fully vectorised.

    The previous implementation used per-group lambdas that re-indexed into
    the full DataFrame on every group (O(n × n_groups)).  With 16 k contracts
    across ~200 strikes that ran millions of index lookups on every 1.5-second
    interval tick.  This version uses three plain groupby-sum calls and a join,
    which is O(n) total and completes in single-digit milliseconds.
    """
    calls  = (data[data['flag'] == 'c']
               .groupby('strike', sort=True)
               .agg(call_dex=('dex', 'sum'), call_gex=('gex', 'sum')))
    puts   = (data[data['flag'] == 'p']
               .groupby('strike', sort=True)
               .agg(put_dex=('dex', 'sum'), put_gex=('gex', 'sum')))
    totals = (data.groupby('strike', sort=True)
               .agg(net_dex=('dex', 'sum'), net_gex=('gex', 'sum'),
                    total_oi=('openInterest', 'sum')))
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


def gex_bar_chart(by_strike_df, spot, ticker, window_pct=0.10,
                  strike_lo=None, strike_hi=None):
    """GEX by strike bar chart.

    Renders ALL strikes in by_strike_df and uses xaxis.range to set the
    initial zoom window.  This way the chart is never empty regardless of
    the delta range selected: the user can always zoom out to see the full
    picture.
    """
    import plotly.graph_objects as go

    # Determine the initial view window from delta bounds (or ±window_pct fallback)
    lo = strike_lo if strike_lo is not None else spot * (1 - window_pct)
    hi = strike_hi if strike_hi is not None else spot * (1 + window_pct)
    if lo >= hi:                          # safety: inversion or zero range
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
    layout = base_layout(f'GEX by Strike — {ticker}')
    layout['xaxis'].update(range=[lo, hi])
    layout.update(yaxis_title='GEX ($M)', xaxis_title='Strike')
    fig.update_layout(**layout)
    return fig


def dex_bar_chart(by_strike_df, spot, ticker, window_pct=0.10,
                  strike_lo=None, strike_hi=None):
    """DEX by strike chart (call / put / net).

    Same approach as gex_bar_chart: all data rendered, xaxis.range for zoom.
    """
    import plotly.graph_objects as go

    lo = strike_lo if strike_lo is not None else spot * (1 - window_pct)
    hi = strike_hi if strike_hi is not None else spot * (1 + window_pct)
    if lo >= hi:
        lo, hi = spot * (1 - window_pct), spot * (1 + window_pct)

    df = by_strike_df.copy()
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
    layout = base_layout(f'DEX by Strike — {ticker}')
    layout['xaxis'].update(range=[lo, hi])
    layout.update(barmode='relative', yaxis_title='DEX ($M)', xaxis_title='Strike')
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

            with store_lock:
                store['raw'] = raw_df
                store['by_strike'] = None
                store['by_expiry'] = None
                store['spot'] = spot
                store['ticker'] = resolved_barchart_ticker
                store['updated_at'] = datetime.now()

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
            return ([], empty, empty, empty, empty, empty,
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
                [], empty, empty, empty, empty, empty,
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
                [], empty, empty, empty, empty, empty,
                status,
                loader['progress'], f"{loader['progress']}%",
                progress_color, progress_animated, progress_striped,
                '',
            )

        bs_df = aggregate_by_strike(filtered_df)
        be_df = aggregate_by_expiry(filtered_df)

        lo, hi = delta_strike_bounds(raw_df, delta_lo, delta_hi)
        stats   = build_stats(filtered_df, bs_df, spot)
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
            stats, fig_gex, fig_dex, fig_exp, fig_oi, fig_vol,
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
