"""
Test suite for dex_gex_dashboard.py — run with: pytest test_dashboard.py -v
Covers: BS greeks, aggregations, GEX analytics, RVol models, edge cases.
"""
import numpy as np
import pandas as pd
import pytest
from datetime import date, timedelta

import dex_gex_dashboard as dg

SPOT = 5500.0
R    = 0.053


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def chain():
    """Synthetic 30-DTE chain around spot."""
    np.random.seed(42)
    exp = (date.today() + timedelta(days=30)).strftime('%Y-%m-%d')
    rows = []
    for K in np.arange(5300, 5701, 5, dtype=float):
        for flag in ['c', 'p']:
            iv = 0.16 + 0.03 * abs(K - SPOT) / SPOT
            g  = dg.bs_gamma(SPOT, K, 30/365, R, iv)
            d  = dg.bs_delta(SPOT, K, 30/365, R, iv, flag)
            oi = float(np.random.randint(100, 3000))
            sign = 1 if flag == 'c' else -1
            rows.append({'strike': K, 'expiry': exp, 'flag': flag,
                         'delta': d, 'gamma': g, 'iv': iv,
                         'openInterest': oi, 'mid': 5.0,
                         'T_days': 30, 'T_years': 30/365,
                         'spot': SPOT, 'q_impl': 0.014,
                         'dex': d*oi*100, 'gex': sign*g*oi*100*SPOT,
                         'gross_gex': g*oi*100*SPOT,
                         'charm': 0.0, 'vanna': 0.0,
                         'charm_exp': 0.0, 'vanna_exp': 0.0})
    return pd.DataFrame(rows)


@pytest.fixture
def ohlc():
    np.random.seed(7)
    n = 250
    dates = pd.date_range('2025-06-01', periods=n, freq='B')
    close = 5000 + np.cumsum(np.random.randn(n) * 15)
    return pd.DataFrame({
        'date': dates,
        'open':  close + np.random.randn(n) * 5,
        'high':  close + np.abs(np.random.randn(n) * 12),
        'low':   close - np.abs(np.random.randn(n) * 12),
        'close': close, 'volume': 1e6,
    })


# ── Black-Scholes-Merton ──────────────────────────────────────────────────────
class TestBSM:
    def test_delta_bounds(self):
        for K in [5000, 5500, 6000]:
            dc = dg.bs_delta(SPOT, K, 30/365, R, 0.16, 'c')
            dp = dg.bs_delta(SPOT, K, 30/365, R, 0.16, 'p')
            assert 0 <= dc <= 1
            assert -1 <= dp <= 0
            # Put-call delta parity (with dividend): dc - dp = e^{-qT}
            assert abs((dc - dp) - np.exp(-dg.SPX_DIV_YIELD * 30/365)) < 1e-9

    def test_dividend_reduces_call_delta(self):
        d_q  = dg.bs_delta(SPOT, SPOT, 30/365, R, 0.16, 'c')
        d_0  = dg.bs_delta(SPOT, SPOT, 30/365, R, 0.16, 'c', q=0.0)
        assert d_q < d_0

    def test_gamma_positive_and_peaks_atm(self):
        g_atm = dg.bs_gamma(SPOT, SPOT, 30/365, R, 0.16)
        g_otm = dg.bs_gamma(SPOT, SPOT*1.05, 30/365, R, 0.16)
        assert g_atm > g_otm > 0

    def test_zero_T_safe(self):
        assert dg.bs_delta(SPOT, SPOT, 0, R, 0.16, 'c') == 0.0
        assert dg.bs_gamma(SPOT, SPOT, 0, R, 0.16) == 0.0
        assert dg.bs_charm(SPOT, SPOT, 0, R, 0.16, 'c') == 0.0
        assert dg.bs_vanna(SPOT, SPOT, 0, R, 0.16) == 0.0
        assert dg.bs_rho(SPOT, SPOT, 0, R, 0.16, 'c') == 0.0

    def test_rho_signs(self):
        assert dg.bs_rho(SPOT, SPOT, 60/365, R, 0.16, 'c') > 0
        assert dg.bs_rho(SPOT, SPOT, 60/365, R, 0.16, 'p') < 0

    def test_greeks_finite(self):
        for fn in [lambda: dg.bs_charm(SPOT, SPOT, 1/8760, R, 0.18, 'c'),
                   lambda: dg.bs_vanna(SPOT, SPOT, 1/8760, R, 0.18),
                   lambda: dg.bs_speed(SPOT, SPOT, 30/365, R, 0.16)]:
            assert np.isfinite(fn())


# ── Aggregation ───────────────────────────────────────────────────────────────
class TestAggregation:
    def test_by_strike_columns(self, chain):
        bs = dg.aggregate_by_strike(chain)
        for col in ['strike', 'net_gex', 'net_dex', 'call_dex', 'put_dex',
                    'total_oi', 'gross_gex']:
            assert col in bs.columns

    def test_totals_match(self, chain):
        bs = dg.aggregate_by_strike(chain)
        assert abs(bs['net_gex'].sum() - chain['gex'].sum()) < 1e-6
        assert abs(bs['net_dex'].sum() - chain['dex'].sum()) < 1e-6

    def test_empty_chain(self):
        empty = pd.DataFrame(columns=['strike','flag','dex','gex','openInterest'])
        bs = dg.aggregate_by_strike(empty)
        assert bs.empty


# ── GEX analytics ─────────────────────────────────────────────────────────────
class TestGEXAnalytics:
    def test_analytics_keys(self, chain):
        bs = dg.aggregate_by_strike(chain)
        a  = dg.compute_gex_analytics(chain, bs, SPOT)
        for k in ['center_of_mass', 'hhi', 'flip_zone_lo', 'flip_zone_hi',
                  'impact_1pct', 'top3_strikes']:
            assert k in a

    def test_hhi_bounds(self, chain):
        bs = dg.aggregate_by_strike(chain)
        a  = dg.compute_gex_analytics(chain, bs, SPOT)
        assert 0 <= a['hhi'] <= 1

    def test_profile_levels(self, chain):
        p = dg.compute_gex_profile(chain, SPOT)
        assert len(p) == 11
        assert set(p['regime'].unique()) <= {'LONG γ', 'SHORT γ'}

    def test_flip_by_expiry(self, chain):
        ft = dg.compute_flip_by_expiry(chain, SPOT)
        assert not ft.empty
        assert 'flip' in ft.columns

    def test_empty_analytics(self):
        assert dg.compute_gex_analytics(pd.DataFrame(), pd.DataFrame(), SPOT) == {}


# ── Max pain / PC ratio ──────────────────────────────────────────────────────
class TestMaxPainPC:
    def test_max_pain_in_range(self, chain):
        mp = dg.compute_max_pain(chain)
        assert mp
        for exp, k in mp.items():
            assert 5300 <= k <= 5700

    def test_pc_ratio(self, chain):
        pc = dg.compute_pc_ratio(chain)
        assert pc['ratio'] is not None and pc['ratio'] > 0


# ── RVol models ───────────────────────────────────────────────────────────────
class TestRVol:
    def test_all_models_present(self, ohlc):
        rv = dg.compute_rvol_all(ohlc, window=126)
        expected = {'Std Dev', 'Parkinson', 'Garman-Klass', 'Hodges-Tompkins',
                    'Rogers-Satchell', 'Yang-Zhang', 'EWMA λ=0.94',
                    'Yang-Zhang 5d', 'Yang-Zhang 21d'}
        assert expected <= set(rv.columns)

    def test_values_positive(self, ohlc):
        rv = dg.compute_rvol_all(ohlc, window=126)
        assert (rv > 0).all().all()
        assert (rv < 200).all().all()

    def test_ewma_more_reactive(self, ohlc):
        rv = dg.compute_rvol_all(ohlc, window=126)
        assert rv['EWMA λ=0.94'].std() > rv['Yang-Zhang'].std()

    def test_har_with_ci(self, ohlc):
        h = dg.compute_har_rv(ohlc)
        assert 'forecast' in h and 'ci_68_lo' in h and 'ci_68_hi' in h
        assert h['ci_68_lo'] <= h['forecast'] <= h['ci_68_hi']

    def test_har_oos(self, ohlc):
        bt = dg.backtest_har_oos(ohlc)
        assert bt and bt['n_oos'] > 0 and bt['rmse'] > 0

    def test_cones(self, ohlc):
        c = dg.compute_rvol_cones(ohlc)
        assert not c.empty and 'current' in c.columns

    def test_regime(self, ohlc):
        yz = dg.rvol_yang_zhang(ohlc.set_index('date'), 30) * 100
        regime, z = dg.detect_vol_regime(yz)
        assert regime in ('HIGH', 'NORMAL', 'LOW')

    def test_short_history(self):
        tiny = pd.DataFrame({'date': pd.date_range('2026-01-01', periods=5),
                              'open': [1]*5, 'high': [1]*5,
                              'low': [1]*5, 'close': [1]*5, 'volume': [1]*5})
        assert dg.compute_rvol_all(tiny).empty
        assert dg.compute_har_rv(tiny) == {}


# ── IV / dividend yield ───────────────────────────────────────────────────────
class TestIVDiv:
    def test_validate_iv_removes_outliers(self):
        df = pd.DataFrame({'expiry': 'e1',
                           'iv': [0.001]*3 + [10.0]*2 + [0.18]*95})
        out = dg.validate_iv(df)
        assert len(out) < len(df)
        assert (out['iv'] >= 0.005).all()

    def test_implied_div_yield_bounds(self):
        rows = []
        for K in np.arange(5450, 5551, 5, dtype=float):
            C = max(SPOT - K + 30, 0.5)
            P = max(K - SPOT + 30, 0.5)
            for flag, mid in [('c', C), ('p', P)]:
                rows.append({'strike': K, 'expiry': 'e1', 'flag': flag,
                             'mid': mid, 'T_years': 30/365})
        q = dg.compute_implied_div_yield(pd.DataFrame(rows), SPOT, R)
        assert 'e1' in q and -0.01 <= q['e1'] <= 0.06


# ── Snapshots / DoD ───────────────────────────────────────────────────────────
class TestSnapshots:
    def test_save_and_load(self, chain, tmp_path, monkeypatch):
        monkeypatch.setattr(dg, 'SNAPSHOT_DIR', str(tmp_path))
        p = dg.save_daily_snapshot(chain, SPOT, 'TEST')
        assert p and 'TEST_' in p
        # Same-day snapshot isn't "previous", so this returns nothing
        bs, meta, d = dg.load_previous_snapshot('TEST')
        assert bs is None

    def test_dod_no_history(self, chain, tmp_path, monkeypatch):
        monkeypatch.setattr(dg, 'SNAPSHOT_DIR', str(tmp_path))
        assert dg.compute_dod_changes(chain, SPOT, 'TEST') == {}


# ── Alert flags ───────────────────────────────────────────────────────────────
class TestAlerts:
    def test_six_flags(self):
        flags = dg.build_alert_flags(
            pd.DataFrame(), SPOT,
            {'gex_flip': SPOT*1.01, 'pc_ratio': 1.0, 'atm_iv': 0.15,
             'exp_move_pct': 1.0, 'vanna_exp': 1e9},
            None, None)
        assert len(flags) == 6
        assert all(f['status'] in ('RED', 'AMBER', 'GREEN', 'GREY')
                   for f in flags)

    def test_short_gamma_red(self):
        flags = dg.build_alert_flags(
            pd.DataFrame(), SPOT,
            {'gex_flip': SPOT*1.02, 'pc_ratio': None, 'atm_iv': None,
             'exp_move_pct': None, 'vanna_exp': None},
            None, None)
        gamma = next(f for f in flags if f['name'] == 'Gamma Regime')
        assert gamma['status'] == 'RED'
