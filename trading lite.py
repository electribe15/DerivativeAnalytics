"""
Trading Lite — Simulation-Only Long-Vol Engine
===============================================
A compact, single-file trading engine for the dashboard-only deployment.
NO Interactive Brokers, NO external connections — pure simulation.

It reuses the data the dashboard already computes (GEX regime, HHI, vol
premium, IV, expected move) and replicates the long-volatility decision
logic developed in the full system, in a simplified form suitable for
in-process use on Streamlit Cloud.

What it does
------------
1. Evaluates the long-vol signal from the current chain analytics
2. On a TRADE signal, records a SIMULATED paper position (no real order)
3. Marks open positions to a simple model value each time it runs
4. Closes positions on profit target / stop / DTE floor
5. Persists everything to paper_trades/ as CSV (survives restarts when
   committed by GitHub Actions, exactly like snapshots)

Philosophy (unchanged from the full system)
-------------------------------------------
- Long vol: risk a small defined premium, asymmetric payoff
- Most days = SKIP (selectivity is the edge)
- Process over profit: this is for validation, not income
- Signal from SPX, sizing for a small (5k) paper account

All state is CSV-based. No database, no IB, no background process.
"""
from __future__ import annotations
import os
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from typing import Optional

import numpy as np
import pandas as pd

# ── Persistence ────────────────────────────────────────────────────────────────
PAPER_DIR        = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'paper_trades')
POSITIONS_CSV    = os.path.join(PAPER_DIR, 'positions.csv')
CONTEXT_CSV      = os.path.join(PAPER_DIR, 'signal_context.csv')

# ── Account / risk (5k paper account) ──────────────────────────────────────────
INITIAL_CAPITAL      = 5000.0
MAX_PREMIUM_PCT      = 2.0     # max premium per trade = 2% of account
MAX_DAILY_LOSS       = 250.0   # circuit breaker (informational in sim)
MAX_DRAWDOWN_PCT     = 50.0    # hard stop (the 50% rule)
SPX_XSP_RATIO        = 10.0    # signal SPX → execution XSP scale

# ── Long-vol selection thresholds (mirror of long_vol_selector.py) ──────────────
IV_CHEAP_BELOW       = 16.0
IV_SKIP_ABOVE        = 28.0
VP_BUY_BELOW         = 2.0
DTE_TARGET           = 45
PROFIT_TARGET_MULT   = 2.5
STOP_LOSS_PCT        = 50.0
MIN_CONFIDENCE       = 0.45
HOLD_DTE_FLOOR       = 14

# Risk-free + dividend (match dashboard)
R_FREE               = 0.053221
Q_DIV                = 0.014


# ══════════════════════════════════════════════════════════════════════════════
# Black-Scholes pricer (self-contained — no dashboard import needed)
# ══════════════════════════════════════════════════════════════════════════════
def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(S, K, T, r, sigma, flag, q=Q_DIV):
    """Black-Scholes-Merton option price (per share)."""
    if T <= 0:
        return max(S - K, 0.0) if flag == 'c' else max(K - S, 0.0)
    if sigma <= 0:
        fwd, kpv = S * math.exp(-q * T), K * math.exp(-r * T)
        return max(fwd - kpv, 0.0) if flag == 'c' else max(kpv - fwd, 0.0)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    dq, dr = math.exp(-q * T), math.exp(-r * T)
    if flag == 'c':
        return S * dq * _norm_cdf(d1) - K * dr * _norm_cdf(d2)
    return K * dr * _norm_cdf(-d2) - S * dq * _norm_cdf(-d1)


def straddle_price(S, K, T, sigma):
    return bs_price(S, K, T, R_FREE, sigma, 'c') + bs_price(S, K, T, R_FREE, sigma, 'p')


# ══════════════════════════════════════════════════════════════════════════════
# Decision
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class LiteDecision:
    action:      str = 'SKIP'        # 'TRADE' | 'SKIP'
    strategy:    str = ''
    confidence:  float = 0.0
    target_dte:  int = DTE_TARGET
    strike:      float = 0.0          # signal-space (SPX) ATM strike
    xsp_strike:  float = 0.0          # execution-space (XSP) strike
    est_premium: float = 0.0          # estimated premium in points (XSP)
    factors:     list = field(default_factory=list)
    conditions:  list = field(default_factory=list)   # per-condition verdicts
    explanation: str = ''             # human-readable why TRADE / why SKIP
    skip_reason: str = ''

    @property
    def summary_html(self) -> str:
        if self.action == 'SKIP':
            return f"<b style='color:#9CA3AF'>⏸ SKIP</b> — {self.skip_reason}"
        color = '#10B981' if self.confidence > 0.6 else '#F59E0B'
        strat = {'long_straddle': '🎯 Long Straddle',
                 'long_strangle': '🎲 Long Strangle'}.get(self.strategy, self.strategy)
        return (f"<b style='color:{color}'>{strat}</b> · "
                f"DTE {self.target_dte} · XSP strike {self.xsp_strike:.0f} · "
                f"premio stim. {self.est_premium:.2f} · "
                f"confidence <b>{int(self.confidence*100)}%</b>")


def evaluate_signal(spot: float, regime: str, hhi: float,
                    vol_premium: Optional[float], atm_iv: Optional[float],
                    expected_move: Optional[float]) -> LiteDecision:
    """Long-vol decision from the analytics the dashboard already computes.

    Simplified mirror of LongVolSelector.select() — uses the core signals
    available without market_context (IV level, VP, regime, HHI).
    """
    d = LiteDecision()
    iv = (atm_iv or 0.18) * 100 if (atm_iv and atm_iv < 1) else (atm_iv or 18.0)
    vp = vol_premium if vol_premium is not None else 5.0
    factors = []
    conditions = []   # (label, met:bool, detail)

    def cond(label, met, detail):
        conditions.append({'label': label, 'met': bool(met), 'detail': detail})

    # Hard skip: vol too expensive to buy
    if iv > IV_SKIP_ABOVE:
        cond("Vol acquistabile", False,
             f"IV {iv:.1f}% > {IV_SKIP_ABOVE}% (soglia max) — vol troppo cara")
        d.skip_reason = f"IV {iv:.1f}% > {IV_SKIP_ABOVE}% — vol troppo cara da comprare"
        d.conditions = conditions
        d.explanation = (
            f"❌ NESSUN TRADE. La volatilità implicita ({iv:.1f}%) è sopra la soglia "
            f"massima di acquisto ({IV_SKIP_ABOVE}%). Comprare vol così cara elimina il "
            f"vantaggio della strategia long-vol: si paga troppo premio. Il sistema "
            f"aspetta che l'IV scenda prima di considerare un acquisto.")
        return d

    score = 0.0
    # 1. Cheap IV (0-0.35)
    if iv < IV_CHEAP_BELOW:
        score += min((IV_CHEAP_BELOW - iv) / IV_CHEAP_BELOW, 1.0) * 0.35
        factors.append(f"IV {iv:.1f}% &lt; {IV_CHEAP_BELOW}% — vol economica")
        cond("IV economica", True,
             f"IV {iv:.1f}% < {IV_CHEAP_BELOW}% — vol a buon mercato (+0.35 max)")
    elif iv < 20:
        score += 0.10
        factors.append(f"IV {iv:.1f}% — moderata")
        cond("IV economica", False,
             f"IV {iv:.1f}% moderata (tra {IV_CHEAP_BELOW}% e 20%) — solo +0.10")
    else:
        cond("IV economica", False,
             f"IV {iv:.1f}% non economica (≥ 20%) — nessun bonus")

    # 2. Low/negative vol premium (0-0.25)
    if vp < VP_BUY_BELOW:
        score += 0.25
        factors.append(f"Vol Premium {vp:+.1f}% basso — IV sottovalutata vs RVol")
        cond("Vol Premium basso", True,
             f"VP {vp:+.1f}% < {VP_BUY_BELOW}% — IV sottovalutata vs realizzata (+0.25)")
    elif vp < 4:
        score += 0.10
        cond("Vol Premium basso", False,
             f"VP {vp:+.1f}% intermedio (tra {VP_BUY_BELOW}% e 4%) — solo +0.10")
    else:
        cond("Vol Premium basso", False,
             f"VP {vp:+.1f}% alto (≥ 4%) — l'IV è cara vs la realizzata, sfavorevole")

    # 3. SHORT gamma regime (0-0.25)
    if regime == 'SHORT':
        score += 0.25
        factors.append("Regime SHORT γ — i dealer amplificano i movimenti")
        cond("Regime SHORT gamma", True,
             "Dealer SHORT gamma — amplificano i movimenti, favorevole per long-vol (+0.25)")
    else:
        cond("Regime SHORT gamma", False,
             "Regime LONG γ — i dealer stabilizzano il mercato, sfavorevole per long-vol")
        factors.append("Regime LONG γ — i dealer stabilizzano (sfavorevole)")

    # 4. Low HHI bonus / high HHI penalty (pinning is bad for long vol)
    if hhi >= 0.06:
        score -= 0.12
        factors.append(f"HHI {hhi:.3f} alto — pinning, il prezzo resta fermo (sfavorevole)")
        cond("Niente pinning forte", False,
             f"HHI {hhi:.3f} ≥ 0.06 — GEX concentrato, il prezzo tende a restare fermo (−0.12)")
    elif hhi < 0.02:
        score += 0.08
        factors.append(f"HHI {hhi:.3f} basso — niente pinning forte")
        cond("Niente pinning forte", True,
             f"HHI {hhi:.3f} < 0.02 — GEX distribuito, il prezzo può muoversi (+0.08)")
    else:
        cond("Niente pinning forte", False,
             f"HHI {hhi:.3f} intermedio (tra 0.02 e 0.06) — nessun bonus né penalità")

    score = max(0.0, min(1.0, score))
    d.confidence = round(score, 3)
    d.conditions = conditions

    _met = [c for c in conditions if c['met']]
    _unmet = [c for c in conditions if not c['met']]

    if score < MIN_CONFIDENCE:
        d.skip_reason = (f"Conviction {score:.2f} &lt; {MIN_CONFIDENCE} — "
                         "condizioni non abbastanza favorevoli per comprare vol")
        # Build explanation of what was missing
        missing = "; ".join(c['label'] for c in _unmet) or "nessuna condizione chiave"
        present = "; ".join(c['label'] for c in _met) or "nessuna"
        d.explanation = (
            f"❌ NESSUN TRADE. Punteggio di convinzione {score:.2f}, sotto la soglia "
            f"minima di {MIN_CONFIDENCE}. Per comprare volatilità servono abbastanza "
            f"condizioni favorevoli allineate.\n\n"
            f"✓ Condizioni soddisfatte: {present}.\n"
            f"✗ Condizioni mancanti: {missing}.\n\n"
            f"Questo è il comportamento corretto: il long-vol entra solo quando "
            f"l'occasione è chiara (vol economica + regime favorevole + niente pinning). "
            f"La maggior parte dei giorni il sistema resta in attesa.")
        return d

    # TRADE — choose strategy
    strategy = 'long_straddle' if (iv < IV_CHEAP_BELOW and score > 0.6) else 'long_strangle'
    em_mult  = 0.0 if strategy == 'long_straddle' else 1.0

    # Strike in signal (SPX) space
    em   = expected_move or (spot * (iv / 100) / math.sqrt(252) * math.sqrt(DTE_TARGET))
    strike_spx = round((spot + em_mult * em) / 5) * 5

    # Translate to XSP execution space
    xsp_strike = round((strike_spx / SPX_XSP_RATIO) / 5) * 5
    xsp_spot   = spot / SPX_XSP_RATIO
    T          = DTE_TARGET / 365.0
    est_prem   = straddle_price(xsp_spot, xsp_strike, T, iv / 100)

    _strat_name = {'long_straddle': 'Long Straddle (ATM)',
                   'long_strangle': 'Long Strangle (OTM)'}.get(strategy, strategy)
    present = "; ".join(c['label'] for c in _met) or "nessuna"
    d.explanation = (
        f"✅ TRADE: {_strat_name}. Punteggio di convinzione {score:.2f}, sopra la "
        f"soglia minima di {MIN_CONFIDENCE}. Le condizioni favorevoli si sono allineate.\n\n"
        f"✓ Condizioni che hanno guidato la decisione: {present}.\n\n"
        f"Strategia scelta: {'straddle ATM' if strategy=='long_straddle' else 'strangle OTM'} "
        f"perché {'IV molto economica e convinzione alta — si compra al denaro per la massima sensibilità al movimento' if strategy=='long_straddle' else 'convinzione moderata — strangle OTM più economico, serve un movimento più ampio'}. "
        f"Strike XSP {xsp_strike:.0f}, scadenza {DTE_TARGET} giorni, premio stimato "
        f"{est_prem:.2f} punti (~${est_prem*100:.0f}). Perdita massima limitata al premio pagato.")

    d.action      = 'TRADE'
    d.strategy    = strategy
    d.strike      = strike_spx
    d.xsp_strike  = xsp_strike
    d.est_premium = round(est_prem, 2)
    d.factors     = factors
    return d


# ══════════════════════════════════════════════════════════════════════════════
# Paper position store (CSV-based)
# ══════════════════════════════════════════════════════════════════════════════
_POS_COLS = ['id', 'open_date', 'mode', 'strategy', 'signal_spot', 'xsp_spot',
             'xsp_strike', 'dte_target', 'entry_iv', 'entry_premium',
             'n_contracts', 'confidence', 'status', 'close_date',
             'exit_premium', 'pnl_usd', 'exit_reason',
             'regime', 'vol_premium', 'hhi', 'decision_note']


def _ensure_dir():
    os.makedirs(PAPER_DIR, exist_ok=True)


def load_positions() -> pd.DataFrame:
    _ensure_dir()
    if os.path.exists(POSITIONS_CSV):
        try:
            return pd.read_csv(POSITIONS_CSV)
        except Exception:
            pass
    return pd.DataFrame(columns=_POS_COLS)


def _save_positions(df: pd.DataFrame):
    _ensure_dir()
    df.to_csv(POSITIONS_CSV, index=False)


def import_positions(uploaded_df: pd.DataFrame, mode: str = 'merge') -> dict:
    """Restore paper positions from an exported CSV.

    Use this after a Streamlit Cloud restart to reload your history.

    mode:
      'merge'   — keep existing rows, add uploaded rows whose id is not present
                  (uploaded rows win on id conflict only if existing is OPEN and
                   uploaded is CLOSED, i.e. uploaded carries newer outcome)
      'replace' — discard current store, use the uploaded CSV as-is

    Returns a summary dict: {imported, skipped, total, errors}.
    """
    summary = {'imported': 0, 'skipped': 0, 'total': 0, 'errors': []}

    # Validate columns
    missing = [c for c in ('id', 'status') if c not in uploaded_df.columns]
    if missing:
        summary['errors'].append(f"Colonne mancanti: {missing}")
        return summary

    # Normalise to expected schema (fill any missing optional columns)
    up = uploaded_df.copy()
    for c in _POS_COLS:
        if c not in up.columns:
            up[c] = ''
    up = up[_POS_COLS]

    if mode == 'replace':
        _save_positions(up)
        summary['imported'] = len(up)
        summary['total']    = len(up)
        return summary

    # merge
    cur = load_positions()
    if cur.empty:
        _save_positions(up)
        summary['imported'] = len(up)
        summary['total']    = len(up)
        return summary

    existing_by_id = {int(r['id']): r for _, r in cur.iterrows()
                      if pd.notna(r.get('id'))}
    rows = list(cur.to_dict('records'))

    for _, r in up.iterrows():
        try:
            rid = int(r['id'])
        except Exception:
            summary['skipped'] += 1
            continue
        if rid in existing_by_id:
            ex = existing_by_id[rid]
            # Uploaded row carries a newer outcome (CLOSED) over an OPEN one
            if str(ex.get('status')) == 'OPEN' and str(r.get('status')) == 'CLOSED':
                for i, rr in enumerate(rows):
                    if int(rr.get('id', -1)) == rid:
                        rows[i] = r.to_dict()
                        summary['imported'] += 1
                        break
            else:
                summary['skipped'] += 1
        else:
            rows.append(r.to_dict())
            summary['imported'] += 1

    merged = pd.DataFrame(rows, columns=_POS_COLS)
    # Drop exact duplicate ids keeping the last (CLOSED wins via above logic)
    merged = merged.drop_duplicates(subset='id', keep='last').reset_index(drop=True)
    _save_positions(merged)
    summary['total'] = len(merged)
    return summary


def open_paper_position(decision: LiteDecision, spot: float,
                        regime: str, vol_premium: float, hhi: float,
                        atm_iv: float) -> int:
    """Record a simulated long-vol position. Returns the new id."""
    df = load_positions()
    new_id = (int(df['id'].max()) + 1) if not df.empty else 1
    xsp_spot = spot / SPX_XSP_RATIO
    row = {
        'id': new_id, 'open_date': date.today().isoformat(), 'mode': 'long_vol',
        'strategy': decision.strategy, 'signal_spot': spot, 'xsp_spot': xsp_spot,
        'xsp_strike': decision.xsp_strike, 'dte_target': decision.target_dte,
        'entry_iv': (atm_iv * 100 if atm_iv and atm_iv < 1 else atm_iv),
        'entry_premium': decision.est_premium, 'n_contracts': 1,
        'confidence': decision.confidence, 'status': 'OPEN', 'close_date': '',
        'exit_premium': '', 'pnl_usd': '', 'exit_reason': '',
        'regime': regime, 'vol_premium': vol_premium, 'hhi': hhi,
        'decision_note': decision.explanation,
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    _save_positions(df)
    return new_id


def mark_and_manage(current_spot: float, current_iv: float) -> list:
    """Mark open positions to model value and close those hitting exit rules.

    Returns a list of (id, action, detail) describing what happened.
    Simulation pricing: revalue the straddle at the current XSP spot and IV,
    with linear theta decay via reduced time-to-expiry.
    """
    df = load_positions()
    if df.empty:
        return []

    events = []
    iv_dec = current_iv / 100 if (current_iv and current_iv > 1) else (current_iv or 0.15)
    xsp_spot_now = current_spot / SPX_XSP_RATIO
    today = date.today()

    # Ensure mutable columns can hold mixed types (avoid float64 coercion errors)
    for _c in ('status', 'close_date', 'exit_premium', 'pnl_usd', 'exit_reason'):
        df[_c] = df[_c].astype(object)

    for idx, pos in df[df['status'] == 'OPEN'].iterrows():
        try:
            open_d = date.fromisoformat(str(pos['open_date']))
        except Exception:
            open_d = today
        days_held = (today - open_d).days
        dte_left  = max(int(pos['dte_target']) - days_held, 0)
        T_left    = max(dte_left / 365.0, 1e-4)

        entry_prem = float(pos['entry_premium'])
        cur_val    = straddle_price(xsp_spot_now, float(pos['xsp_strike']),
                                    T_left, iv_dec)
        ratio = cur_val / entry_prem if entry_prem > 0 else 0.0

        reason = None
        if ratio >= PROFIT_TARGET_MULT:
            reason = f"Profit target {PROFIT_TARGET_MULT:.1f}× ({ratio:.1f}×)"
        elif ratio <= (STOP_LOSS_PCT / 100.0):
            reason = f"Stop loss — premio a {ratio*100:.0f}% dell'entry"
        elif dte_left <= HOLD_DTE_FLOOR:
            reason = f"DTE floor ({dte_left}g) — chiusura prima dell'accelerazione theta"

        if reason:
            # Long vol P&L: (exit - entry) × 100; minus simple cost estimate
            cost = (1.25 * 4) + (entry_prem * 100 * 0.015 * 4)  # 2 legs, round trip
            pnl  = (cur_val - entry_prem) * 100 - cost
            df.at[idx, 'status']       = 'CLOSED'
            df.at[idx, 'close_date']   = today.isoformat()
            df.at[idx, 'exit_premium'] = round(cur_val, 2)
            df.at[idx, 'pnl_usd']      = round(pnl, 2)
            df.at[idx, 'exit_reason']  = reason
            events.append((int(pos['id']), 'CLOSE', f"{reason}  P&L=${pnl:+.0f}"))

    _save_positions(df)
    return events


# ══════════════════════════════════════════════════════════════════════════════
# Risk / capital status
# ══════════════════════════════════════════════════════════════════════════════
def compute_unrealized(current_spot: float, current_iv: float) -> dict:
    """Compute live unrealized P&L for OPEN positions WITHOUT changing state.

    Read-only: revalues each open straddle at the current XSP spot and IV,
    using time decay from days held.  Returns a dict keyed by position id
    plus a total, so the dashboard can show real-time latent P&L between
    the daily engine runs.

    Returns: {'by_id': {id: {...}}, 'total_unrealized': float, 'n_open': int}
    """
    df = load_positions()
    out = {'by_id': {}, 'total_unrealized': 0.0, 'n_open': 0}
    if df.empty:
        return out

    iv_dec = current_iv / 100 if (current_iv and current_iv > 1) else (current_iv or 0.15)
    xsp_spot_now = current_spot / SPX_XSP_RATIO
    today = date.today()
    total = 0.0

    for _, pos in df[df['status'] == 'OPEN'].iterrows():
        try:
            open_d = date.fromisoformat(str(pos['open_date']))
        except Exception:
            open_d = today
        days_held = (today - open_d).days
        dte_left  = max(int(pos['dte_target']) - days_held, 0)
        T_left    = max(dte_left / 365.0, 1e-4)

        entry_prem = float(pos['entry_premium'])
        cur_val    = straddle_price(xsp_spot_now, float(pos['xsp_strike']),
                                    T_left, iv_dec)
        ratio  = cur_val / entry_prem if entry_prem > 0 else 0.0
        # Unrealized P&L (no exit costs applied — position still open)
        unreal = (cur_val - entry_prem) * 100
        total += unreal
        out['by_id'][int(pos['id'])] = {
            'current_value': round(cur_val, 2),
            'entry_premium': round(entry_prem, 2),
            'ratio':         round(ratio, 2),
            'unrealized':    round(unreal, 2),
            'dte_left':      dte_left,
            'days_held':     days_held,
        }

    out['total_unrealized'] = round(total, 2)
    out['n_open'] = len(out['by_id'])
    return out


def capital_status() -> dict:
    """Compute paper account status from closed-trade P&L."""
    df = load_positions()
    closed = df[df['status'] == 'CLOSED'] if not df.empty else df
    cum_pnl = float(pd.to_numeric(closed['pnl_usd'], errors='coerce').sum()) \
              if not closed.empty else 0.0
    account = INITIAL_CAPITAL + cum_pnl
    drawdown = min(0.0, cum_pnl)
    dd_pct = abs(drawdown) / INITIAL_CAPITAL * 100 if INITIAL_CAPITAL else 0.0
    return {
        'initial_capital': INITIAL_CAPITAL,
        'account_value':   account,
        'cumulative_pnl':  cum_pnl,
        'drawdown_pct':    dd_pct,
        'max_drawdown_pct': MAX_DRAWDOWN_PCT,
        'is_safe':         dd_pct < MAX_DRAWDOWN_PCT,
        'danger_zone':     dd_pct > MAX_DRAWDOWN_PCT * 0.8,
        'n_open':          int((df['status'] == 'OPEN').sum()) if not df.empty else 0,
        'n_closed':        len(closed),
    }


def can_open_new() -> tuple[bool, str]:
    """Check whether a new paper position may be opened (risk gates)."""
    cs = capital_status()
    if not cs['is_safe']:
        return False, f"Hard stop drawdown {cs['drawdown_pct']:.0f}% ≥ {MAX_DRAWDOWN_PCT:.0f}%"
    if cs['n_open'] >= 1:
        return False, "Massimo 1 posizione aperta (conto 5k)"
    return True, "OK"


# ══════════════════════════════════════════════════════════════════════════════
# Process metrics (validation phase)
# ══════════════════════════════════════════════════════════════════════════════
def process_metrics(snapshot_dir: str) -> dict:
    """Lightweight process tracker for the dashboard-only version."""
    df = load_positions()
    closed = df[df['status'] == 'CLOSED'] if not df.empty else df
    n_closed = len(closed)

    # Data continuity from snapshots
    n_snaps = 0
    if os.path.isdir(snapshot_dir):
        n_snaps = len([f for f in os.listdir(snapshot_dir)
                       if f.startswith('SPX_') and f.endswith('.csv')])

    win_rate = None
    if n_closed > 0:
        pnls = pd.to_numeric(closed['pnl_usd'], errors='coerce')
        win_rate = float((pnls > 0).mean())

    return {
        'snapshots':    n_snaps,
        'n_closed':     n_closed,
        'n_open':       int((df['status'] == 'OPEN').sum()) if not df.empty else 0,
        'sample_target': 20,
        'sample_pct':   min(n_closed / 20 * 100, 100),
        'win_rate':     win_rate,
    }
