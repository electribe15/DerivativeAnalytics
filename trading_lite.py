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
import json
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
MAX_PREMIUM_PCT      = 8.0     # hard cap: max premium per trade = 8% of account ($400)
                               # ($1,250 on 5k). Anything above this is BLOCKED.
                               # On a small account, an ATM XSP straddle (~$2,200)
                               # exceeds this — by design the engine prefers cheaper
                               # structures (strangle, single put) and shorter DTE,
                               # and SKIPS entirely if nothing fits the budget.
PREMIUM_PREFER_PCT    = 5.0    # soft target: prefer structures under 5% ($250)
MAX_DAILY_LOSS       = 250.0   # circuit breaker (informational in sim)
MAX_DRAWDOWN_PCT     = 50.0    # hard stop (the 50% rule)
SPX_XSP_RATIO        = 10.0    # signal SPX → execution XSP scale

# ── Long-vol selection thresholds (mirror of long_vol_selector.py) ──────────────
IV_CHEAP_BELOW       = 16.0
IV_SKIP_ABOVE        = 28.0
VP_BUY_BELOW         = 2.0
DTE_TARGET           = 45
PROFIT_TARGET_MULT   = 2.5     # full profit target — hard ceiling, always closes
STOP_LOSS_PCT        = 50.0    # stop loss: close if premium falls to this % of entry
MIN_CONFIDENCE       = 0.45
HOLD_DTE_FLOOR       = 14

# ── Trailing stop (Option A) — lock in profit before it evaporates ──────────────
# Once a position's value reaches TRAIL_ARM_MULT × entry, a trailing stop arms.
# After that, if the value retraces TRAIL_GIVEBACK_PCT % from its peak, the
# position is closed to bank the gain — instead of waiting for the full 2.5×
# target that may never come. This directly addresses the "a winning position
# turning into a loss" problem.
TRAIL_ARM_MULT       = 1.5     # trailing stop activates once value ≥ 1.5× entry
TRAIL_GIVEBACK_PCT   = 25.0    # close if value falls 25% from the peak reached

# ── VolDex exit (best-effort reinforcement of the same principle) ───────────────
# The long-vol thesis is "I bought vol cheap; close when vol is no longer cheap."
# If VolDex has risen VOLDEX_EXIT_MULT × above its entry level AND the position
# is in profit, the thesis has played out — close. This only fires when VolDex
# is computable; it never blocks the engine if VolDex is unavailable.
VOLDEX_EXIT_MULT     = 1.40    # close if VolDex ≥ 1.40× the entry VolDex (+40%)

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


def legs_value(legs: list, S: float, T: float, sigma: float) -> float:
    """Mark-to-model value of an explicit list of option legs.

    Generalises straddle_price() to any combination of legs (straddle,
    strangle, single put, etc.) — BUY legs add value, SELL legs subtract,
    each leg's own strike/type/quantity is respected. This is the valuation
    used by mark_and_manage()/compute_unrealized() for OPEN positions,
    so a strangle (two different strikes) is revalued correctly instead of
    being approximated as a straddle on a single strike.
    """
    total = 0.0
    for l in legs:
        sign = 1.0 if l.get('side', 'BUY') == 'BUY' else -1.0
        qty  = float(l.get('qty', 1))
        flag = 'c' if l.get('type') == 'CALL' else 'p'
        px   = bs_price(S, float(l['strike']), T, R_FREE, sigma, flag)
        total += sign * qty * px
    return total


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
    legs:        list = field(default_factory=list)    # explicit option legs
    explanation: str = ''             # human-readable why TRADE / why SKIP
    skip_reason: str = ''

    @property
    def summary_html(self) -> str:
        if self.action == 'SKIP':
            return f"<b style='color:#9CA3AF'>⏸ SKIP</b> — {self.skip_reason}"
        color = '#10B981' if self.confidence > 0.6 else '#F59E0B'
        strat = {'long_straddle': '🎯 Long Straddle',
                 'long_strangle': '🎲 Long Strangle',
                 'long_put': '🛡 Long Put (tail)'}.get(self.strategy, self.strategy)
        return (f"<b style='color:{color}'>{strat}</b> · "
                f"DTE {self.target_dte} · XSP strike {self.xsp_strike:.0f} · "
                f"premio stim. {self.est_premium:.2f} · "
                f"confidence <b>{int(self.confidence*100)}%</b>")


def evaluate_signal(spot: float, regime: str, hhi: float,
                    vol_premium: Optional[float], atm_iv: Optional[float],
                    expected_move: Optional[float],
                    voldex: Optional[float] = None,
                    calldex: Optional[float] = None,
                    putdex: Optional[float] = None,
                    taildex: Optional[float] = None,
                    skew_trend: Optional[float] = None) -> LiteDecision:
    """Long-vol decision from the analytics the dashboard already computes.

    Simplified mirror of LongVolSelector.select() — uses the core signals
    available without market_context (IV level, VP, regime, HHI), PLUS the
    VolDex suite when available:
      voldex      — ATM 30-day implied vol (%) — preferred IV source over
                    the raw single-strike atm_iv when present (cleaner signal)
      calldex/putdex — ~16-delta OTM call/put IV (%) — used for the skew reading
      taildex     — ~10-delta (deeper OTM) put IV (%) — tail-risk proxy
      skew_trend  — today's (putdex-calldex) minus its recent rolling average,
                    in percentage points. Positive = skew widening (stress
                    building). Computed by the caller from voldex_history.csv
                    since this module stays self-contained / file-system free.

    All VolDex-derived parameters are OPTIONAL and the function degrades
    gracefully to the original IV-only logic if they are None — the
    automatic engine must never block on a chain that can't produce VolDex.
    """
    d = LiteDecision()
    # Prefer VolDex (cleaner ATM measure) over the single-strike atm_iv
    if voldex is not None:
        iv = voldex
    else:
        iv = (atm_iv or 0.18) * 100 if (atm_iv and atm_iv < 1) else (atm_iv or 18.0)
    vp = vol_premium if vol_premium is not None else 5.0
    factors = []
    conditions = []   # (label, met:bool, detail)

    def cond(label, met, detail):
        conditions.append({'label': label, 'met': bool(met), 'detail': detail})

    iv_source = "VolDex" if voldex is not None else "IV singolo strike"

    # Hard skip: vol too expensive to buy
    if iv > IV_SKIP_ABOVE:
        cond("Vol acquistabile", False,
             f"IV ({iv_source}) {iv:.1f}% > {IV_SKIP_ABOVE}% (soglia max) — vol troppo cara")
        d.skip_reason = f"IV {iv:.1f}% > {IV_SKIP_ABOVE}% — vol troppo cara da comprare"
        d.conditions = conditions
        d.explanation = (
            f"❌ NESSUN TRADE. La volatilità implicita ({iv_source}: {iv:.1f}%) è sopra "
            f"la soglia massima di acquisto ({IV_SKIP_ABOVE}%). Comprare vol così cara "
            f"elimina il vantaggio della strategia long-vol: si paga troppo premio. "
            f"Il sistema aspetta che l'IV scenda prima di considerare un acquisto.")
        return d

    score = 0.0
    # 1. Cheap IV (0-0.30) — uses VolDex if available, else single-strike IV
    if iv < IV_CHEAP_BELOW:
        score += min((IV_CHEAP_BELOW - iv) / IV_CHEAP_BELOW, 1.0) * 0.30
        factors.append(f"IV ({iv_source}) {iv:.1f}% &lt; {IV_CHEAP_BELOW}% — vol economica")
        cond("IV economica", True,
             f"IV ({iv_source}) {iv:.1f}% < {IV_CHEAP_BELOW}% — vol a buon mercato (+0.30 max)")
    elif iv < 20:
        score += 0.10
        factors.append(f"IV ({iv_source}) {iv:.1f}% — moderata")
        cond("IV economica", False,
             f"IV ({iv_source}) {iv:.1f}% moderata (tra {IV_CHEAP_BELOW}% e 20%) — solo +0.10")
    else:
        cond("IV economica", False,
             f"IV ({iv_source}) {iv:.1f}% non economica (≥ 20%) — nessun bonus")

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

    # 5. NEW — Skew dynamics (0-0.10): widening Put-Call skew = stress building
    if skew_trend is not None:
        if skew_trend > 0.5:
            score += 0.10
            factors.append(f"Skew Put−Call in espansione (+{skew_trend:.2f}pt "
                           "vs media recente) — stress crescente")
            cond("Skew in espansione", True,
                 f"Skew oggi {skew_trend:+.2f}pt sopra la media recente — "
                 "il mercato sta pagando più protezione al ribasso (+0.10)")
        elif skew_trend < -0.5:
            cond("Skew in espansione", False,
                 f"Skew oggi {skew_trend:+.2f}pt sotto la media recente — "
                 "in contrazione, nessun bonus")
        else:
            cond("Skew in espansione", False,
                 f"Skew stabile ({skew_trend:+.2f}pt vs media) — nessun bonus")
    else:
        cond("Skew in espansione", False, "Dato non disponibile (serve storico VolDex)")

    score = max(0.0, min(1.0, score))
    d.confidence = round(score, 3)
    d.conditions = conditions

    _met = [c for c in conditions if c['met']]
    _unmet = [c for c in conditions if not c['met']]

    if score < MIN_CONFIDENCE:
        d.skip_reason = (f"Conviction {score:.2f} &lt; {MIN_CONFIDENCE} — "
                         "condizioni non abbastanza favorevoli per comprare vol")
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

    # ── Strategy selection (budget-aware) ─────────────────────────────────────────
    # On a small account an ATM XSP straddle (~$2,200) blows past the premium
    # cap. So instead of picking one strategy and opening it at any cost, we
    # build candidate structures, compute each one's real premium, and choose
    # the cheapest that BOTH fits the signal AND fits the budget. If nothing
    # fits the hard cap, we SKIP — never open an oversized position.
    xsp_spot   = spot / SPX_XSP_RATIO
    T          = DTE_TARGET / 365.0
    iv_dec     = iv / 100
    # Expected move over the TRADE horizon (DTE_TARGET), not a short-term 0DTE
    # figure. A 1-sigma move at 45 days sets where the OTM legs sit; using a
    # tiny intraday EM would place them almost ATM and make them expensive.
    em_horizon = xsp_spot * iv_dec * math.sqrt(DTE_TARGET / 365.0)
    # If a same-horizon expected_move was supplied, prefer it; otherwise use
    # the model figure above. (expected_move from 0DTE is too small — ignore it
    # when it implies a sub-0.5-sigma strike.)
    em_xsp = em_horizon
    if expected_move:
        em_candidate = expected_move / SPX_XSP_RATIO
        # only trust the supplied EM if it's at least ~half the horizon sigma
        if em_candidate >= 0.5 * em_horizon:
            em_xsp = em_candidate

    def _leg(side, typ, strike):
        px = bs_price(xsp_spot, strike, T, R_FREE, iv_dec,
                      'c' if typ == 'CALL' else 'p')
        return {'side': side, 'type': typ, 'strike': strike, 'qty': 1,
                'premium_pts': round(px, 2), 'premium_usd': round(px * 100, 0)}

    def _build(strategy):
        """Return (legs, ref_strike) for a strategy name."""
        if strategy == 'long_straddle':
            k = round(xsp_spot / 5) * 5
            return [_leg('BUY','CALL',k), _leg('BUY','PUT',k)], k
        if strategy == 'long_strangle':
            kc = round((xsp_spot + em_xsp) / 5) * 5
            kp = round((xsp_spot - em_xsp) / 5) * 5
            return [_leg('BUY','CALL',kc), _leg('BUY','PUT',kp)], kc
        # long_put — single OTM put, cheapest structure
        kp = round((xsp_spot - em_xsp) / 5) * 5
        return [_leg('BUY','PUT',kp)], kp

    # Candidate ranking by signal preference (best-fit first)
    tail_spread = (taildex - putdex) if (taildex is not None and putdex is not None) else None
    bias_tail   = (tail_spread is not None and tail_spread > 1.5 and regime == 'SHORT')
    if bias_tail:
        factors.append(f"TailDex−PutDex {tail_spread:+.2f}pt — coda più cara del corpo, "
                       "rischio di coda in aumento")
        preference = ['long_put', 'long_strangle', 'long_straddle']
    elif iv < IV_CHEAP_BELOW and score > 0.6:
        # Would prefer straddle, but it's the most expensive — keep it last-resort
        preference = ['long_strangle', 'long_put', 'long_straddle']
    else:
        preference = ['long_strangle', 'long_put', 'long_straddle']

    hard_cap   = INITIAL_CAPITAL * MAX_PREMIUM_PCT / 100.0      # $ ceiling
    prefer_cap = INITIAL_CAPITAL * PREMIUM_PREFER_PCT / 100.0   # $ soft target

    # Evaluate all candidates
    candidates = []
    for strat in preference:
        legs_c, ref_c = _build(strat)
        prem_pts = sum(l['premium_pts'] * l.get('qty', 1) for l in legs_c)
        prem_usd = prem_pts * 100
        candidates.append((strat, legs_c, ref_c, prem_pts, prem_usd))

    # Pick: cheapest that fits the SOFT target; else cheapest under HARD cap;
    # else nothing fits → SKIP
    fits_soft = [c for c in candidates if c[4] <= prefer_cap]
    fits_hard = [c for c in candidates if c[4] <= hard_cap]
    chosen = None
    if fits_soft:
        chosen = min(fits_soft, key=lambda c: c[4])
    elif fits_hard:
        chosen = min(fits_hard, key=lambda c: c[4])

    if chosen is None:
        cheapest = min(candidates, key=lambda c: c[4])
        d.action = 'SKIP'
        d.confidence = round(score, 3)
        d.skip_reason = (f"Premio minimo ${cheapest[4]:.0f} &gt; cap "
                         f"${hard_cap:.0f} ({MAX_PREMIUM_PCT:.0f}% di "
                         f"${INITIAL_CAPITAL:.0f})")
        cond("Premio entro budget", False,
             f"Struttura più economica ({cheapest[0]}) costa ${cheapest[4]:.0f}, "
             f"oltre il tetto di ${hard_cap:.0f} — posizione troppo grande per il conto")
        d.conditions = conditions
        d.explanation = (
            f"❌ NESSUN TRADE. Il segnale era favorevole (conviction {score:.2f}), "
            f"ma anche la struttura più economica disponibile ({cheapest[0]}) "
            f"costerebbe ${cheapest[4]:.0f} di premio — oltre il tetto massimo di "
            f"${hard_cap:.0f} ({MAX_PREMIUM_PCT:.0f}% del capitale). Aprire questa "
            f"posizione metterebbe a rischio una quota eccessiva del conto da "
            f"${INITIAL_CAPITAL:.0f}. Il sistema non apre posizioni sovradimensionate: "
            f"con questo capitale, su XSP a {DTE_TARGET} giorni i premi sono troppo "
            f"alti. Servirebbe più capitale, una scadenza più breve, o attendere "
            f"un'IV più bassa che abbassi i premi.")
        return d

    strategy, legs, xsp_strike, est_prem, est_usd = chosen
    cond("Premio entro budget", True,
         f"{strategy} costa ${est_usd:.0f} — entro il tetto di ${hard_cap:.0f} "
         f"({est_usd/INITIAL_CAPITAL*100:.0f}% del capitale)")

    strike_spx = round(xsp_strike * SPX_XSP_RATIO / 5) * 5

    _strat_name = {'long_straddle': 'Long Straddle (ATM)',
                   'long_strangle': 'Long Strangle (OTM)',
                   'long_put': 'Long Put (tail, OTM)'}.get(strategy, strategy)
    _legs_txt = "  |  ".join(
        f"{l['side']} {l.get('qty',1)}x {l['type']} {l['strike']:.0f} @ "
        f"{l['premium_pts']:.2f} (~${l['premium_usd']:.0f})" for l in legs)
    present = "; ".join(c['label'] for c in _met) or "nessuna"
    _why_strategy = {
        'long_straddle': 'IV molto economica e convinzione alta — straddle ATM per la '
                         'massima sensibilità al movimento (struttura più cara, scelta '
                         'solo se rientra nel budget)',
        'long_strangle': 'strangle OTM — più economico dello straddle, serve un '
                         'movimento più ampio ma costa meno premio',
        'long_put': 'singola put OTM — la struttura più economica e a rischio definito; '
                    'esposizione asimmetrica al ribasso, premio contenuto',
    }.get(strategy, '')
    d.explanation = (
        f"✅ TRADE: {_strat_name}. Punteggio di convinzione {score:.2f}, sopra la "
        f"soglia minima di {MIN_CONFIDENCE}. Le condizioni favorevoli si sono allineate.\n\n"
        f"✓ Condizioni che hanno guidato la decisione: {present}.\n\n"
        f"Gambe dell'operazione (su XSP):\n{_legs_txt}\n\n"
        f"Strategia scelta perché {_why_strategy}. "
        f"Scadenza {DTE_TARGET} giorni, premio totale {est_prem:.2f} punti "
        f"(~${est_usd:.0f}, {est_usd/INITIAL_CAPITAL*100:.0f}% del capitale). "
        f"Perdita massima limitata al premio pagato.")

    d.action      = 'TRADE'
    d.strategy    = strategy
    d.strike      = strike_spx
    d.xsp_strike  = xsp_strike
    d.est_premium = round(est_prem, 2)
    d.legs        = legs
    d.factors     = factors
    return d


# ══════════════════════════════════════════════════════════════════════════════
# Paper position store (CSV-based)
# ══════════════════════════════════════════════════════════════════════════════
_POS_COLS = ['id', 'open_date', 'mode', 'strategy', 'signal_spot', 'xsp_spot',
             'xsp_strike', 'dte_target', 'entry_iv', 'entry_premium',
             'n_contracts', 'confidence', 'status', 'close_date',
             'exit_premium', 'pnl_usd', 'exit_reason',
             'regime', 'vol_premium', 'hhi', 'decision_note', 'legs_json',
             'voldex', 'calldex', 'putdex', 'taildex', 'skew_trend',
             'peak_ratio']


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


def reconstruct_legs_for_row(row) -> Optional[list]:
    """Approximate the option legs for a PAST trade that has no legs_json.

    WARNING: this is a RECONSTRUCTION, not the original data. It re-derives
    the legs from the saved xsp_spot, xsp_strike, entry_iv, strategy and DTE.
    The strikes may differ slightly from what was actually chosen at the time
    because rounding and the expected-move estimate are recomputed now.
    Use only for display of historical trades; treat the numbers as estimates.

    Returns a list of leg dicts (same shape as decision.legs) or None.
    """
    try:
        strat   = str(row.get('strategy', ''))
        xsp_spot = float(row.get('xsp_spot') or 0)
        xsp_k    = float(row.get('xsp_strike') or 0)
        iv       = float(row.get('entry_iv') or 0)
        dte      = int(float(row.get('dte_target') or DTE_TARGET))
    except Exception:
        return None

    if xsp_spot <= 0 or iv <= 0:
        return None

    iv_dec = iv / 100 if iv > 1 else iv
    T      = max(dte / 365.0, 1e-4)

    legs = []
    if strat == 'long_straddle':
        k = xsp_k if xsp_k > 0 else round(xsp_spot / 5) * 5
        cp = bs_price(xsp_spot, k, T, R_FREE, iv_dec, 'c')
        pp = bs_price(xsp_spot, k, T, R_FREE, iv_dec, 'p')
        legs = [
            {'side': 'BUY', 'type': 'CALL', 'strike': k,
             'premium_pts': round(cp, 2), 'premium_usd': round(cp * 100, 0)},
            {'side': 'BUY', 'type': 'PUT', 'strike': k,
             'premium_pts': round(pp, 2), 'premium_usd': round(pp * 100, 0)},
        ]
    elif strat == 'long_strangle':
        # Reconstruct OTM strikes around spot using the same EM logic
        em_xsp = xsp_spot * iv_dec / math.sqrt(252) * math.sqrt(dte)
        kc = xsp_k if xsp_k > 0 else round((xsp_spot + em_xsp) / 5) * 5
        kp = round((xsp_spot - em_xsp) / 5) * 5
        cp = bs_price(xsp_spot, kc, T, R_FREE, iv_dec, 'c')
        pp = bs_price(xsp_spot, kp, T, R_FREE, iv_dec, 'p')
        legs = [
            {'side': 'BUY', 'type': 'CALL', 'strike': kc,
             'premium_pts': round(cp, 2), 'premium_usd': round(cp * 100, 0)},
            {'side': 'BUY', 'type': 'PUT', 'strike': kp,
             'premium_pts': round(pp, 2), 'premium_usd': round(pp * 100, 0)},
        ]
    else:
        return None

    return legs if legs else None


def backfill_legs() -> dict:
    """Reconstruct and SAVE approximate legs for past trades missing legs_json.

    Marks reconstructed legs with 'reconstructed': True so the UI can flag them
    as estimates. Returns a summary {filled, skipped, total}.
    """
    df = load_positions()
    summary = {'filled': 0, 'skipped': 0, 'total': len(df)}
    if df.empty:
        return summary

    if 'legs_json' not in df.columns:
        df['legs_json'] = ''
    df['legs_json'] = df['legs_json'].astype(object)

    for idx, row in df.iterrows():
        existing = row.get('legs_json')
        if isinstance(existing, str) and existing.strip():
            summary['skipped'] += 1          # already has real legs
            continue
        legs = reconstruct_legs_for_row(row)
        if legs:
            for l in legs:
                l['reconstructed'] = True     # flag as estimate
            df.at[idx, 'legs_json'] = json.dumps(legs)
            summary['filled'] += 1
        else:
            summary['skipped'] += 1

    _save_positions(df)
    return summary


def open_paper_position(decision: LiteDecision, spot: float,
                        regime: str, vol_premium: float, hhi: float,
                        atm_iv: float,
                        voldex: Optional[float] = None,
                        calldex: Optional[float] = None,
                        putdex: Optional[float] = None,
                        taildex: Optional[float] = None,
                        skew_trend: Optional[float] = None) -> int:
    """Record a simulated long-vol position. Returns the new id.

    The VolDex suite values (when available) are stored alongside the
    trade for transparency — same spirit as regime/vol_premium/hhi — so
    the historical record shows exactly what each signal read at entry.
    """
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
        'legs_json': json.dumps(decision.legs) if decision.legs else '',
        'voldex': voldex, 'calldex': calldex, 'putdex': putdex,
        'taildex': taildex, 'skew_trend': skew_trend,
        'peak_ratio': 1.0,   # entry value / entry premium = 1.0 at open
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    _save_positions(df)
    return new_id


def _position_legs(pos) -> Optional[list]:
    """Parse the stored legs_json for a position row, if present."""
    lj = pos.get('legs_json') if hasattr(pos, 'get') else pos['legs_json']
    if isinstance(lj, str) and lj.strip():
        try:
            return json.loads(lj)
        except Exception:
            return None
    return None


def mark_and_manage(current_spot: float, current_iv: float,
                    current_voldex: Optional[float] = None) -> list:
    """Mark open positions to model value and close those hitting exit rules.

    Returns a list of (id, action, detail) describing what happened.
    Simulation pricing: revalue the position's ACTUAL legs (straddle,
    strangle, or single put) at the current XSP spot and IV, with theta
    decay via reduced time-to-expiry.

    Exit rules, in priority order:
      1. Profit target  — value ≥ PROFIT_TARGET_MULT × entry (hard ceiling)
      2. Trailing stop  — once value peaked ≥ TRAIL_ARM_MULT × entry, close if
                          it gives back TRAIL_GIVEBACK_PCT % from that peak
                          (locks in profit before it evaporates)
      3. VolDex exit    — if current VolDex ≥ VOLDEX_EXIT_MULT × entry VolDex
                          and position in profit (vol thesis played out;
                          best-effort, only when VolDex is provided)
      4. Stop loss      — value ≤ STOP_LOSS_PCT % of entry (cut the loss)
      5. DTE floor      — ≤ HOLD_DTE_FLOOR days left (avoid theta acceleration)
    """
    df = load_positions()
    if df.empty:
        return []

    events = []
    iv_dec = current_iv / 100 if (current_iv and current_iv > 1) else (current_iv or 0.15)
    xsp_spot_now = current_spot / SPX_XSP_RATIO
    today = date.today()

    # Ensure mutable columns can hold mixed types (avoid float64 coercion errors)
    for _c in ('status', 'close_date', 'exit_premium', 'pnl_usd', 'exit_reason',
               'peak_ratio'):
        if _c in df.columns:
            df[_c] = df[_c].astype(object)
    if 'peak_ratio' not in df.columns:
        df['peak_ratio'] = 1.0

    for idx, pos in df[df['status'] == 'OPEN'].iterrows():
        try:
            open_d = date.fromisoformat(str(pos['open_date']))
        except Exception:
            open_d = today
        days_held = (today - open_d).days
        dte_left  = max(int(pos['dte_target']) - days_held, 0)
        T_left    = max(dte_left / 365.0, 1e-4)

        entry_prem = float(pos['entry_premium'])
        legs = _position_legs(pos)
        if legs:
            cur_val = legs_value(legs, xsp_spot_now, T_left, iv_dec)
            n_legs  = len(legs)
        else:
            # Legacy fallback: same-strike straddle approximation
            cur_val = straddle_price(xsp_spot_now, float(pos['xsp_strike']),
                                     T_left, iv_dec)
            n_legs  = 2
        ratio = cur_val / entry_prem if entry_prem > 0 else 0.0

        # Update the running peak ratio (for the trailing stop)
        try:
            prev_peak = float(pos.get('peak_ratio') or 1.0)
        except Exception:
            prev_peak = 1.0
        peak = max(prev_peak, ratio)
        df.at[idx, 'peak_ratio'] = round(peak, 3)

        reason = None
        # 1. Full profit target — hard ceiling
        if ratio >= PROFIT_TARGET_MULT:
            reason = f"Profit target {PROFIT_TARGET_MULT:.1f}× ({ratio:.1f}×)"
        # 2. Trailing stop — armed once peak ≥ TRAIL_ARM_MULT, fires on giveback
        elif (peak >= TRAIL_ARM_MULT and
              ratio <= peak * (1.0 - TRAIL_GIVEBACK_PCT / 100.0) and
              ratio > 1.0):
            reason = (f"Trailing stop — profitto bloccato a {ratio:.2f}× "
                      f"(picco {peak:.2f}×, ripiego {TRAIL_GIVEBACK_PCT:.0f}%)")
        # 3. VolDex exit — vol bought cheap is now expensive (best-effort)
        elif (current_voldex is not None and pos.get('voldex') not in (None, '') and
              ratio > 1.0):
            try:
                entry_vd = float(pos['voldex'])
                if entry_vd > 0 and current_voldex >= entry_vd * VOLDEX_EXIT_MULT:
                    reason = (f"Uscita VolDex — vol salita a {current_voldex:.1f}% "
                              f"(da {entry_vd:.1f}% all'entrata, +{VOLDEX_EXIT_MULT:.0%} "
                              f"raggiunto) — tesi long-vol avverata, ratio {ratio:.2f}×")
            except Exception:
                pass
        # 4. Stop loss — cut the loss
        if reason is None and ratio <= (STOP_LOSS_PCT / 100.0):
            reason = f"Stop loss — premio a {ratio*100:.0f}% dell'entry"
        # 5. DTE floor — exit before theta accelerates
        if reason is None and dte_left <= HOLD_DTE_FLOOR:
            reason = f"DTE floor ({dte_left}g) — chiusura prima dell'accelerazione theta"

        if reason:
            # Long vol P&L: (exit - entry) × 100; cost scales with leg count
            cost = (1.25 * 2 * n_legs) + (entry_prem * 100 * 0.015 * 2 * n_legs)
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
        legs = _position_legs(pos)
        if legs:
            cur_val = legs_value(legs, xsp_spot_now, T_left, iv_dec)
        else:
            cur_val = straddle_price(xsp_spot_now, float(pos['xsp_strike']),
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


def archive_and_reset() -> dict:
    """Archive the current positions history to a timestamped file and start
    fresh with an empty store.

    The old positions.csv is renamed to positions_archive_YYYYMMDD_HHMMSS.csv
    in the same folder (nothing is deleted — full audit trail preserved), and
    a new empty positions.csv takes its place. The paper account effectively
    restarts from the initial capital with zero open/closed trades, while the
    archived file remains for reference.

    Returns a summary dict: {archived, archive_path, rows_archived}.
    """
    df = load_positions()
    if df.empty:
        return {'archived': False, 'archive_path': None, 'rows_archived': 0,
                'reason': 'Storico già vuoto — niente da archiviare.'}

    _ensure_dir()
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    archive_path = os.path.join(PAPER_DIR, f'positions_archive_{stamp}.csv')
    df.to_csv(archive_path, index=False)              # save a copy
    # Replace live store with an empty one (same schema)
    pd.DataFrame(columns=_POS_COLS).to_csv(POSITIONS_CSV, index=False)
    return {'archived': True, 'archive_path': archive_path,
            'rows_archived': len(df)}


def list_archives() -> list:
    """Return the list of archived history files (most recent first)."""
    if not os.path.isdir(PAPER_DIR):
        return []
    arc = [f for f in os.listdir(PAPER_DIR)
           if f.startswith('positions_archive_') and f.endswith('.csv')]
    return sorted(arc, reverse=True)


def find_oversized_open(current_spot: float = None,
                        current_iv: float = None) -> list:
    """Identify OPEN positions whose entry premium exceeds the current hard cap.

    These are typically legacy trades opened by an earlier version of the
    engine (before budget-aware sizing existed) that are too large for the
    account. Returns a list of dicts describing each, with current marked
    value if spot/iv are provided. Read-only — does not change anything.
    """
    df = load_positions()
    if df.empty:
        return []
    cap = INITIAL_CAPITAL * MAX_PREMIUM_PCT / 100.0
    out = []
    for _, pos in df[df['status'] == 'OPEN'].iterrows():
        try:
            prem_usd = float(pos['entry_premium']) * 100
        except Exception:
            continue
        if prem_usd > cap:
            info = {
                'id': int(pos['id']),
                'strategy': pos.get('strategy', ''),
                'entry_premium_usd': round(prem_usd, 0),
                'cap_usd': round(cap, 0),
                'pct_of_capital': round(prem_usd / INITIAL_CAPITAL * 100, 1),
            }
            out.append(info)
    return out


def close_position_manual(pos_id: int, current_spot: float,
                          current_iv: float) -> dict:
    """Manually close a single OPEN position at current marked value.

    Used to clean up oversized legacy positions. Marks the position CLOSED
    with reason 'Chiusura manuale (sovradimensionata)' and records the P&L.
    Returns a summary dict.
    """
    df = load_positions()
    if df.empty:
        return {'closed': False, 'reason': 'nessuna posizione'}

    iv_dec = current_iv / 100 if (current_iv and current_iv > 1) else (current_iv or 0.15)
    xsp_spot_now = current_spot / SPX_XSP_RATIO
    today = date.today()

    for _c in ('status', 'close_date', 'exit_premium', 'pnl_usd', 'exit_reason'):
        df[_c] = df[_c].astype(object)

    mask = (df['id'] == pos_id) & (df['status'] == 'OPEN')
    if not mask.any():
        return {'closed': False, 'reason': f'posizione #{pos_id} non aperta'}

    idx = df[mask].index[0]
    pos = df.loc[idx]
    try:
        open_d = date.fromisoformat(str(pos['open_date']))
    except Exception:
        open_d = today
    days_held = (today - open_d).days
    dte_left  = max(int(pos['dte_target']) - days_held, 0)
    T_left    = max(dte_left / 365.0, 1e-4)
    entry_prem = float(pos['entry_premium'])

    legs = _position_legs(pos)
    if legs:
        cur_val = legs_value(legs, xsp_spot_now, T_left, iv_dec)
        n_legs  = len(legs)
    else:
        cur_val = straddle_price(xsp_spot_now, float(pos['xsp_strike']), T_left, iv_dec)
        n_legs  = 2
    cost = (1.25 * 2 * n_legs) + (entry_prem * 100 * 0.015 * 2 * n_legs)
    pnl  = (cur_val - entry_prem) * 100 - cost

    df.at[idx, 'status']       = 'CLOSED'
    df.at[idx, 'close_date']   = today.isoformat()
    df.at[idx, 'exit_premium'] = round(cur_val, 2)
    df.at[idx, 'pnl_usd']      = round(pnl, 2)
    df.at[idx, 'exit_reason']  = 'Chiusura manuale (posizione sovradimensionata legacy)'
    _save_positions(df)
    return {'closed': True, 'id': pos_id, 'pnl_usd': round(pnl, 2)}


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
