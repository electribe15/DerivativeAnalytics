#!/usr/bin/env python3
"""
Paper Engine Runner — daily automatic long-vol simulation
=========================================================
Run by GitHub Actions once per day after the US close.  It:

  1. Fetches the current SPX option chain (same source as the dashboard)
  2. Computes the signals (GEX regime, HHI, vol premium, IV, expected move)
  3. Evaluates the long-vol decision via trading_lite
  4. Marks open paper positions to model value and closes those hitting exits
  5. Opens a new simulated position if the signal says TRADE and risk allows
  6. Writes everything to paper_trades/positions.csv

The workflow then commits paper_trades/ to the repo, so the history
accumulates automatically and survives Streamlit Cloud restarts — exactly
like the daily chain snapshots.  The dashboard only READS this file.

This is pure simulation — no broker, no orders, no external connections.
"""
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dex_gex_dashboard as dg
import trading_lite as tl
import short_vol_lite as sv


def log(msg: str):
    print(f"[paper-engine {datetime.utcnow().isoformat(timespec='seconds')}Z] {msg}")


def main() -> int:
    log("Starting daily paper-engine run")

    # ── 1. Fetch chain ────────────────────────────────────────────────────────
    try:
        raw, spot = dg.fetch_options_data('$SPX', force_refresh=True)
    except Exception as e:
        log(f"Chain fetch failed: {e} — aborting (no state change)")
        return 0  # do not fail the workflow; Barchart can be flaky

    if raw is None or raw.empty:
        log("Empty chain — aborting (no state change)")
        return 0

    log(f"Chain: {len(raw)} contracts, spot={spot:.1f}")

    # ── 2. Compute signals (same functions the dashboard uses) ────────────────
    by_strike = dg.aggregate_by_strike(raw)
    ga = dg.compute_gex_analytics(raw, by_strike, spot) or {}
    m0 = dg.compute_0dte_metrics(raw, spot) or {}
    _reg_full = ga.get('regime')
    regime = ('LONG' if (_reg_full and 'LONG' in _reg_full) else
              'SHORT' if _reg_full else
              ('LONG' if ga.get('net_gex_total', 0) >= 0 else 'SHORT'))
    hhi    = ga.get('hhi', 0.0)
    atm_iv = m0.get('atm_iv')
    em     = m0.get('exp_move_pts')

    # Vol premium from EWMA RVol vs VIX
    vp = None
    try:
        ohlc = dg.fetch_ohlc_history('$SPX', n_calendar_days=400)
        vix  = dg.fetch_vix_history(n_calendar_days=60)
        if ohlc is not None and not ohlc.empty and vix is not None and not vix.empty:
            rv = dg.compute_rvol_all(ohlc, window=126)
            vixv = float(vix['vix'].iloc[-1] * 100)
            if 'EWMA λ=0.94' in rv.columns:
                vp = vixv - float(rv['EWMA λ=0.94'].iloc[-1])
    except Exception as e:
        log(f"Vol premium computation failed (non-fatal): {e}")

    log(f"Signals: regime={regime} hhi={hhi:.4f} "
        f"atm_iv={atm_iv} vp={vp} em={em}")

    # ── 3. Save chain snapshot too (so backtest history grows) ────────────────
    try:
        dg.save_daily_snapshot(raw, spot, '$SPX')
    except Exception as e:
        log(f"Snapshot save failed (non-fatal): {e}")

    # GEX/DEX derived-metrics history (Net GEX, flip, regime, HHI, P/C)
    try:
        dg.save_gex_metrics_snapshot(raw, by_strike, spot)
    except Exception as e:
        log(f"GEX metrics snapshot failed (non-fatal): {e}")

    # Premium/IV snapshot — the "specchietto premi" daily history
    try:
        _pp = dg.save_premium_snapshot(raw, spot, atm_iv)
        if _pp:
            log(f"Premium snapshot saved: {os.path.basename(_pp)}")
        else:
            log("Premium snapshot not saved (no eligible expiries)")
    except Exception as e:
        log(f"Premium snapshot failed (non-fatal): {e}")

    # ── 3b. VolDex suite — same chain, no extra fetch needed ──────────────────
    voldex = calldex = putdex = taildex = skew_trend = None
    vx = None
    try:
        vx = dg.compute_voldex(raw, spot)
        if vx.get('error'):
            log(f"VolDex not computed (non-fatal): {vx['error']}")
        else:
            voldex, calldex, putdex, taildex = (
                vx['voldex'], vx['calldex'], vx['putdex'], vx['taildex'])
            dg.save_voldex_snapshot(vx)
            log(f"VolDex: {voldex}  CallDex: {calldex}  "
                f"PutDex: {putdex}  TailDex: {taildex}")

            # Skew trend: today's (putdex-calldex) vs its recent rolling average
            if putdex is not None and calldex is not None:
                today_skew = putdex - calldex
                hist = dg.load_voldex_history()
                if not hist.empty and len(hist) >= 3:
                    hist_skew = (hist['putdex'] - hist['calldex']).dropna()
                    # Exclude today's just-written row from the baseline average
                    hist_skew = hist_skew.iloc[:-1] if len(hist_skew) > 1 else hist_skew
                    if len(hist_skew) >= 2:
                        skew_trend = round(today_skew - float(hist_skew.tail(10).mean()), 2)
                        log(f"Skew trend: {skew_trend:+.2f}pt vs recent average")
    except Exception as e:
        log(f"VolDex computation failed (non-fatal): {e}")

    # ── 4. Mark & manage existing open positions ──────────────────────────────
    iv_now = voldex if voldex is not None else (
        (atm_iv * 100 if atm_iv and atm_iv < 1 else atm_iv) or 15.0)
    events = tl.mark_and_manage(spot, iv_now, current_voldex=voldex)
    for pid, act, detail in events:
        log(f"Position #{pid} {act}: {detail}")
    if not events:
        log("No open positions hit exit conditions")

    # ── 5. Evaluate signal and open new position if warranted ─────────────────
    decision = tl.evaluate_signal(spot, regime, hhi, vp, atm_iv, em,
                                  voldex=voldex, calldex=calldex,
                                  putdex=putdex, taildex=taildex,
                                  skew_trend=skew_trend)
    log(f"Decision: {decision.action} {decision.strategy} "
        f"conf={decision.confidence} {decision.skip_reason}")

    # Log the decision (TRADE or SKIP) for selection-logic validation
    try:
        tl.log_daily_decision(decision, spot, voldex=voldex,
                              expected_move_pts=em)
    except Exception as e:
        log(f"Skip-log write failed (non-fatal): {e}")

    if decision.action == 'TRADE':
        can, why = tl.can_open_new()
        if can:
            pid = tl.open_paper_position(decision, spot, regime,
                                         vp or 0, hhi, atm_iv or 0.15,
                                         voldex=voldex, calldex=calldex,
                                         putdex=putdex, taildex=taildex,
                                         skew_trend=skew_trend)
            log(f"OPENED simulated position #{pid}: {decision.strategy} "
                f"XSP strike {decision.xsp_strike} premium {decision.est_premium}")
        else:
            log(f"Signal TRADE but cannot open: {why}")
    else:
        log("SKIP — no action (correct most days for long-vol)")

    # ── 5b. SHORT-VOL system (separate account, runs in parallel) ─────────────
    try:
        sv_events = sv.mark_and_manage(spot, iv_now)
        for ev in sv_events:
            log(f"[SHORT] Position #{ev['id']} closed: {ev['reason']} "
                f"P&L ${ev['pnl_usd']:+.0f}")
        if not sv_events:
            log("[SHORT] No open positions hit exit conditions")

        sv_term = dg.compute_term_structure_slope(vx) if vx else None
        sv_term_state = sv_term.get('state') if sv_term else None
        sv_decision = sv.evaluate_signal(spot, regime, hhi, vp, atm_iv, em,
                                         voldex=voldex, calldex=calldex,
                                         putdex=putdex, taildex=taildex,
                                         skew_trend=skew_trend,
                                         term_state=sv_term_state)
        log(f"[SHORT] Decision: {sv_decision.action} {sv_decision.strategy} "
            f"conf={sv_decision.confidence} {sv_decision.skip_reason}")
        try:
            sv.log_daily_decision(sv_decision, spot, voldex=voldex,
                                  expected_move_pts=em)
        except Exception as e:
            log(f"[SHORT] Skip-log write failed (non-fatal): {e}")

        if sv_decision.action == 'TRADE':
            # one open position at a time for the short-vol account
            _svdf = sv.load_positions()
            _sv_open = len(_svdf[_svdf['status'] == 'OPEN']) if not _svdf.empty else 0
            if _sv_open == 0:
                _svrow = sv.open_paper_position(sv_decision, spot, iv_now)
                log(f"[SHORT] OPENED #{_svrow.get('id')}: {sv_decision.strategy} "
                    f"credito ${sv_decision.credit_usd:.0f} "
                    f"perdita max ${sv_decision.max_loss_usd:.0f}")
            else:
                log(f"[SHORT] Signal TRADE but {_sv_open} position(s) already open")
        else:
            log("[SHORT] SKIP — no action")

        sv_acc = sv.account_summary(spot, iv_now)
        log(f"[SHORT] Account: equity ${sv_acc['equity']:.0f} "
            f"(realizzato ${sv_acc['realized']:+.0f}) open={sv_acc['n_open']} "
            f"closed={sv_acc['n_closed']}")
    except Exception as e:
        log(f"[SHORT] short-vol block failed (non-fatal): {e}")

    # ── 6. Summary ────────────────────────────────────────────────────────────
    cs = tl.capital_status()
    log(f"Account: ${cs['account_value']:.0f}  cum P&L ${cs['cumulative_pnl']:+.0f}  "
        f"open={cs['n_open']} closed={cs['n_closed']}")
    log("Run complete")
    return 0


if __name__ == '__main__':
    sys.exit(main())
