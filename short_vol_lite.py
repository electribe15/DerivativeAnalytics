# -*- coding: utf-8 -*-
"""short_vol_lite.py — Motore SHORT-VOL speculare a trading_lite.py.

Sistema simulato che VENDE volatilità con strutture a RISCHIO DEFINITO
(credit spread e iron condor, mai opzioni nude), su un conto paper separato
da $5.000 indipendente dal long-vol.

Filosofia (speculare al long-vol):
  - Long-vol compra quando: vol ECONOMICA + regime SHORT gamma
  - Short-vol vende quando:  vol CARA     + regime LONG gamma + niente stress

Lo short-vol incassa il premio e profitta dal decadimento temporale / dal
mercato che resta in un range. Le perdite, per natura, sono più grandi e più
rapide dei guadagni: per questo SOLO strutture a rischio definito e regole di
uscita molto reattive (stop sul multiplo del credito incassato).

Constraint condivisi col long-vol: simulazione pura, $5.000, segnale su SPX
ed esecuzione scalata su XSP (ratio 10), filosofia di validazione (processo
prima del profitto). Modulo autonomo: nessun accesso a market_context, usa
solo le analitiche che la dashboard già calcola.
"""
import os
import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional
import pandas as pd

# ── Persistence (SEPARATA dal long-vol) ─────────────────────────────────────────
PAPER_DIR        = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'paper_trades_short')
POSITIONS_CSV    = os.path.join(PAPER_DIR, 'positions_short.csv')
SKIPLOG_CSV      = os.path.join(PAPER_DIR, 'skip_log_short.csv')

# ── Account / risk (conto short-vol indipendente da $5.000) ─────────────────────
INITIAL_CAPITAL      = 5000.0
SPX_XSP_RATIO        = 10.0
# Rischio definito: la perdita massima di ogni struttura è capata. Limitiamo la
# perdita massima potenziale per trade a una % del conto (più prudente del cap
# sul premio del long-vol, perché qui la perdita > credito incassato).
MAX_RISK_PCT         = 8.0     # max perdita potenziale per trade = 8% ($400)
PREFER_RISK_PCT      = 5.0     # preferenza: strutture sotto il 5% ($250)

# ── Soglie di selezione SHORT-VOL (speculari al long-vol) ───────────────────────
IV_RICH_ABOVE        = 20.0    # vol "cara": IV sopra questa soglia → vendibile
IV_SKIP_BELOW        = 12.0    # vol troppo bassa: premio insufficiente, SKIP
VP_SELL_ABOVE        = 4.0     # vendi quando il vol premium è ALTO (IV>>RVol)
DTE_TARGET           = 30      # short-vol preferisce scadenze più brevi (più theta)
MIN_CONFIDENCE       = 0.40
HOLD_DTE_FLOOR       = 7       # chiudi entro 7 DTE (gamma risk esplode a scadenza)

# ── Regole di uscita short-vol (asimmetriche: stop più reattivo) ────────────────
PROFIT_TARGET_PCT    = 50.0    # chiudi quando hai incassato il 50% del credito max
STOP_LOSS_MULT       = 2.0     # stop: chiudi se la perdita = 2× il credito incassato
IV_CRASH_EXIT_MULT   = 1.50    # circuit breaker: se IV esplode +50% dall'entrata, chiudi

# ── Filtri aggiuntivi specifici short-vol (che il long-vol non ha) ──────────────
# Lo short-vol DEVE evitare di vendere prima di un'espansione di vol. Questi
# filtri bloccano la vendita in condizioni di stress in arrivo.
SKEW_BLOCK_ABOVE     = 1.0     # se lo skew si allarga oltre +1pt → stress → SKIP
TAIL_BLOCK_SPREAD    = 2.5     # se (taildex-putdex) > 2.5 → coda prezzata → SKIP

# ── Crash-protection filters (event window + term structure) ────────────────────
# I crash si concentrano attorno agli eventi macro e quando la term structure è
# invertita (backwardation = panico in corso). Bloccare la vendita in queste
# finestre evita i colpi alla fonte, al costo di saltare alcune giornate.
EVENT_BLOCK_DAYS_BEFORE = 1    # niente nuove vendite se un evento macro è entro N giorni
# Eventi macro ricorrenti ad alto impatto sull'indice. Date specifiche note
# vanno aggiunte qui (formato 'YYYY-MM-DD'). FOMC/CPI/NFP cambiano ogni mese,
# quindi si aggiornano periodicamente. Lista vuota = filtro eventi inattivo.
MACRO_EVENT_DATES = [
    # Esempio: '2026-07-30',  # FOMC
    # Aggiornare con le date reali di FOMC, CPI, NFP, OPEX trimestrali
]
BLOCK_ON_BACKWARDATION = True  # niente vendite se la term structure è invertita

# Risk-free + dividend (match dashboard)
R_FREE               = 0.053221
Q_DIV                = 0.014


# ══════════════════════════════════════════════════════════════════════════════
# Black-Scholes pricer (self-contained)
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


def legs_value(legs: list, S: float, T: float, sigma: float) -> float:
    """Mark-to-model net value of a list of legs (BUY +, SELL −), per contract.

    For a short structure the net value is typically NEGATIVE (we are net
    short premium): we received credit at entry and must pay to close. The
    P&L is (credit_received + net_value) where net_value is current cost to
    unwind. Handled explicitly in compute_unrealized().
    """
    total = 0.0
    for l in legs:
        sign = 1.0 if l.get('side', 'SELL') == 'BUY' else -1.0
        qty  = float(l.get('qty', 1))
        flag = 'c' if l.get('type') == 'CALL' else 'p'
        px   = bs_price(S, float(l['strike']), T, R_FREE, sigma, flag)
        total += sign * qty * px
    return total


# ══════════════════════════════════════════════════════════════════════════════
# Decision
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class ShortVolDecision:
    action:       str = 'SKIP'        # 'TRADE' | 'SKIP'
    strategy:     str = ''            # 'bull_put_spread'|'bear_call_spread'|'iron_condor'
    confidence:   float = 0.0
    target_dte:   int = DTE_TARGET
    xsp_short_k:  float = 0.0          # short strike (XSP)
    xsp_long_k:   float = 0.0          # protective long strike (XSP)
    credit:       float = 0.0          # net credit received (points, XSP)
    max_loss:     float = 0.0          # defined max loss (points, XSP)
    factors:      list = field(default_factory=list)
    conditions:   list = field(default_factory=list)
    legs:         list = field(default_factory=list)
    explanation:  str = ''
    skip_reason:  str = ''
    skip_category: str = ''

    @property
    def credit_usd(self) -> float:
        return self.credit * 100.0

    @property
    def max_loss_usd(self) -> float:
        return self.max_loss * 100.0

    @property
    def summary_html(self) -> str:
        if self.action == 'SKIP':
            return f"<b style='color:#9CA3AF'>⏸ SKIP</b> — {self.skip_reason}"
        color = '#10B981' if self.confidence > 0.6 else '#F59E0B'
        strat = {'bull_put_spread': '🐂 Bull Put Spread',
                 'bear_call_spread': '🐻 Bear Call Spread',
                 'iron_condor': '🦅 Iron Condor'}.get(self.strategy, self.strategy)
        return (f"<b style='color:{color}'>{strat}</b> · "
                f"DTE {self.target_dte} · credito {self.credit:.2f} "
                f"(${self.credit_usd:.0f}) · perdita max {self.max_loss:.2f} "
                f"(${self.max_loss_usd:.0f}) · "
                f"confidence <b>{int(self.confidence*100)}%</b>")


# ══════════════════════════════════════════════════════════════════════════════
# Signal evaluation (mirror of long-vol, opposite conditions)
# ══════════════════════════════════════════════════════════════════════════════
def _days_to_next_event(today: date = None) -> Optional[int]:
    """Days until the nearest upcoming macro event, or None if none configured
    / none upcoming. Used to block new short-vol sales right before events."""
    if not MACRO_EVENT_DATES:
        return None
    today = today or date.today()
    upcoming = []
    for s in MACRO_EVENT_DATES:
        try:
            ev = date.fromisoformat(s)
            if ev >= today:
                upcoming.append((ev - today).days)
        except Exception:
            continue
    return min(upcoming) if upcoming else None


def evaluate_signal(spot: float, regime: str, hhi: float,
                    vol_premium: Optional[float], atm_iv: Optional[float],
                    expected_move: Optional[float],
                    voldex: Optional[float] = None,
                    calldex: Optional[float] = None,
                    putdex: Optional[float] = None,
                    taildex: Optional[float] = None,
                    skew_trend: Optional[float] = None,
                    term_state: Optional[str] = None) -> ShortVolDecision:
    """Short-vol decision — sells volatility when it is RICH and the regime is
    stabilising (LONG gamma), with defined-risk structures.

    Mirror of the long-vol selector but with OPPOSITE conditions, PLUS extra
    safety filters the long-vol does not need: it must NOT sell into building
    stress (widening skew, priced tail, macro events, backwardation), because
    that is when vol expands and short-vol loses big. All VolDex-derived
    params are optional and the function degrades gracefully.

    term_state: 'contango'|'backwardation'|'flat'|None — from
    compute_term_structure_slope(). Backwardation blocks new sales.
    """
    d = ShortVolDecision()

    # ── Crash-protection block A: imminent macro event ────────────────────
    _ev_days = _days_to_next_event()
    if _ev_days is not None and _ev_days <= EVENT_BLOCK_DAYS_BEFORE:
        d.skip_category = 'Evento macro imminente'
        d.skip_reason = (f"Evento macro entro {_ev_days}g — niente vendite vol "
                         "a ridosso di eventi ad alto impatto")
        d.conditions = [{'label': 'Nessun evento imminente', 'met': False,
                         'detail': f"Evento macro tra {_ev_days} giorni — rischio "
                                   "di gap, lo short-vol si astiene"}]
        d.explanation = (
            f"❌ NESSUN TRADE. Un evento macro ad alto impatto è previsto entro "
            f"{_ev_days} giorno/i. Vendere vol a ridosso di FOMC/CPI/NFP espone a "
            f"gap improvvisi: il sistema si astiene fino a evento passato.")
        return d

    # ── Crash-protection block B: backwardation (panic in progress) ───────
    if BLOCK_ON_BACKWARDATION and term_state == 'backwardation':
        d.skip_category = 'Term structure invertita'
        d.skip_reason = ("Term structure in backwardation — vol breve > vol lunga, "
                         "stress acuto in corso, non vendere")
        d.conditions = [{'label': 'Term structure normale (contango)', 'met': False,
                         'detail': "Backwardation: la vol a breve supera quella a "
                                   "lunga — segnale di panico, sfavorevole allo short-vol"}]
        d.explanation = (
            "❌ NESSUN TRADE. La struttura a termine della volatilità è invertita "
            "(backwardation): la vol a breve è più alta di quella a lunga, segnale "
            "classico di stress acuto in corso. Vendere vol ora sarebbe vendere nel "
            "mezzo di un possibile crash. Il sistema si astiene.")
        return d

    if voldex is not None:
        iv = voldex
    else:
        iv = (atm_iv or 0.18) * 100 if (atm_iv and atm_iv < 1) else (atm_iv or 18.0)
    vp = vol_premium if vol_premium is not None else 0.0
    iv_source = "VolDex" if voldex is not None else "IV singolo strike"
    conditions = []
    factors = []

    def cond(label, met, detail):
        conditions.append({'label': label, 'met': bool(met), 'detail': detail})

    # ── Hard skip 1: vol too cheap — premium not worth the tail risk ──────
    if iv < IV_SKIP_BELOW:
        d.skip_category = 'IV troppo bassa'
        d.skip_reason = (f"IV {iv:.1f}% &lt; {IV_SKIP_BELOW}% — premio troppo magro "
                         "per giustificare il rischio di coda")
        cond("Vol vendibile", False,
             f"IV ({iv_source}) {iv:.1f}% < {IV_SKIP_BELOW}% — premio insufficiente")
        d.conditions = conditions
        d.explanation = (
            f"❌ NESSUN TRADE. La volatilità ({iv_source}: {iv:.1f}%) è troppo bassa: "
            f"il premio incassabile non compensa il rischio di coda dello short-vol. "
            f"Vendere vol così a buon mercato è il rischio peggiore per questa strategia.")
        return d

    # ── Hard skip 2: stress building — never sell into an expanding skew ──
    if skew_trend is not None and skew_trend > SKEW_BLOCK_ABOVE:
        d.skip_category = 'Stress in arrivo (skew)'
        d.skip_reason = (f"Skew in forte espansione (+{skew_trend:.2f}pt) — stress "
                         "in arrivo, vendere vol ora è pericoloso")
        cond("Niente stress in arrivo", False,
             f"Skew +{skew_trend:.2f}pt > {SKEW_BLOCK_ABOVE}pt — il mercato prezza "
             "protezione crescente, rischio di espansione vol")
        d.conditions = conditions
        d.explanation = (
            f"❌ NESSUN TRADE. Lo skew Put−Call si sta allargando rapidamente "
            f"(+{skew_trend:.2f}pt sopra la media): è il segnale che il mercato teme "
            f"un movimento. Vendere vol proprio prima di un'espansione è l'errore "
            f"classico dello short-vol. Il sistema si astiene.")
        return d

    # ── Hard skip 3: tail priced rich — crash risk priced in ──────────────
    if taildex is not None and putdex is not None and (taildex - putdex) > TAIL_BLOCK_SPREAD:
        d.skip_category = 'Coda prezzata (tail)'
        d.skip_reason = (f"Coda cara (TailDex−PutDex {taildex-putdex:.1f} > "
                         f"{TAIL_BLOCK_SPREAD}) — rischio crash prezzato")
        cond("Coda non prezzata", False,
             f"TailDex−PutDex {taildex-putdex:.1f} > {TAIL_BLOCK_SPREAD} — "
             "il mercato prezza un rischio estremo, sfavorevole allo short-vol")
        d.conditions = conditions
        d.explanation = (
            f"❌ NESSUN TRADE. La coda (TailDex) è prezzata molto sopra il corpo "
            f"(PutDex): il mercato sta pagando per proteggersi da un crash. Vendere "
            f"vol in queste condizioni espone proprio al rischio che il mercato teme.")
        return d

    score = 0.0

    # 1. Rich IV (0-0.30) — opposite of long-vol's "cheap IV"
    if iv > IV_RICH_ABOVE:
        score += min((iv - IV_RICH_ABOVE) / IV_RICH_ABOVE, 1.0) * 0.30
        factors.append(f"IV ({iv_source}) {iv:.1f}% &gt; {IV_RICH_ABOVE}% — vol cara, premio ricco")
        cond("IV cara", True,
             f"IV ({iv_source}) {iv:.1f}% > {IV_RICH_ABOVE}% — vol cara da vendere (+0.30 max)")
    elif iv > 16:
        score += 0.10
        cond("IV cara", False,
             f"IV ({iv_source}) {iv:.1f}% moderata (tra 16% e {IV_RICH_ABOVE}%) — solo +0.10")
    else:
        cond("IV cara", False,
             f"IV ({iv_source}) {iv:.1f}% non cara (≤ 16%) — nessun bonus")

    # 2. High vol premium (0-0.25) — opposite of long-vol's "low VP"
    if vp > VP_SELL_ABOVE:
        score += 0.25
        factors.append(f"Vol Premium {vp:+.1f}% alto — IV sopravvalutata vs RVol, vendibile")
        cond("Vol Premium alto", True,
             f"VP {vp:+.1f}% > {VP_SELL_ABOVE}% — IV sopravvalutata vs realizzata (+0.25)")
    elif vp > 2:
        score += 0.10
        cond("Vol Premium alto", False,
             f"VP {vp:+.1f}% intermedio (tra 2% e {VP_SELL_ABOVE}%) — solo +0.10")
    else:
        cond("Vol Premium alto", False,
             f"VP {vp:+.1f}% basso (≤ 2%) — l'IV non è cara vs realizzata, sfavorevole")

    # 3. LONG gamma regime (0-0.25) — opposite of long-vol's "SHORT gamma"
    if regime == 'LONG':
        score += 0.25
        factors.append("Regime LONG γ — i dealer stabilizzano, mercato in range")
        cond("Regime LONG gamma", True,
             "Dealer LONG gamma — smorzano i movimenti, favorevole per short-vol (+0.25)")
    else:
        cond("Regime LONG gamma", False,
             "Regime SHORT γ — i dealer amplificano i movimenti, sfavorevole per short-vol")
        factors.append("Regime SHORT γ — i dealer amplificano (sfavorevole allo short-vol)")

    # 4. High HHI bonus (pinning is GOOD for short vol — opposite of long-vol)
    if hhi >= 0.06:
        score += 0.10
        factors.append(f"HHI {hhi:.3f} alto — pinning, il prezzo resta fermo (favorevole)")
        cond("Pinning presente", True,
             f"HHI {hhi:.3f} ≥ 0.06 — GEX concentrato, il prezzo tende a restare fermo (+0.10)")
    elif hhi < 0.02:
        score -= 0.08
        cond("Pinning presente", False,
             f"HHI {hhi:.3f} < 0.02 — GEX distribuito, il prezzo può muoversi (−0.08)")
    else:
        cond("Pinning presente", False,
             f"HHI {hhi:.3f} intermedio (tra 0.02 e 0.06) — nessun bonus né penalità")

    # 5. Stable/contracting skew bonus (0-0.10) — opposite of long-vol
    if skew_trend is not None:
        if skew_trend < -0.3:
            score += 0.10
            factors.append(f"Skew in contrazione ({skew_trend:.2f}pt) — stress che rientra")
            cond("Skew stabile/in calo", True,
                 f"Skew {skew_trend:+.2f}pt sotto la media — protezione che rientra (+0.10)")
        elif skew_trend <= 0.5:
            score += 0.05
            cond("Skew stabile/in calo", True,
                 f"Skew stabile ({skew_trend:+.2f}pt) — nessun stress in arrivo (+0.05)")
        else:
            cond("Skew stabile/in calo", False,
                 f"Skew in lieve espansione ({skew_trend:+.2f}pt) — cautela")
    else:
        cond("Skew stabile/in calo", False, "Dato non disponibile (serve storico VolDex)")

    score = max(0.0, min(1.0, score))
    d.confidence = round(score, 3)
    d.conditions = conditions
    d.factors = factors

    if score < MIN_CONFIDENCE:
        d.action = 'SKIP'
        d.skip_category = 'Conviction bassa'
        d.skip_reason = (f"Conviction {score:.2f} &lt; {MIN_CONFIDENCE} — condizioni "
                         "non abbastanza favorevoli per vendere vol")
        d.explanation = (
            f"❌ NESSUN TRADE. Punteggio {score:.2f} sotto la soglia {MIN_CONFIDENCE}. "
            f"Le condizioni per vendere vol (vol cara, regime stabile, niente stress) "
            f"non sono sufficientemente allineate. Lo SKIP è la scelta corretta.")
        return d

    # ── Build the defined-risk structure ──────────────────────────────────
    d.action = 'TRADE'
    em = expected_move if (expected_move and expected_move > 0) else spot * (iv/100) * math.sqrt(DTE_TARGET/365)
    _build_structure(d, spot, iv, em, regime, taildex, putdex, calldex)
    return d


def _build_structure(d, spot, iv, em, regime, taildex, putdex, calldex):
    """Build a defined-risk short-vol structure scaled to XSP, respecting the
    max-risk budget. Chooses iron condor (neutral, range-bound) when no
    directional lean, else a single credit spread on the safer side.

    Strikes placed ~1 expected-move out (≈16-delta short strike), with the
    protective long leg one width further. Width chosen so max loss fits the
    budget. Everything in XSP space (SPX/10).
    """
    sigma = iv / 100.0
    T = d.target_dte / 365.0
    xsp_spot = spot / SPX_XSP_RATIO
    xsp_em = em / SPX_XSP_RATIO

    # Candidate widths (XSP points). Max loss per spread = (width - credit)*100.
    # Pick the width whose max loss fits MAX_RISK_PCT of capital.
    hard_risk = INITIAL_CAPITAL * MAX_RISK_PCT / 100.0   # $ max loss allowed

    def price_leg(K, flag):
        return bs_price(xsp_spot, K, T, R_FREE, sigma, flag)

    # short strikes ~1 EM out (≈1 sigma ≈ 16-delta), rounded to 5-pt XSP grid
    def round5(x):
        return round(x / 5.0) * 5.0

    put_short  = round5(xsp_spot - xsp_em)
    call_short = round5(xsp_spot + xsp_em)

    # Decide structure: iron condor by default (range-bound short vol).
    # If a slight directional lean exists (tail skew), use a single spread on
    # the safer side: rich put tail → bear call spread (sell upside instead).
    use_condor = True
    lean = None
    if taildex is not None and putdex is not None and (taildex - putdex) > 1.0:
        # downside is feared/priced → avoid selling puts → sell call side only
        use_condor = False
        lean = 'bear_call_spread'

    best = None
    for width in (5, 10, 15, 20, 25):
        if use_condor:
            pl = price_leg(put_short, 'p') - price_leg(put_short - width, 'p')
            cl = price_leg(call_short, 'c') - price_leg(call_short + width, 'c')
            credit = pl + cl
            max_loss = width - credit  # worst side caps loss (one side only can be ITM)
            legs = [
                {'side': 'SELL', 'type': 'PUT',  'strike': put_short,         'qty': 1},
                {'side': 'BUY',  'type': 'PUT',  'strike': put_short - width, 'qty': 1},
                {'side': 'SELL', 'type': 'CALL', 'strike': call_short,        'qty': 1},
                {'side': 'BUY',  'type': 'CALL', 'strike': call_short + width,'qty': 1},
            ]
            strat = 'iron_condor'
            sk_short, sk_long = put_short, put_short - width
        elif lean == 'bear_call_spread':
            credit = price_leg(call_short, 'c') - price_leg(call_short + width, 'c')
            max_loss = width - credit
            legs = [
                {'side': 'SELL', 'type': 'CALL', 'strike': call_short,         'qty': 1},
                {'side': 'BUY',  'type': 'CALL', 'strike': call_short + width, 'qty': 1},
            ]
            strat = 'bear_call_spread'
            sk_short, sk_long = call_short, call_short + width
        else:
            credit = price_leg(put_short, 'p') - price_leg(put_short - width, 'p')
            max_loss = width - credit
            legs = [
                {'side': 'SELL', 'type': 'PUT', 'strike': put_short,         'qty': 1},
                {'side': 'BUY',  'type': 'PUT', 'strike': put_short - width, 'qty': 1},
            ]
            strat = 'bull_put_spread'
            sk_short, sk_long = put_short, put_short - width

        if credit <= 0:
            continue
        max_loss_usd = max_loss * 100.0
        if max_loss_usd <= hard_risk:
            best = (strat, legs, credit, max_loss, sk_short, sk_long, max_loss_usd)
            # prefer the smallest width that fits & is under the soft target
            if max_loss_usd <= INITIAL_CAPITAL * PREFER_RISK_PCT / 100.0:
                break

    if best is None:
        d.action = 'SKIP'
        d.skip_category = 'Budget insufficiente'
        d.skip_reason = ("Nessuna struttura a rischio definito rientra nel budget "
                         f"di perdita max ({MAX_RISK_PCT:.0f}% = ${hard_risk:.0f})")
        d.explanation = (
            "❌ NESSUN TRADE. Anche lo spread più stretto avrebbe una perdita massima "
            f"oltre il budget (${hard_risk:.0f}). Il sistema si astiene invece di "
            "assumere un rischio sproporzionato per il conto.")
        return

    strat, legs, credit, max_loss, sk_short, sk_long, max_loss_usd = best
    d.strategy   = strat
    d.legs       = legs
    d.credit     = round(credit, 2)
    d.max_loss   = round(max_loss, 2)
    d.xsp_short_k = sk_short
    d.xsp_long_k  = sk_long
    _names = {'iron_condor': 'Iron Condor', 'bull_put_spread': 'Bull Put Spread',
              'bear_call_spread': 'Bear Call Spread'}
    d.explanation = (
        f"✅ VENDITA VOL — {_names.get(strat, strat)}. Incassa un credito di "
        f"{credit:.2f} punti (${credit*100:.0f}) con perdita massima definita di "
        f"{max_loss:.2f} punti (${max_loss_usd:.0f}). Struttura a rischio limitato: "
        f"profitta se il mercato resta nel range fino a scadenza. Vol cara ({iv:.1f}%) "
        f"+ regime stabile = condizioni favorevoli per incassare il decadimento.")


# ══════════════════════════════════════════════════════════════════════════════
# Position management (separate ledger)
# ══════════════════════════════════════════════════════════════════════════════
def _ensure_dir():
    os.makedirs(PAPER_DIR, exist_ok=True)


_POS_COLS = ['id', 'open_date', 'status', 'strategy', 'dte_target',
             'xsp_short_k', 'xsp_long_k', 'credit', 'max_loss', 'entry_iv',
             'entry_spot', 'legs_json', 'confidence',
             'close_date', 'close_reason', 'pnl_usd', 'peak_profit_pct']


def load_positions() -> pd.DataFrame:
    if os.path.exists(POSITIONS_CSV):
        try:
            return pd.read_csv(POSITIONS_CSV)
        except Exception:
            pass
    return pd.DataFrame(columns=_POS_COLS)


def _save_positions(df: pd.DataFrame):
    _ensure_dir()
    df.to_csv(POSITIONS_CSV, index=False)


def open_paper_position(decision: ShortVolDecision, spot: float,
                        entry_iv: float) -> dict:
    """Open a simulated short-vol position into the separate ledger."""
    import json
    if decision.action != 'TRADE':
        return {}
    df = load_positions()
    new_id = (int(df['id'].max()) + 1) if not df.empty else 1
    row = {
        'id': new_id, 'open_date': date.today().isoformat(), 'status': 'OPEN',
        'strategy': decision.strategy, 'dte_target': decision.target_dte,
        'xsp_short_k': decision.xsp_short_k, 'xsp_long_k': decision.xsp_long_k,
        'credit': decision.credit, 'max_loss': decision.max_loss,
        'entry_iv': round(entry_iv, 2), 'entry_spot': round(spot, 2),
        'legs_json': json.dumps(decision.legs), 'confidence': decision.confidence,
        'close_date': '', 'close_reason': '', 'pnl_usd': '',
        'peak_profit_pct': 0.0,
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    _save_positions(df)
    return row


def compute_unrealized(row, current_spot: float, current_iv: float,
                       days_held: int = 0) -> dict:
    """Mark-to-model P&L of an open short-vol position.

    P&L = credit_received − cost_to_close_now. cost_to_close = current net
    cost to buy back the structure = −legs_value (legs_value is BUY+/SELL−,
    so for a net-short structure it is negative; cost to close = −that).
    """
    import json
    try:
        legs = json.loads(row['legs_json'])
    except Exception:
        legs = []
    T = max((row['dte_target'] - days_held), 1) / 365.0
    xsp_spot = current_spot / SPX_XSP_RATIO
    sigma = current_iv / 100.0 if current_iv else (row.get('entry_iv', 18) / 100.0)
    net = legs_value(legs, xsp_spot, T, sigma)   # BUY+ SELL−
    cost_to_close = -net                          # pay to unwind
    credit = float(row['credit'])
    pnl_pts = credit - cost_to_close
    pnl_usd = pnl_pts * 100.0
    credit_usd = credit * 100.0
    profit_pct = (pnl_usd / credit_usd * 100.0) if credit_usd else 0.0
    return {'pnl_usd': pnl_usd, 'pnl_pts': pnl_pts, 'profit_pct': profit_pct,
            'cost_to_close': cost_to_close}


def mark_and_manage(current_spot: float, current_iv: float) -> list:
    """Re-evaluate open short-vol positions and apply exit rules.

    Exit rules (asymmetric — stop more reactive than the profit target):
      1. Profit target: captured ≥ PROFIT_TARGET_PCT of the credit → close
      2. Stop loss: loss ≥ STOP_LOSS_MULT × credit → close (defined-risk
         backstop fires before max loss)
      3. IV crash exit: IV ≥ IV_CRASH_EXIT_MULT × entry → close (vol exploded)
      4. DTE floor: ≤ HOLD_DTE_FLOOR days → close (gamma risk near expiry)
    """
    df = load_positions()
    if df.empty:
        return []
    results = []
    for idx, row in df[df['status'] == 'OPEN'].iterrows():
        try:
            held = (date.today() - date.fromisoformat(row['open_date'])).days
        except Exception:
            held = 0
        u = compute_unrealized(row, current_spot, current_iv, held)
        profit_pct = u['profit_pct']
        # track peak profit
        peak = max(float(row.get('peak_profit_pct', 0) or 0), profit_pct)
        df.at[idx, 'peak_profit_pct'] = round(peak, 1)

        close_reason = None
        if profit_pct >= PROFIT_TARGET_PCT:
            close_reason = f"Profit target {PROFIT_TARGET_PCT:.0f}% del credito raggiunto"
        elif u['pnl_pts'] <= -STOP_LOSS_MULT * float(row['credit']):
            close_reason = f"Stop loss: perdita ≥ {STOP_LOSS_MULT:.0f}× il credito"
        elif row.get('entry_iv') and current_iv >= IV_CRASH_EXIT_MULT * float(row['entry_iv']):
            close_reason = (f"IV crash exit: IV {current_iv:.0f}% ≥ "
                            f"{IV_CRASH_EXIT_MULT:.2f}× l'entrata ({row['entry_iv']:.0f}%)")
        elif (row['dte_target'] - held) <= HOLD_DTE_FLOOR:
            close_reason = f"DTE floor: ≤ {HOLD_DTE_FLOOR}g a scadenza"

        if close_reason:
            df.at[idx, 'status'] = 'CLOSED'
            df.at[idx, 'close_date'] = date.today().isoformat()
            df.at[idx, 'close_reason'] = close_reason
            df.at[idx, 'pnl_usd'] = round(u['pnl_usd'], 2)
            results.append({'id': row['id'], 'closed': True,
                            'reason': close_reason, 'pnl_usd': u['pnl_usd']})
    _save_positions(df)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Decision log + process metrics (separate from long-vol)
# ══════════════════════════════════════════════════════════════════════════════
def log_daily_decision(decision, spot: float, voldex=None,
                       expected_move_pts=None) -> None:
    """Append one row per engine run to the SHORT-vol decision log."""
    _ensure_dir()
    row = {
        'date': date.today().isoformat(),
        'action': decision.action,
        'strategy': decision.strategy or '',
        'confidence': decision.confidence,
        'skip_category': decision.skip_category or ('' if decision.action == 'TRADE' else 'Altro'),
        'skip_reason': (decision.skip_reason or '').replace('&lt;', '<').replace('&gt;', '>'),
        'spot': round(spot, 2) if spot else None,
        'voldex': voldex,
        'credit': decision.credit if decision.action == 'TRADE' else None,
        'max_loss': decision.max_loss if decision.action == 'TRADE' else None,
    }
    cols = list(row.keys())
    if os.path.exists(SKIPLOG_CSV):
        try:
            old = pd.read_csv(SKIPLOG_CSV)
            old = old[old['date'] != row['date']]
            allrows = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
        except Exception:
            allrows = pd.DataFrame([row], columns=cols)
    else:
        allrows = pd.DataFrame([row], columns=cols)
    allrows.to_csv(SKIPLOG_CSV, index=False)


def load_skip_log() -> pd.DataFrame:
    if os.path.exists(SKIPLOG_CSV):
        try:
            df = pd.read_csv(SKIPLOG_CSV)
            df['date'] = pd.to_datetime(df['date'])
            return df.sort_values('date').reset_index(drop=True)
        except Exception:
            pass
    return pd.DataFrame(columns=['date', 'action', 'strategy', 'confidence',
                                 'skip_category', 'skip_reason', 'spot', 'voldex',
                                 'credit', 'max_loss'])


def skip_log_summary() -> dict:
    df = load_skip_log()
    if df.empty:
        return {'total_days': 0, 'n_trade': 0, 'n_skip': 0,
                'skip_breakdown': {}, 'trade_rate': None}
    n_trade = int((df['action'] == 'TRADE').sum())
    n_skip = int((df['action'] == 'SKIP').sum())
    total = len(df)
    breakdown = (df[df['action'] == 'SKIP']['skip_category'].value_counts().to_dict())
    return {'total_days': total, 'n_trade': n_trade, 'n_skip': n_skip,
            'trade_rate': round(n_trade / total * 100, 1) if total else None,
            'skip_breakdown': breakdown}


def process_metrics() -> dict:
    """Process tracker for the short-vol system (mirror of long-vol)."""
    df = load_positions()
    closed = df[df['status'] == 'CLOSED'] if not df.empty else df
    open_p = df[df['status'] == 'OPEN'] if not df.empty else df
    n_closed = len(closed)
    n_open = len(open_p)

    realized = 0.0
    win_rate = None
    if n_closed > 0:
        pnl = pd.to_numeric(closed['pnl_usd'], errors='coerce').fillna(0)
        realized = float(pnl.sum())
        wins = int((pnl > 0).sum())
        win_rate = wins / n_closed

    return {
        'n_closed': n_closed, 'n_open': n_open,
        'realized_pnl': realized, 'win_rate': win_rate,
        'sample_target': 20, 'sample_pct': min(n_closed / 20 * 100, 100),
        'initial_capital': INITIAL_CAPITAL,
    }


def account_summary(current_spot: float = None, current_iv: float = None) -> dict:
    """Capital, realized + unrealized P&L, equity for the short-vol account."""
    df = load_positions()
    pm = process_metrics()
    realized = pm['realized_pnl']
    unreal = 0.0
    if current_spot and not df.empty:
        for _, row in df[df['status'] == 'OPEN'].iterrows():
            try:
                held = (date.today() - date.fromisoformat(row['open_date'])).days
            except Exception:
                held = 0
            u = compute_unrealized(row, current_spot, current_iv or row.get('entry_iv', 18), held)
            unreal += u['pnl_usd']
    equity = INITIAL_CAPITAL + realized + unreal
    return {'capital': INITIAL_CAPITAL, 'realized': realized,
            'unrealized': unreal, 'equity': equity,
            'n_open': pm['n_open'], 'n_closed': pm['n_closed']}
