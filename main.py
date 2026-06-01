"""
Trade Guardian Engine v3 — Dual-Agent AI Filter System
Agent A: Quantitative Analyst  (volume, volatility, market health)
Agent B: Technical Analyst      (RSI, MACD, EMA, Bollinger, MTF)
Supervisor: Conflict Resolution Matrix -> Simple English verdict
Data: CoinGecko (primary), CoinGlass (derivatives fallback to volume logic)
Zero pydantic. Python 3.11 safe.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
import httpx
import asyncio
import time
import statistics
import os
import math

app = FastAPI(title="Trade Guardian v3 — Dual-Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_POSITIONS = 5
positions = {}

GECKO  = "https://api.coingecko.com/api/v3"
CGLASS = "https://open-api.coinglass.com/public/v2"

# ── Cache layer (respect CoinGecko 10 req/s free tier) ───────────────────────
_cache = {}
CACHE_TTL = 55  # seconds

async def cached_get(url, params=None, headers=None):
    key = url + str(sorted((params or {}).items()))
    now = time.time()
    if key in _cache and now - _cache[key]["ts"] < CACHE_TTL:
        return _cache[key]["data"]
    async with httpx.AsyncClient(timeout=12, follow_redirects=True) as c:
        r = await c.get(url, params=params or {}, headers=headers or {"Accept": "application/json"})
        r.raise_for_status()
        data = r.json()
    _cache[key] = {"data": data, "ts": now}
    return data

# ── Symbol helpers ─────────────────────────────────────────────────────────────
GECKO_MAP = {
    "BTC":"bitcoin","ETH":"ethereum","SOL":"solana","BNB":"binancecoin",
    "XRP":"ripple","ADA":"cardano","DOGE":"dogecoin","AVAX":"avalanche-2",
    "DOT":"polkadot","MATIC":"matic-network","LINK":"chainlink","UNI":"uniswap",
    "LTC":"litecoin","ATOM":"cosmos","NEAR":"near","ARB":"arbitrum",
    "OP":"optimism","SUI":"sui","APT":"aptos","INJ":"injective-protocol",
    "TRX":"tron","TON":"the-open-network","PEPE":"pepe","WIF":"dogwifcoin",
    "BONK":"bonk","JUP":"jupiter-exchange-solana","SEI":"sei-network",
    "TIA":"celestia","WLD":"worldcoin-wld","RENDER":"render-token",
    "FET":"fetch-ai","NOT":"notcoin","ZK":"zksync","IO":"io-net",
    "HBAR":"hedera-hashgraph","VET":"vechain","ALGO":"algorand",
    "SAND":"the-sandbox","MANA":"decentraland","CHZ":"chiliz",
    "FTM":"fantom","ONE":"harmony","EGLD":"elrond-erd-2",
    "THETA":"theta-token","FIL":"filecoin","ICP":"internet-computer",
    "SHIB":"shiba-inu","FLOKI":"floki","BRETT":"brett",
}

def to_gecko_id(symbol):
    b = symbol.upper()
    for suf in ["USDT","BUSD","USD","BTC","ETH"]:
        if b.endswith(suf): b = b[:-len(suf)]
    b = b.replace("-SWAP","").replace("-","").strip()
    return GECKO_MAP.get(b, b.lower())

# ── CoinGecko fetchers ────────────────────────────────────────────────────────

async def gecko_market(gecko_id):
    """Current price, volume, market cap, 24h change."""
    data = await cached_get(GECKO + "/coins/markets", {
        "vs_currency": "usd",
        "ids": gecko_id,
        "order": "market_cap_desc",
        "per_page": 1,
        "page": 1,
        "price_change_percentage": "1h,24h,7d",
    })
    if not data:
        raise ValueError("Coin not found: " + gecko_id)
    return data[0]

async def gecko_ohlc(gecko_id, days=1):
    """OHLC candles. days=1 -> 30min candles. days=7 -> 4h candles."""
    data = await cached_get(GECKO + "/coins/{}/ohlc".format(gecko_id),
                            {"vs_currency": "usd", "days": str(days)})
    # returns [[ts, open, high, low, close], ...]
    return data

async def gecko_market_chart(gecko_id, days=1):
    """Prices + volumes as time series."""
    data = await cached_get(GECKO + "/coins/{}/market_chart".format(gecko_id),
                            {"vs_currency": "usd", "days": str(days)})
    return data  # {prices:[[ts,p],...], volumes:[[ts,v],...]}

async def gecko_coin_detail(gecko_id):
    """Full coin detail including community/dev data."""
    data = await cached_get(GECKO + "/coins/{}".format(gecko_id), {
        "localization": "false", "tickers": "false",
        "community_data": "false", "developer_data": "false",
    })
    return data

async def coinglass_funding(symbol):
    """Try CoinGlass for funding rate. Returns None if blocked."""
    try:
        data = await cached_get(CGLASS + "/funding_usd_margin_list",
                                {"symbol": symbol.upper().replace("USDT","")})
        rates = data.get("data", [])
        if rates:
            return float(rates[0].get("fundingRate", 0))
    except Exception:
        pass
    return None

# ── Indicator library ─────────────────────────────────────────────────────────

def ema_series(values, span):
    if not values: return []
    k = 2.0 / (span + 1)
    out = [values[0]]
    for v in values[1:]:
        out.append(v * k + out[-1] * (1.0 - k))
    return out

def ema_val(values, span):
    s = ema_series(values, span)
    return s[-1] if s else 0.0

def calc_rsi(closes, period=14):
    if len(closes) < period + 1: return 50.0
    gains  = [max(closes[i]-closes[i-1], 0) for i in range(1, len(closes))]
    losses = [max(closes[i-1]-closes[i], 0) for i in range(1, len(closes))]
    ag = statistics.mean(gains[-period:])
    al = statistics.mean(losses[-period:])
    if al == 0: return 100.0
    return round(100.0 - 100.0/(1.0 + ag/al), 2)

def calc_macd(closes):
    if len(closes) < 35:
        return {"macd": 0, "signal": 0, "hist": 0, "bull": False}
    e12 = ema_series(closes, 12)
    e26 = ema_series(closes, 26)
    ml  = [e12[i] - e26[i] for i in range(len(e26))]
    sl  = ema_series(ml, 9)
    hist = ml[-1] - sl[-1]
    return {"macd": round(ml[-1],8), "signal": round(sl[-1],8),
            "hist": round(hist,8), "bull": ml[-1] > sl[-1]}

def calc_bollinger(closes, period=20, std_dev=2):
    if len(closes) < period:
        p = closes[-1] if closes else 0
        return {"upper": p, "mid": p, "lower": p, "pct_b": 0.5}
    window = closes[-period:]
    mid    = statistics.mean(window)
    sd     = statistics.stdev(window) if len(window) > 1 else 0
    upper  = mid + std_dev * sd
    lower  = mid - std_dev * sd
    price  = closes[-1]
    pct_b  = (price - lower) / (upper - lower) if upper != lower else 0.5
    return {"upper": round(upper,8), "mid": round(mid,8),
            "lower": round(lower,8), "pct_b": round(pct_b,4)}

def calc_volatility(closes, period=14):
    if len(closes) < 2: return 0.0
    returns = [abs(closes[i]/closes[i-1]-1) for i in range(1, len(closes))]
    return round(statistics.mean(returns[-period:]) * 100, 4)  # %

def calc_volume_spike(volumes):
    """Returns ratio of latest volume vs 20-period average."""
    if len(volumes) < 5: return 1.0
    avg = statistics.mean(volumes[:-1][-20:]) if len(volumes) > 1 else volumes[0]
    return round(volumes[-1] / avg, 3) if avg else 1.0

def support_resistance(closes, highs, lows, lookback=20):
    """Simple swing high/low support & resistance."""
    if len(closes) < lookback:
        p = closes[-1] if closes else 0
        return {"support": round(p*0.97,6), "resistance": round(p*1.03,6)}
    recent_highs = highs[-lookback:]
    recent_lows  = lows[-lookback:]
    resistance   = max(recent_highs)
    support      = min(recent_lows)
    return {"support": round(support,8), "resistance": round(resistance,8)}

# ── AGENT A: Quantitative Analyst ─────────────────────────────────────────────

def analyze_quant(market, chart_1d, funding_rate, direction):
    """
    Returns: {"score": 0-100, "verdict": "valid|risky|dead",
              "label": str, "detail": str, "warnings": [str]}
    """
    score    = 50
    warnings = []
    details  = []

    prices  = [p[1] for p in chart_1d.get("prices",  [])]
    volumes = [v[1] for v in chart_1d.get("volumes", [])]

    if not prices or not volumes:
        return {"score":50,"verdict":"risky","label":"INSUFFICIENT DATA",
                "detail":"Not enough market history available.","warnings":["Limited data."]}

    # 1. Volume spike analysis
    vol_spike = calc_volume_spike(volumes)
    price_chg_24h = market.get("price_change_percentage_24h") or 0
    price_chg_1h  = market.get("price_change_percentage_1h_in_currency") or 0

    if vol_spike > 2.0:
        if abs(price_chg_24h) > 3:
            score += 20
            details.append("Strong volume spike ({:.1f}x) confirms price move.".format(vol_spike))
        else:
            # High volume, low price move = churning/exhaustion
            score -= 20
            warnings.append("Volume {:.1f}x average but price barely moved — exhaustion/churn signal.".format(vol_spike))
    elif vol_spike < 0.5:
        score -= 25
        warnings.append("Volume only {:.1f}x average. Move lacks conviction — possible fakeout.".format(vol_spike))
        details.append("Low volume detected. Price action not backed by participation.")
    else:
        score += 5
        details.append("Volume at {:.1f}x average — normal participation.".format(vol_spike))

    # 2. Volatility check
    volatility = calc_volatility(prices)
    if volatility > 5.0:
        score -= 15
        warnings.append("Extreme volatility ({:.1f}%). High risk of sudden reversal.".format(volatility))
    elif volatility > 2.5:
        score -= 5
        details.append("Elevated volatility ({:.1f}%). Widen stops.".format(volatility))
    elif volatility < 0.3:
        score -= 10
        warnings.append("Near-zero volatility. Market asleep — breakout or manipulation risk.")
    else:
        score += 10
        details.append("Healthy volatility ({:.1f}%).".format(volatility))

    # 3. Funding rate (from CoinGlass or fallback)
    if funding_rate is not None:
        fr_pct = funding_rate * 100
        if fr_pct > 0.08:
            score -= 15
            warnings.append("Funding rate extremely high ({:.4f}%). Longs are overheated. Short squeeze risk.".format(fr_pct))
        elif fr_pct < -0.05:
            score -= 10
            warnings.append("Funding rate deeply negative ({:.4f}%). Shorts overloaded. Long squeeze risk.".format(fr_pct))
        elif abs(fr_pct) < 0.02:
            score += 10
            details.append("Neutral funding rate ({:.4f}%). No directional crowding.".format(fr_pct))
        else:
            score += 5
    else:
        # Fallback: infer from volume + price divergence
        if vol_spike > 2.0 and abs(price_chg_1h) < 0.5:
            score -= 15
            warnings.append("Derivatives data unavailable. Volume surge with flat price suggests overheating.")
        details.append("Funding rate unavailable — using volume/price divergence logic.")

    # 4. Market cap health
    market_cap = market.get("market_cap", 0) or 0
    if market_cap < 10_000_000:
        score -= 20
        warnings.append("Very low market cap (< $10M). Extremely susceptible to manipulation.")
    elif market_cap < 100_000_000:
        score -= 10
        warnings.append("Small cap coin (< $100M). Higher volatility and manipulation risk.")
    elif market_cap > 1_000_000_000:
        score += 10
        details.append("Large cap coin (> $1B). More stable and liquid.")

    # 5. 24h price direction alignment with trade direction
    if direction == "LONG":
        if price_chg_24h > 5:   score += 10
        elif price_chg_24h < -5: score -= 15
    else:
        if price_chg_24h < -5:  score += 10
        elif price_chg_24h > 5:  score -= 15

    score = max(0, min(100, score))

    if score >= 65:
        verdict = "valid"
        label   = "VOLUME SUPPORTS MOVE"
    elif score >= 40:
        verdict = "risky"
        label   = "RISKY MARKET CONDITIONS"
    else:
        verdict = "dead"
        label   = "NO MARKET FUEL"

    return {
        "score":    score,
        "verdict":  verdict,
        "label":    label,
        "detail":   " | ".join(details) if details else "Analysis complete.",
        "warnings": warnings,
        "vol_spike":   vol_spike,
        "volatility":  volatility,
        "funding_rate": funding_rate,
    }

# ── AGENT B: Technical Analyst ─────────────────────────────────────────────────

def analyze_technical(ohlc_short, ohlc_long, current_price, direction):
    """
    ohlc_short: recent candles (approx 5m-1h equivalent)
    ohlc_long:  weekly candles (4h equivalent)
    Returns: {"score":0-100,"verdict":"valid|risky|dead","label":str,...}
    """
    score    = 50
    warnings = []
    details  = []

    def parse_ohlc(raw):
        if not raw: return [], [], [], []
        opens  = [float(c[1]) for c in raw]
        highs  = [float(c[2]) for c in raw]
        lows   = [float(c[3]) for c in raw]
        closes = [float(c[4]) for c in raw]
        return opens, highs, lows, closes

    o_s, h_s, l_s, c_s = parse_ohlc(ohlc_short)
    o_l, h_l, l_l, c_l = parse_ohlc(ohlc_long)

    if not c_s:
        return {"score":50,"verdict":"risky","label":"INSUFFICIENT CHART DATA",
                "detail":"Not enough OHLC data.","warnings":["Limited candle history."]}

    # ── Short-term indicators (recent candles ~ 1h view)
    rsi_short = calc_rsi(c_s, 14)
    macd_s    = calc_macd(c_s)
    bb_s      = calc_bollinger(c_s)
    e9_s      = ema_val(c_s, 9)
    e21_s     = ema_val(c_s, 21)
    e50_s     = ema_val(c_s, min(50, len(c_s)-1)) if len(c_s) > 10 else e21_s
    sr_s      = support_resistance(c_s, h_s, l_s)

    # ── Long-term indicators (weekly candles ~ 4h view)
    rsi_long  = calc_rsi(c_l, 14) if c_l else 50.0
    macd_l    = calc_macd(c_l)    if c_l else {"bull": False}
    e21_l     = ema_val(c_l, 21)  if c_l else current_price
    sr_l      = support_resistance(c_l, h_l, l_l) if c_l else sr_s

    price = current_price

    # ── 1. EMA Ribbon alignment
    if direction == "LONG":
        ema_aligned = price > e9_s > e21_s
        ema_stack   = e9_s > e21_s > e50_s
    else:
        ema_aligned = price < e9_s < e21_s
        ema_stack   = e9_s < e21_s < e50_s

    if ema_aligned and ema_stack:
        score += 20
        details.append("EMA ribbon fully aligned (9/21/50) with {} direction.".format(direction))
    elif ema_aligned:
        score += 10
        details.append("EMA 9/21 aligned. EMA50 not yet confirmed.")
    else:
        score -= 15
        warnings.append("EMA ribbon not aligned. Price fighting EMAs — weak structure.")

    # ── 2. RSI analysis (short-term)
    if direction == "LONG":
        if rsi_short > 70:
            score -= 20
            warnings.append("RSI overbought at {:.1f}. High reversal risk.".format(rsi_short))
        elif rsi_short > 55:
            score += 15
            details.append("RSI bullish momentum at {:.1f}.".format(rsi_short))
        elif rsi_short > 45:
            score += 5
            details.append("RSI neutral at {:.1f}. Momentum building.".format(rsi_short))
        elif rsi_short < 30:
            score += 10  # oversold = potential bounce
            details.append("RSI oversold at {:.1f}. Bounce zone for longs.".format(rsi_short))
        else:
            score -= 10
            warnings.append("RSI weak at {:.1f}. Momentum not in your favor.".format(rsi_short))
    else:
        if rsi_short < 30:
            score -= 20
            warnings.append("RSI oversold at {:.1f}. Bounce likely against your short.".format(rsi_short))
        elif rsi_short < 45:
            score += 15
            details.append("RSI bearish at {:.1f}. Supports short position.".format(rsi_short))
        elif rsi_short > 70:
            score += 10
            details.append("RSI overbought at {:.1f}. Supports short.".format(rsi_short))
        else:
            score -= 10
            warnings.append("RSI neutral at {:.1f}. No bearish conviction yet.".format(rsi_short))

    # ── 3. MACD cross (short-term)
    if direction == "LONG":
        if macd_s["bull"]:
            score += 15
            details.append("MACD bullish cross confirmed.")
        else:
            score -= 10
            warnings.append("MACD bearish. Momentum not supporting long.")
    else:
        if not macd_s["bull"]:
            score += 15
            details.append("MACD bearish cross confirmed.")
        else:
            score -= 10
            warnings.append("MACD bullish. Momentum fighting your short.")

    # ── 4. Bollinger Band position
    pb = bb_s["pct_b"]
    if direction == "LONG":
        if pb > 0.9:
            score -= 15
            warnings.append("Price at upper Bollinger Band ({:.0f}%). Overbought zone.".format(pb*100))
        elif pb < 0.2:
            score += 10
            details.append("Price at lower Bollinger Band — potential bounce zone.")
        elif 0.4 < pb < 0.7:
            score += 8
            details.append("Price mid-Bollinger. Healthy momentum zone.")
    else:
        if pb < 0.1:
            score -= 15
            warnings.append("Price at lower Bollinger Band ({:.0f}%). Oversold — short squeeze risk.".format(pb*100))
        elif pb > 0.8:
            score += 10
            details.append("Price at upper Bollinger — potential short entry zone.")

    # ── 5. MTF coherence: short vs long timeframe RSI
    rsi_conflict = (direction == "LONG" and rsi_long < 45) or \
                   (direction == "SHORT" and rsi_long > 55)
    if rsi_conflict:
        score -= 20
        warnings.append("RULE 1 VIOLATED: Short-term signal conflicts with long-term trend. Counter-trend trap risk.")
    else:
        score += 10
        details.append("Timeframes agree on directional bias.")

    # ── 6. Support / Resistance proximity
    dist_to_resistance = ((sr_s["resistance"] - price) / price) * 100
    dist_to_support    = ((price - sr_s["support"]) / price) * 100
    if direction == "LONG":
        if dist_to_resistance < 1.5:
            score -= 20
            warnings.append("Price only {:.1f}% from resistance at ${:.4f}. Low upside before wall.".format(
                dist_to_resistance, sr_s["resistance"]))
        elif dist_to_resistance > 5:
            score += 10
            details.append("Clear runway to resistance at ${:.4f} ({:.1f}% away).".format(
                sr_s["resistance"], dist_to_resistance))
    else:
        if dist_to_support < 1.5:
            score -= 20
            warnings.append("Price only {:.1f}% from support at ${:.4f}. Short may bounce here.".format(
                dist_to_support, sr_s["support"]))
        elif dist_to_support > 5:
            score += 10
            details.append("Clear drop to support at ${:.4f} ({:.1f}% away).".format(
                sr_s["support"], dist_to_support))

    score = max(0, min(100, score))

    if score >= 65:
        verdict = "valid"
        label   = "CHART STRUCTURE CONFIRMED"
    elif score >= 40:
        verdict = "risky"
        label   = "MIXED TECHNICAL SIGNALS"
    else:
        verdict = "dead"
        label   = "CHART STRUCTURE BROKEN"

    return {
        "score":    score,
        "verdict":  verdict,
        "label":    label,
        "detail":   " | ".join(details) if details else "Analysis complete.",
        "warnings": warnings,
        "rsi_short": rsi_short,
        "rsi_long":  rsi_long,
        "macd_bull": macd_s["bull"],
        "bb_pct_b":  pb,
        "ema9":      round(e9_s, 6),
        "ema21":     round(e21_s, 6),
        "ema50":     round(e50_s, 6),
        "support":   sr_s["support"],
        "resistance":sr_s["resistance"],
    }

# ── SUPERVISOR: Conflict Resolution Matrix ─────────────────────────────────────

MATRIX = {
    ("valid",  "valid"):  ("HIGH_CONFIDENCE", "green",  "EXECUTE",      "All systems go. Strong volume and clear chart path."),
    ("valid",  "risky"):  ("CAUTION",          "yellow", "WAIT",         "Good market data, but bad timing on the chart. Wait for confirmation."),
    ("risky",  "valid"):  ("CAUTION",          "yellow", "REDUCE_SIZE",  "Chart looks good, but market conditions are risky. Trade smaller."),
    ("dead",   "valid"):  ("TRAP_ALERT",        "red",    "DO_NOT_TRADE", "Chart looks good, but volume is missing. This move is likely a fakeout."),
    ("valid",  "dead"):   ("RESISTANCE_WALL",   "red",    "DO_NOT_TRADE", "Volume is strong, but price is hitting a structural wall. Wait for breakout."),
    ("dead",   "dead"):   ("NO_TRADE",          "red",    "STAY_OUT",     "Dangerous conditions on both fronts. Preserve capital."),
    ("risky",  "risky"):  ("CAUTION",           "yellow", "WAIT",         "Mixed signals across the board. No clean edge identified."),
    ("valid",  "valid"):  ("HIGH_CONFIDENCE",   "green",  "EXECUTE",      "All systems aligned."),
    ("dead",   "risky"):  ("NO_TRADE",          "red",    "STAY_OUT",     "No volume backing and weak chart. Dangerous environment."),
    ("risky",  "dead"):   ("TRAP_ALERT",        "red",    "DO_NOT_TRADE", "Chart broken with risky market. Do not force a trade here."),
}

def merge_verdicts(quant, ta, direction, current_price, symbol):
    qv = quant["verdict"]
    tv = ta["verdict"]
    key = (qv, tv)
    conf, color, action, base_msg = MATRIX.get(key, ("CAUTION","yellow","WAIT","Proceed with caution."))

    # Build plain English verdict
    plain_english = generate_plain_english(conf, color, quant, ta, direction, current_price, base_msg)
    safe_passage  = generate_safe_passage(conf, color, ta, current_price, direction)
    risk_warning  = generate_risk_warning(quant, ta, conf)

    return {
        "confidence":    conf,
        "color":         color,
        "action":        action,
        "quant_score":   quant["score"],
        "ta_score":      ta["score"],
        "quant_verdict": qv,
        "ta_verdict":    tv,
        "quant_label":   quant["label"],
        "ta_label":      ta["label"],
        "plain_english": plain_english,
        "safe_passage":  safe_passage,
        "risk_warning":  risk_warning,
        "all_warnings":  quant["warnings"] + ta["warnings"],
    }

def generate_plain_english(conf, color, quant, ta, direction, price, base_msg):
    d = "up" if direction == "LONG" else "down"
    opp = "buying" if direction == "LONG" else "selling"
    if conf == "HIGH_CONFIDENCE":
        return "The price looks ready to go {}. Big traders are {} and the chart confirms the move. It looks safe to enter.".format(d, opp)
    elif conf == "CAUTION" and quant["verdict"] == "valid":
        return "The market has good volume and momentum, but the chart is not yet showing a clean signal. The timing is off right now."
    elif conf == "CAUTION" and ta["verdict"] == "valid":
        return "The chart pattern looks promising, but the broader market conditions are risky right now. Trade carefully."
    elif conf == "TRAP_ALERT":
        return "This is a trap. The price looks like it is moving {}, but there is not enough volume to support it. Do not buy this move.".format(d)
    elif conf == "RESISTANCE_WALL":
        return "There is strong volume behind this move, but the price is hitting a major wall. It is likely to get rejected here."
    elif conf == "NO_TRADE":
        return "Both our analysts agree: this is not the right moment. The market is dangerous and the chart is broken. Stay out."
    else:
        return "Mixed signals detected. " + base_msg

def generate_safe_passage(conf, color, ta, price, direction):
    support    = ta.get("support",    price * 0.97)
    resistance = ta.get("resistance", price * 1.03)
    if color == "green":
        return "You may proceed with your planned position size. Suggested stop loss below ${:.4f} (support level).".format(support)
    elif conf == "CAUTION" and direction == "LONG":
        return "Wait for the price to pull back to ${:.4f} (support level) before entering, or wait for a confirmed 15-minute candle close above current price.".format(round(support, 4))
    elif conf == "CAUTION" and direction == "SHORT":
        return "Wait for the price to rally to ${:.4f} (resistance level) before shorting, or reduce your position size by half.".format(round(resistance, 4))
    elif conf in ("TRAP_ALERT", "RESISTANCE_WALL", "NO_TRADE"):
        if direction == "LONG":
            return "Do not enter now. Wait for the price to drop to ${:.4f} (support) and get 2 consecutive bullish candle closes on the 15-minute chart before trying again.".format(round(support, 4))
        else:
            return "Do not enter now. Wait for the price to rally to ${:.4f} (resistance) and watch for 2 bearish candle closes on the 15-minute chart before shorting.".format(round(resistance, 4))
    return "Reduce position size by 50% and place a tight stop loss."

def generate_risk_warning(quant, ta, conf):
    all_w = quant["warnings"] + ta["warnings"]
    if not all_w:
        if conf == "HIGH_CONFIDENCE":
            return "No major risks detected. Monitor for sudden market-wide events (news, macro)."
        return "Standard market risks apply. Always use a stop loss."
    # Pick the most severe warning
    priority = ["manipulation","exhaustion","overbought","oversold","extreme","fakeout",
                "overheated","conflict","divergen","resistance","volume"]
    for kw in priority:
        for w in all_w:
            if kw.lower() in w.lower():
                return "Risk: " + w
    return "Risk: " + all_w[0]

# ── Master update function ────────────────────────────────────────────────────

async def run_dual_agent(symbol):
    s = positions.get(symbol)
    if s is None: return
    gid       = s["gecko_id"]
    direction = s["direction"]

    try:
        # Parallel fetch
        market_data, chart_1d, chart_7d, ohlc_1d, ohlc_7d = await asyncio.gather(
            gecko_market(gid),
            gecko_market_chart(gid, days=1),
            gecko_market_chart(gid, days=7),
            gecko_ohlc(gid, days=1),
            gecko_ohlc(gid, days=7),
        )

        current_price = float(market_data.get("current_price", s["entry_price"]))
        s["current_price"] = current_price

        # PnL
        ep, sz = s["entry_price"], s["position_size"]
        if direction == "LONG":
            s["pnl_usd"] = round((current_price - ep) * sz, 4)
            s["pnl_pct"] = round(((current_price - ep) / ep) * 100, 4)
        else:
            s["pnl_usd"] = round((ep - current_price) * sz, 4)
            s["pnl_pct"] = round(((ep - current_price) / ep) * 100, 4)

        # Try derivatives data
        funding_rate = await coinglass_funding(symbol)

        # Run agents
        quant_result = analyze_quant(market_data, chart_1d, funding_rate, direction)
        ta_result    = analyze_technical(ohlc_1d, ohlc_7d, current_price, direction)

        # Supervisor merge
        verdict = merge_verdicts(quant_result, ta_result, direction, current_price, symbol)

        # Store
        s["market_data"]    = {
            "price":         current_price,
            "volume_24h":    market_data.get("total_volume", 0),
            "market_cap":    market_data.get("market_cap", 0),
            "change_24h":    market_data.get("price_change_percentage_24h", 0),
            "change_1h":     market_data.get("price_change_percentage_1h_in_currency", 0),
        }
        s["quant"]          = quant_result
        s["ta"]             = ta_result
        s["verdict"]        = verdict
        s["signal_status"]  = "LIVE"
        s["last_update"]    = time.time()

    except Exception as e:
        s["verdict"] = s.get("verdict", {})
        s["verdict"]["plain_english"] = "Data error: {}. Retrying next cycle.".format(str(e))
        s["signal_status"] = "ERROR"

async def monitor_loop(symbol):
    while symbol in positions:
        await run_dual_agent(symbol)
        await asyncio.sleep(60)

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "monitor_dashboard.html")
    if os.path.exists(p):
        return FileResponse(p, media_type="text/html")
    return HTMLResponse("<h2>monitor_dashboard.html not found</h2>", status_code=404)

@app.post("/init")
async def init_pos(request: Request):
    body          = await request.json()
    symbol        = str(body.get("symbol","")).strip().upper()
    direction     = str(body.get("direction","LONG")).strip().upper()
    entry_price   = float(body.get("entry_price", 0))
    position_size = float(body.get("position_size", 0))

    if not symbol:              raise HTTPException(400, "symbol required")
    if entry_price <= 0:        raise HTTPException(400, "entry_price must be > 0")
    if position_size <= 0:      raise HTTPException(400, "position_size must be > 0")
    if direction not in ("LONG","SHORT"): raise HTTPException(400, "direction must be LONG or SHORT")
    if len(positions) >= MAX_POSITIONS and symbol not in positions:
        raise HTTPException(400, "Maximum {} positions reached".format(MAX_POSITIONS))

    gecko_id = to_gecko_id(symbol)

    # Validate coin exists
    try:
        mkt = await gecko_market(gecko_id)
        price = float(mkt.get("current_price", entry_price))
    except Exception as e:
        raise HTTPException(400, "Cannot find '{}' on CoinGecko. Try the full coin name (e.g. 'bitcoin'). Error: {}".format(gecko_id, str(e)))

    positions[symbol] = {
        "symbol":       symbol,
        "gecko_id":     gecko_id,
        "direction":    direction,
        "entry_price":  entry_price,
        "position_size":position_size,
        "started_at":   time.time(),
        "last_update":  time.time(),
        "current_price":price,
        "pnl_usd":      0.0,
        "pnl_pct":      0.0,
        "market_data":  {},
        "quant":        {"score":50,"verdict":"risky","label":"INITIALISING","detail":"","warnings":[]},
        "ta":           {"score":50,"verdict":"risky","label":"INITIALISING","detail":"","warnings":[]},
        "verdict": {
            "confidence":"CAUTION","color":"yellow","action":"WAIT",
            "plain_english":"Agents are analysing the market. Please wait ~60 seconds for first results.",
            "safe_passage":"Analysis in progress...",
            "risk_warning":"Analysis in progress...",
            "quant_score":50,"ta_score":50,
            "quant_verdict":"risky","ta_verdict":"risky",
            "quant_label":"INITIALISING","ta_label":"INITIALISING",
            "all_warnings":[],
        },
        "signal_status":"PENDING",
    }

    asyncio.create_task(monitor_loop(symbol))
    return {"status":"initialized","symbol":symbol,"gecko_id":gecko_id}

@app.get("/all")
async def get_all():
    return {
        "count": len(positions),
        "max":   MAX_POSITIONS,
        "positions": {
            sym: {
                "symbol":        s["symbol"],
                "direction":     s["direction"],
                "current_price": s["current_price"],
                "pnl_pct":       s["pnl_pct"],
                "pnl_usd":       s["pnl_usd"],
                "confidence":    s["verdict"].get("confidence","—"),
                "color":         s["verdict"].get("color","grey"),
                "action":        s["verdict"].get("action","—"),
                "signal_status": s["signal_status"],
                "last_update":   s["last_update"],
            }
            for sym, s in positions.items()
        }
    }

@app.get("/state/{symbol}")
async def get_state(symbol: str):
    sym = symbol.strip().upper()
    if sym not in positions:
        raise HTTPException(404, "No position for {}".format(sym))
    return positions[sym]

@app.delete("/close/{symbol}")
async def close_pos(symbol: str):
    sym = symbol.strip().upper()
    if sym in positions:
        del positions[sym]
        return {"status":"closed","symbol":sym}
    raise HTTPException(404, "No position for {}".format(sym))

@app.get("/health")
async def health():
    return {"status":"ok","active":len(positions),"max":MAX_POSITIONS}
