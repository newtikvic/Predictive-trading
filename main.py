"""
Trade Guardian Engine v4
Primary data:   MEXC  (price, OHLCV, order book, 24h stats)
Secondary data: OKX   (confirmation candles, funding rate, OI)
Market intel:   CoinGecko (market cap, 24h change sentiment)
Architecture:   Dual-Agent AI Filter
  Agent A - Quant: volume, volatility, market health, derivatives
  Agent B - TA:    RSI, MACD, EMA(9/21/50), Bollinger, support/resistance, MTF
  Supervisor:      Conflict resolution matrix -> Simple English verdict
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

app = FastAPI(title="Trade Guardian Engine v4")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_POSITIONS = 5
positions = {}

# ── API bases ─────────────────────────────────────────────────────────────────
MEXC   = "https://api.mexc.com"
OKX    = "https://www.okx.com"
GECKO  = "https://api.coingecko.com/api/v3"

# ── Request cache (55s TTL — respects free tier rate limits) ──────────────────
_cache = {}
CACHE_TTL = 55

async def get(url, params=None, headers=None, timeout=12):
    key = url + str(sorted((params or {}).items()))
    now = time.time()
    if key in _cache and now - _cache[key]["ts"] < CACHE_TTL:
        return _cache[key]["data"]
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as c:
        r = await c.get(url, params=params or {},
                        headers=headers or {"Accept": "application/json",
                                           "User-Agent": "TradeGuardian/4.0"})
        r.raise_for_status()
        data = r.json()
    _cache[key] = {"data": data, "ts": now}
    return data

# ── Symbol converters ─────────────────────────────────────────────────────────

def to_mexc(symbol):
    """BTCUSDT -> BTCUSDT (MEXC uses standard format)"""
    s = symbol.upper()
    if s.endswith("USDT") or s.endswith("BUSD"):
        return s
    return s + "USDT"

def to_okx(symbol):
    """BTCUSDT -> BTC-USDT-SWAP"""
    s = symbol.upper()
    for suf in ["USDT", "BUSD", "USD"]:
        if s.endswith(suf):
            return s[:-len(suf)] + "-USDT-SWAP"
    return s + "-USDT-SWAP"

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
    "FTM":"fantom","FIL":"filecoin","ICP":"internet-computer",
    "SHIB":"shiba-inu","FLOKI":"floki","GRT":"the-graph",
    "AAVE":"aave","MKR":"maker","SNX":"synthetix-network-token",
    "CRV":"curve-dao-token","LDO":"lido-dao","RPL":"rocket-pool",
    "RUNE":"thorchain","KAVA":"kava","MINA":"mina-protocol",
}

def to_gecko(symbol):
    b = symbol.upper()
    for suf in ["USDT", "BUSD", "USD"]:
        if b.endswith(suf):
            b = b[:-len(suf)]
    b = b.replace("-SWAP", "").replace("-", "").strip()
    return GECKO_MAP.get(b, b.lower())

# ── MEXC fetchers (primary) ───────────────────────────────────────────────────

async def mexc_ticker(sym):
    """Current price."""
    d = await get(MEXC + "/api/v3/ticker/price", {"symbol": sym})
    return float(d["price"])

async def mexc_24h(sym):
    """24h stats: price, volume, change."""
    return await get(MEXC + "/api/v3/ticker/24hr", {"symbol": sym})

async def mexc_klines(sym, interval, limit=60):
    """
    OHLCV candles.
    interval: 1m,5m,15m,30m,1h,4h,1d
    Returns list oldest-first: [open_time, open, high, low, close, volume, ...]
    """
    raw = await get(MEXC + "/api/v3/klines",
                    {"symbol": sym, "interval": interval, "limit": limit})
    return raw  # already oldest-first from MEXC

async def mexc_depth(sym, limit=50):
    """Order book."""
    return await get(MEXC + "/api/v3/depth", {"symbol": sym, "limit": limit})

# ── OKX fetchers (secondary / confirmation) ───────────────────────────────────

async def okx_get(path, params=None):
    d = await get(OKX + path, params or {})
    if str(d.get("code")) != "0":
        raise ValueError("OKX {}: {}".format(d.get("code"), d.get("msg", "")))
    return d["data"]

async def okx_candles(inst, bar, limit=60):
    """OKX candles for confirmation. Returns oldest-first."""
    d = await okx_get("/api/v5/market/candles",
                      {"instId": inst, "bar": bar, "limit": str(limit)})
    return list(reversed(d))

async def okx_funding(inst):
    """Funding rate from OKX perps."""
    try:
        d = await okx_get("/api/v5/public/funding-rate", {"instId": inst})
        return float(d[0]["fundingRate"])
    except Exception:
        return None

async def okx_oi(inst):
    """Open Interest from OKX."""
    try:
        d = await okx_get("/api/v5/public/open-interest", {"instId": inst})
        return float(d[0]["oi"])
    except Exception:
        return None

# ── CoinGecko fetcher (market intel) ─────────────────────────────────────────

async def gecko_market(gecko_id):
    """Market cap, 24h change, community sentiment."""
    try:
        d = await get(GECKO + "/coins/markets", {
            "vs_currency": "usd",
            "ids": gecko_id,
            "order": "market_cap_desc",
            "per_page": "1",
            "page": "1",
            "price_change_percentage": "1h,24h,7d",
        })
        return d[0] if d else {}
    except Exception:
        return {}

# ── Indicator library ─────────────────────────────────────────────────────────

def closes_from_klines(klines):
    """Extract close prices from klines array."""
    return [float(k[4]) for k in klines]

def highs_from_klines(klines):
    return [float(k[2]) for k in klines]

def lows_from_klines(klines):
    return [float(k[3]) for k in klines]

def volumes_from_klines(klines):
    return [float(k[5]) for k in klines]

def ema_series(values, span):
    if not values:
        return []
    k = 2.0 / (span + 1)
    out = [float(values[0])]
    for v in values[1:]:
        out.append(float(v) * k + out[-1] * (1.0 - k))
    return out

def ema_val(values, span):
    s = ema_series(values, span)
    return s[-1] if s else 0.0

def calc_rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50.0
    gains  = [max(closes[i] - closes[i-1], 0.0) for i in range(1, len(closes))]
    losses = [max(closes[i-1] - closes[i], 0.0) for i in range(1, len(closes))]
    ag = statistics.mean(gains[-period:])
    al = statistics.mean(losses[-period:])
    if al == 0:
        return 100.0
    return round(100.0 - 100.0 / (1.0 + ag / al), 2)

def calc_macd(closes):
    if len(closes) < 35:
        return {"macd": 0.0, "signal": 0.0, "hist": 0.0, "bull": False}
    e12  = ema_series(closes, 12)
    e26  = ema_series(closes, 26)
    ml   = [e12[i] - e26[i] for i in range(len(e26))]
    sl   = ema_series(ml, 9)
    hist = ml[-1] - sl[-1]
    return {
        "macd":   round(ml[-1], 8),
        "signal": round(sl[-1], 8),
        "hist":   round(hist, 8),
        "bull":   ml[-1] > sl[-1],
    }

def calc_bollinger(closes, period=20, std_mult=2.0):
    if len(closes) < period:
        p = closes[-1] if closes else 0.0
        return {"upper": p, "mid": p, "lower": p, "pct_b": 0.5}
    window = closes[-period:]
    mid    = statistics.mean(window)
    sd     = statistics.stdev(window) if len(window) > 1 else 0.0
    upper  = mid + std_mult * sd
    lower  = mid - std_mult * sd
    price  = closes[-1]
    pct_b  = (price - lower) / (upper - lower) if upper != lower else 0.5
    return {
        "upper": round(upper, 8),
        "mid":   round(mid, 8),
        "lower": round(lower, 8),
        "pct_b": round(max(0.0, min(1.0, pct_b)), 4),
    }

def calc_volatility(closes, period=14):
    if len(closes) < 2:
        return 0.0
    returns = [abs(closes[i] / closes[i-1] - 1.0) for i in range(1, len(closes))
               if closes[i-1] != 0]
    if not returns:
        return 0.0
    return round(statistics.mean(returns[-period:]) * 100.0, 4)

def calc_vol_spike(volumes):
    if len(volumes) < 3:
        return 1.0
    hist = volumes[:-1][-20:]
    avg  = statistics.mean(hist) if hist else 1.0
    cur  = volumes[-1]
    return round(cur / avg, 3) if avg > 0 else 1.0

def calc_support_resistance(closes, highs, lows, lookback=20):
    if not closes:
        return {"support": 0.0, "resistance": 0.0}
    price = closes[-1]
    if len(closes) < lookback:
        return {"support": round(price * 0.97, 8), "resistance": round(price * 1.03, 8)}
    rh = highs[-lookback:]
    rl = lows[-lookback:]
    return {
        "support":    round(min(rl), 8),
        "resistance": round(max(rh), 8),
    }

def calc_ob_imbalance(bids, asks, price):
    lo, hi = price * 0.98, price * 1.02
    bv = sum(float(b[1]) for b in bids if float(b[0]) >= lo)
    av = sum(float(a[1]) for a in asks if float(a[0]) <= hi)
    total = bv + av
    if total == 0:
        return 0.0, False
    ratio = bv / (av + 1e-9)
    wall  = max(bv, av) > 3.0 * (total / 2.0)
    return round(ratio, 3), wall

# ── AGENT A — Quantitative Analyst ───────────────────────────────────────────

def agent_quant(ticker_24h, klines_1h, funding_rate, oi_now, oi_prev,
                gecko_mkt, direction):
    score    = 50
    warnings = []
    details  = []

    # --- Volume spike (from MEXC 1h candles)
    vols      = volumes_from_klines(klines_1h)
    vol_spike = calc_vol_spike(vols)
    price_chg = float(ticker_24h.get("priceChangePercent", 0))

    if vol_spike > 2.0:
        if abs(price_chg) >= 2.0:
            score += 20
            details.append("Volume {:.1f}x average confirms price move.".format(vol_spike))
        else:
            score -= 20
            warnings.append(
                "Volume {:.1f}x average but price barely moved — "
                "exhaustion or churn signal.".format(vol_spike)
            )
    elif vol_spike < 0.5:
        score -= 25
        warnings.append(
            "Volume only {:.1f}x average. "
            "Move lacks conviction — possible fakeout.".format(vol_spike)
        )
    else:
        score += 5
        details.append("Volume {:.1f}x average — normal participation.".format(vol_spike))

    # --- Volatility
    closes     = closes_from_klines(klines_1h)
    volatility = calc_volatility(closes)
    if volatility > 5.0:
        score -= 15
        warnings.append("Extreme volatility {:.2f}%. High reversal risk.".format(volatility))
    elif volatility > 2.5:
        score -= 5
        details.append("Elevated volatility {:.2f}%. Widen stops.".format(volatility))
    elif volatility < 0.2:
        score -= 10
        warnings.append("Near-zero volatility. Breakout or manipulation risk.")
    else:
        score += 10
        details.append("Healthy volatility {:.2f}%.".format(volatility))

    # --- OKX Funding rate (secondary confirmation)
    if funding_rate is not None:
        fr_pct = funding_rate * 100.0
        if fr_pct > 0.08:
            score -= 15
            warnings.append(
                "Funding rate very high ({:.4f}%). "
                "Longs overheated — short squeeze risk.".format(fr_pct)
            )
        elif fr_pct < -0.05:
            score -= 10
            warnings.append(
                "Funding rate deeply negative ({:.4f}%). "
                "Shorts overloaded — long squeeze risk.".format(fr_pct)
            )
        elif abs(fr_pct) < 0.02:
            score += 10
            details.append("Neutral funding rate ({:.4f}%). No crowding.".format(fr_pct))
        else:
            score += 5
    else:
        # Fallback: volume/price divergence as proxy
        if vol_spike > 2.0 and abs(price_chg) < 0.5:
            score -= 15
            warnings.append(
                "Derivatives unavailable. "
                "High volume + flat price = overheating proxy signal."
            )

    # --- OKX Open Interest delta
    if oi_now is not None and oi_prev is not None and oi_prev > 0:
        oi_chg_pct = ((oi_now - oi_prev) / oi_prev) * 100.0
        if direction == "LONG":
            if oi_chg_pct > 2.0 and price_chg > 0:
                score += 10
                details.append("OI growing +{:.2f}% with price. Institutional longs building.".format(oi_chg_pct))
            elif oi_chg_pct < -2.0 and price_chg > 0:
                score -= 15
                warnings.append("OI declining while price rises. Short squeeze exhaustion likely.")
        else:
            if oi_chg_pct > 2.0 and price_chg < 0:
                score += 10
                details.append("OI growing +{:.2f}% with falling price. Short pressure building.".format(oi_chg_pct))

    # --- CoinGecko market cap risk
    mkt_cap = gecko_mkt.get("market_cap", 0) or 0
    if mkt_cap > 0:
        if mkt_cap < 10_000_000:
            score -= 20
            warnings.append("Micro-cap < $10M. Extreme manipulation risk.")
        elif mkt_cap < 100_000_000:
            score -= 10
            warnings.append("Small-cap < $100M. Elevated volatility and manipulation risk.")
        elif mkt_cap > 1_000_000_000:
            score += 8
            details.append("Large-cap > $1B. More stable and liquid.")

    # --- 24h direction alignment
    if direction == "LONG":
        if price_chg > 5.0:
            score += 8
        elif price_chg < -5.0:
            score -= 12
    else:
        if price_chg < -5.0:
            score += 8
        elif price_chg > 5.0:
            score -= 12

    score = max(0, min(100, score))

    if score >= 65:
        verdict, label = "valid", "VOLUME SUPPORTS MOVE"
    elif score >= 40:
        verdict, label = "risky", "RISKY MARKET CONDITIONS"
    else:
        verdict, label = "dead", "NO MARKET FUEL"

    return {
        "score":        score,
        "verdict":      verdict,
        "label":        label,
        "detail":       " | ".join(details) if details else "Analysis complete.",
        "warnings":     warnings,
        "vol_spike":    vol_spike,
        "volatility":   volatility,
        "funding_rate": funding_rate,
        "oi_delta_pct": round(((oi_now - oi_prev) / (oi_prev + 1e-9)) * 100, 3)
                        if oi_now is not None and oi_prev is not None else None,
    }

# ── AGENT B — Technical Analyst ───────────────────────────────────────────────

def agent_ta(klines_short, klines_long, current_price, direction):
    score    = 50
    warnings = []
    details  = []

    closes_s = closes_from_klines(klines_short)
    highs_s  = highs_from_klines(klines_short)
    lows_s   = lows_from_klines(klines_short)
    closes_l = closes_from_klines(klines_long)

    if not closes_s:
        return {
            "score": 50, "verdict": "risky",
            "label": "INSUFFICIENT CHART DATA",
            "detail": "Not enough OHLC data.", "warnings": ["Limited candle history."],
            "rsi_short": 50, "rsi_long": 50, "macd_bull": False,
            "bb_pct_b": 0.5, "ema9": 0, "ema21": 0, "ema50": 0,
            "support": 0, "resistance": 0,
        }

    # --- EMA ribbon (9/21/50) from short candles
    e9  = ema_val(closes_s, 9)
    e21 = ema_val(closes_s, 21)
    e50 = ema_val(closes_s, min(50, len(closes_s) - 1)) if len(closes_s) > 10 else e21
    p   = current_price

    if direction == "LONG":
        ema_aligned = p > e9 > e21
        ema_stacked = e9 > e21 > e50
    else:
        ema_aligned = p < e9 < e21
        ema_stacked = e9 < e21 < e50

    if ema_aligned and ema_stacked:
        score += 20
        details.append("EMA ribbon 9/21/50 fully stacked with {} direction.".format(direction))
    elif ema_aligned:
        score += 10
        details.append("EMA 9/21 aligned. EMA50 not yet confirmed.")
    else:
        score -= 15
        warnings.append("EMA ribbon misaligned. Price fighting moving averages.")

    # --- RSI (short-term)
    rsi_s = calc_rsi(closes_s, 14)
    if direction == "LONG":
        if rsi_s > 72:
            score -= 20
            warnings.append("RSI overbought at {:.1f}. High reversal risk.".format(rsi_s))
        elif rsi_s > 55:
            score += 15
            details.append("RSI bullish at {:.1f}.".format(rsi_s))
        elif rsi_s > 45:
            score += 5
            details.append("RSI neutral at {:.1f}. Momentum building.".format(rsi_s))
        elif rsi_s < 28:
            score += 8
            details.append("RSI oversold at {:.1f}. Potential bounce zone.".format(rsi_s))
        else:
            score -= 10
            warnings.append("RSI weak at {:.1f}. Momentum not in your favor.".format(rsi_s))
    else:
        if rsi_s < 28:
            score -= 20
            warnings.append("RSI oversold at {:.1f}. Bounce likely against your short.".format(rsi_s))
        elif rsi_s < 45:
            score += 15
            details.append("RSI bearish at {:.1f}. Supports short.".format(rsi_s))
        elif rsi_s > 72:
            score += 10
            details.append("RSI overbought at {:.1f}. Short entry zone.".format(rsi_s))
        else:
            score -= 8
            warnings.append("RSI neutral at {:.1f}. No strong bearish signal yet.".format(rsi_s))

    # --- MACD
    macd = calc_macd(closes_s)
    if direction == "LONG":
        if macd["bull"]:
            score += 15
            details.append("MACD bullish cross confirmed.")
        else:
            score -= 10
            warnings.append("MACD bearish. Momentum not supporting long.")
    else:
        if not macd["bull"]:
            score += 15
            details.append("MACD bearish cross confirmed.")
        else:
            score -= 10
            warnings.append("MACD bullish. Momentum fighting your short.")

    # --- Bollinger Bands
    bb = calc_bollinger(closes_s)
    pb = bb["pct_b"]
    if direction == "LONG":
        if pb > 0.9:
            score -= 15
            warnings.append("Price at upper Bollinger ({:.0f}%). Overbought zone.".format(pb * 100))
        elif pb < 0.2:
            score += 10
            details.append("Price at lower Bollinger. Potential bounce zone.")
        elif 0.4 < pb < 0.7:
            score += 8
            details.append("Price mid-Bollinger. Healthy momentum zone.")
    else:
        if pb < 0.1:
            score -= 15
            warnings.append("Price at lower Bollinger ({:.0f}%). Oversold — short squeeze risk.".format(pb * 100))
        elif pb > 0.8:
            score += 10
            details.append("Price at upper Bollinger. Short entry zone.")

    # --- MTF conflict check (short vs long RSI)
    rsi_l = calc_rsi(closes_l, 14) if closes_l else 50.0
    conflict = (direction == "LONG" and rsi_l < 45) or \
               (direction == "SHORT" and rsi_l > 55)
    if conflict:
        score -= 20
        warnings.append(
            "MTF CONFLICT: Short-term signal contradicts longer trend. "
            "Counter-trend trap risk. Rule 1 violated."
        )
    else:
        score += 10
        details.append("Timeframes agree on directional bias.")

    # --- Support / Resistance proximity
    sr = calc_support_resistance(closes_s, highs_s, lows_s)
    dist_res = ((sr["resistance"] - p) / p) * 100 if p > 0 else 5.0
    dist_sup = ((p - sr["support"]) / p) * 100 if p > 0 else 5.0

    if direction == "LONG":
        if dist_res < 1.5:
            score -= 20
            warnings.append(
                "Price only {:.1f}% from resistance ${:.6g}. "
                "Very limited upside before wall.".format(dist_res, sr["resistance"])
            )
        elif dist_res > 5.0:
            score += 10
            details.append(
                "Clear runway to resistance ${:.6g} ({:.1f}% away).".format(
                    sr["resistance"], dist_res)
            )
    else:
        if dist_sup < 1.5:
            score -= 20
            warnings.append(
                "Price only {:.1f}% from support ${:.6g}. "
                "Short may bounce here.".format(dist_sup, sr["support"])
            )
        elif dist_sup > 5.0:
            score += 10
            details.append(
                "Clear path to support ${:.6g} ({:.1f}% away).".format(
                    sr["support"], dist_sup)
            )

    score = max(0, min(100, score))

    if score >= 65:
        verdict, label = "valid", "CHART STRUCTURE CONFIRMED"
    elif score >= 40:
        verdict, label = "risky", "MIXED TECHNICAL SIGNALS"
    else:
        verdict, label = "dead", "CHART STRUCTURE BROKEN"

    return {
        "score":     score,
        "verdict":   verdict,
        "label":     label,
        "detail":    " | ".join(details) if details else "Analysis complete.",
        "warnings":  warnings,
        "rsi_short": rsi_s,
        "rsi_long":  rsi_l,
        "macd_bull": macd["bull"],
        "bb_pct_b":  pb,
        "ema9":      round(e9, 8),
        "ema21":     round(e21, 8),
        "ema50":     round(e50, 8),
        "support":   sr["support"],
        "resistance":sr["resistance"],
    }

# ── SUPERVISOR — Conflict Resolution Matrix ───────────────────────────────────

# Each entry: (quant_verdict, ta_verdict) -> (confidence, color, action, base_msg)
MATRIX = {
    ("valid",  "valid"):  ("HIGH_CONFIDENCE", "green",  "EXECUTE",      "Strong volume and clear chart structure. All systems go."),
    ("valid",  "risky"):  ("CAUTION",          "yellow", "WAIT",         "Market is healthy, but chart timing is off. Wait for confirmation."),
    ("valid",  "dead"):   ("RESISTANCE_WALL",  "red",    "DO_NOT_TRADE", "Strong volume, but price is hitting a structural wall."),
    ("risky",  "valid"):  ("CAUTION",          "yellow", "REDUCE_SIZE",  "Chart looks good, but market conditions are risky."),
    ("risky",  "risky"):  ("CAUTION",          "yellow", "WAIT",         "Mixed signals across both agents. No clean edge."),
    ("risky",  "dead"):   ("TRAP_ALERT",       "red",    "DO_NOT_TRADE", "Weak market and broken chart. Do not force a trade."),
    ("dead",   "valid"):  ("TRAP_ALERT",       "red",    "DO_NOT_TRADE", "Chart looks good but no volume backing. Likely a fakeout."),
    ("dead",   "risky"):  ("NO_TRADE",         "red",    "STAY_OUT",     "No volume and weak chart. Dangerous environment."),
    ("dead",   "dead"):   ("NO_TRADE",         "red",    "STAY_OUT",     "Both agents flag danger. Preserve capital."),
}

def plain_english(conf, color, quant, ta, direction):
    d   = "up" if direction == "LONG" else "down"
    act = "buying" if direction == "LONG" else "selling"
    if conf == "HIGH_CONFIDENCE":
        return (
            "The price looks ready to go {}. Big traders are {} and the chart "
            "confirms the move. It looks safe to enter.".format(d, act)
        )
    if conf == "CAUTION" and quant["verdict"] == "valid":
        return (
            "The market has strong volume and momentum, but the chart is not yet "
            "showing a clean signal. The timing is off right now — wait for confirmation."
        )
    if conf == "CAUTION" and ta["verdict"] == "valid":
        return (
            "The chart pattern looks promising, but broader market conditions are risky. "
            "Trade smaller than usual or wait for a safer setup."
        )
    if conf == "CAUTION":
        return (
            "Mixed signals from both analysts. There is no clear edge in the market "
            "right now. Patience is the correct move."
        )
    if conf == "TRAP_ALERT":
        return (
            "This is likely a trap. The price looks like it is moving {}, "
            "but there is not enough volume or structure to support it. "
            "Do not enter this move.".format(d)
        )
    if conf == "RESISTANCE_WALL":
        return (
            "There is strong volume behind this move, but the price is hitting "
            "a major structural wall. It is likely to get rejected here. "
            "Wait for a confirmed breakout."
        )
    return (
        "Both analysts agree the conditions are dangerous right now. "
        "The safest action is to stay out and protect your capital."
    )

def safe_passage(conf, color, ta, price, direction):
    sup = ta.get("support",    price * 0.97)
    res = ta.get("resistance", price * 1.03)
    if color == "green":
        return (
            "You may enter with your planned size. "
            "Place stop loss below ${:.6g} (support level).".format(sup)
        )
    if conf == "CAUTION":
        if direction == "LONG":
            return (
                "Wait for a pullback to ${:.6g} (support) before entering, "
                "or wait for two consecutive 15-minute bullish candle closes "
                "above current price.".format(round(sup, 6))
            )
        else:
            return (
                "Wait for price to rally to ${:.6g} (resistance) before shorting, "
                "or reduce your position size by half to limit risk.".format(round(res, 6))
            )
    if direction == "LONG":
        return (
            "Do not enter now. Wait for the price to drop to ${:.6g} "
            "then watch for two consecutive bullish candle closes on the "
            "15-minute chart before trying again.".format(round(sup, 6))
        )
    return (
        "Do not enter now. Wait for price to rally to ${:.6g} "
        "then watch for two consecutive bearish candle closes on the "
        "15-minute chart before shorting.".format(round(res, 6))
    )

def risk_warning(quant, ta, conf):
    all_w = quant.get("warnings", []) + ta.get("warnings", [])
    if not all_w:
        if conf == "HIGH_CONFIDENCE":
            return (
                "No major risks detected. "
                "Always monitor for sudden macro events or news."
            )
        return "Standard market risks apply. Always use a stop loss."
    priority = [
        "manipulation", "exhaustion", "overbought", "oversold",
        "extreme", "fakeout", "overheated", "conflict", "divergen",
        "resistance", "volume", "squeeze",
    ]
    for kw in priority:
        for w in all_w:
            if kw.lower() in w.lower():
                return "Risk: " + w
    return "Risk: " + all_w[0]

def supervisor(quant, ta, direction, price):
    qv  = quant["verdict"]
    tv  = ta["verdict"]
    key = (qv, tv)
    conf, color, action, _ = MATRIX.get(
        key, ("CAUTION", "yellow", "WAIT", "Proceed with caution.")
    )
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
        "plain_english": plain_english(conf, color, quant, ta, direction),
        "safe_passage":  safe_passage(conf, color, ta, price, direction),
        "risk_warning":  risk_warning(quant, ta, conf),
        "all_warnings":  quant.get("warnings", []) + ta.get("warnings", []),
    }

# ── Master update ─────────────────────────────────────────────────────────────

async def run_analysis(symbol):
    s = positions.get(symbol)
    if s is None:
        return

    mexc_sym = s["mexc_sym"]
    okx_inst = s["okx_inst"]
    gecko_id = s["gecko_id"]
    direction = s["direction"]

    try:
        # ── Primary: MEXC ──────────────────────────────────────────────────
        ticker, klines_1h, klines_15m, klines_4h, depth = await asyncio.gather(
            mexc_24h(mexc_sym),
            mexc_klines(mexc_sym, "1h",  60),
            mexc_klines(mexc_sym, "15m", 60),
            mexc_klines(mexc_sym, "4h",  60),
            mexc_depth(mexc_sym, 50),
        )

        current_price = float(ticker.get("lastPrice", s["entry_price"]))
        s["current_price"] = current_price

        # PnL
        ep, sz = s["entry_price"], s["position_size"]
        if direction == "LONG":
            s["pnl_usd"] = round((current_price - ep) * sz, 4)
            s["pnl_pct"] = round(((current_price - ep) / ep) * 100, 4)
        else:
            s["pnl_usd"] = round((ep - current_price) * sz, 4)
            s["pnl_pct"] = round(((ep - current_price) / ep) * 100, 4)

        # Order book imbalance
        ob_ratio, liq_wall = calc_ob_imbalance(
            depth.get("bids", []), depth.get("asks", []), current_price
        )
        s["ob_ratio"]  = ob_ratio
        s["liq_wall"]  = liq_wall

        # ── Secondary: OKX (confirmation) ─────────────────────────────────
        okx_fund, oi_now_raw, okx_c1h = await asyncio.gather(
            okx_funding(okx_inst),
            okx_oi(okx_inst),
            okx_candles(okx_inst, "1H", 60),
        )

        oi_prev = s.get("_oi_prev")
        oi_now  = oi_now_raw
        s["_oi_prev"] = oi_now

        # ── Market intel: CoinGecko ────────────────────────────────────────
        gecko_mkt = await gecko_market(gecko_id)

        # ── Run agents ────────────────────────────────────────────────────
        q_result = agent_quant(
            ticker, klines_1h, okx_fund,
            oi_now, oi_prev, gecko_mkt, direction
        )

        # TA uses MEXC 15m as primary, MEXC 4h as long-term context
        # OKX 1h candles used as second confirmation if available
        ta_short = klines_15m if klines_15m else okx_c1h
        ta_long  = klines_4h

        ta_result = agent_ta(ta_short, ta_long, current_price, direction)

        # ── Supervisor merge ───────────────────────────────────────────────
        verdict = supervisor(q_result, ta_result, direction, current_price)

        # ── Store ──────────────────────────────────────────────────────────
        s["market_data"] = {
            "price":       current_price,
            "volume_24h":  float(ticker.get("volume", 0)) * current_price,
            "quote_vol":   float(ticker.get("quoteVolume", 0)),
            "change_24h":  float(ticker.get("priceChangePercent", 0)),
            "market_cap":  gecko_mkt.get("market_cap", 0) or 0,
            "ob_ratio":    ob_ratio,
            "liq_wall":    liq_wall,
        }
        s["quant"]          = q_result
        s["ta"]             = ta_result
        s["verdict"]        = verdict
        s["signal_status"]  = "LIVE"
        s["last_update"]    = time.time()

    except Exception as e:
        if "verdict" not in s:
            s["verdict"] = {}
        s["verdict"]["plain_english"] = (
            "Data error: {}. Retrying next cycle.".format(str(e))
        )
        s["signal_status"] = "ERROR"

async def monitor_loop(symbol):
    while symbol in positions:
        await run_analysis(symbol)
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
    symbol        = str(body.get("symbol", "")).strip().upper()
    direction     = str(body.get("direction", "LONG")).strip().upper()
    entry_price   = float(body.get("entry_price", 0))
    position_size = float(body.get("position_size", 0))

    if not symbol:
        raise HTTPException(400, "symbol is required")
    if entry_price <= 0:
        raise HTTPException(400, "entry_price must be greater than 0")
    if position_size <= 0:
        raise HTTPException(400, "position_size must be greater than 0")
    if direction not in ("LONG", "SHORT"):
        raise HTTPException(400, "direction must be LONG or SHORT")
    if len(positions) >= MAX_POSITIONS and symbol not in positions:
        raise HTTPException(400, "Maximum {} positions reached. Close one to add another.".format(MAX_POSITIONS))

    mexc_sym = to_mexc(symbol)
    okx_inst = to_okx(symbol)
    gecko_id = to_gecko(symbol)

    # Validate symbol on MEXC
    try:
        ticker = await mexc_24h(mexc_sym)
        price  = float(ticker.get("lastPrice", entry_price))
    except Exception as e:
        raise HTTPException(
            400,
            "Cannot find '{}' on MEXC. "
            "Try formats like BTCUSDT, ETHUSDT, SOLUSDT. "
            "Error: {}".format(mexc_sym, str(e))
        )

    default_verdict = {
        "confidence":    "CAUTION",
        "color":         "yellow",
        "action":        "WAIT",
        "plain_english": "Agents are analysing the market. Please wait ~60 seconds for first results.",
        "safe_passage":  "Analysis in progress...",
        "risk_warning":  "Analysis in progress...",
        "quant_score":   50,
        "ta_score":      50,
        "quant_verdict": "risky",
        "ta_verdict":    "risky",
        "quant_label":   "INITIALISING",
        "ta_label":      "INITIALISING",
        "all_warnings":  [],
    }
    default_agent = {
        "score": 50, "verdict": "risky", "label": "INITIALISING",
        "detail": "", "warnings": [],
        "rsi_short": 50, "rsi_long": 50, "macd_bull": False,
        "bb_pct_b": 0.5, "ema9": 0, "ema21": 0, "ema50": 0,
        "support": 0, "resistance": 0,
        "vol_spike": 1.0, "volatility": 0.0, "funding_rate": None,
    }

    positions[symbol] = {
        "symbol":        symbol,
        "mexc_sym":      mexc_sym,
        "okx_inst":      okx_inst,
        "gecko_id":      gecko_id,
        "direction":     direction,
        "entry_price":   entry_price,
        "position_size": position_size,
        "started_at":    time.time(),
        "last_update":   time.time(),
        "current_price": price,
        "pnl_usd":       0.0,
        "pnl_pct":       0.0,
        "ob_ratio":      0.0,
        "liq_wall":      False,
        "market_data":   {},
        "quant":         default_agent.copy(),
        "ta":            default_agent.copy(),
        "verdict":       default_verdict,
        "signal_status": "PENDING",
        "_oi_prev":      None,
    }

    asyncio.create_task(monitor_loop(symbol))
    return {
        "status":          "initialized",
        "symbol":          symbol,
        "mexc_symbol":     mexc_sym,
        "okx_instrument":  okx_inst,
        "gecko_id":        gecko_id,
    }

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
                "confidence":    s["verdict"].get("confidence", "—"),
                "color":         s["verdict"].get("color", "grey"),
                "action":        s["verdict"].get("action", "—"),
                "signal_status": s["signal_status"],
                "last_update":   s["last_update"],
            }
            for sym, s in positions.items()
        },
    }

@app.get("/state/{symbol}")
async def get_state(symbol: str):
    sym = symbol.strip().upper()
    if sym not in positions:
        raise HTTPException(404, "No active position for {}".format(sym))
    s = positions[sym]
    # Return everything except internal cache key
    return {k: v for k, v in s.items() if not k.startswith("_")}

@app.delete("/close/{symbol}")
async def close_pos(symbol: str):
    sym = symbol.strip().upper()
    if sym in positions:
        del positions[sym]
        return {"status": "closed", "symbol": sym}
    raise HTTPException(404, "No active position for {}".format(sym))

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "active": len(positions),
        "max":    MAX_POSITIONS,
        "sources": ["MEXC (primary)", "OKX (secondary)", "CoinGecko (market intel)"],
    }
