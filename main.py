"""
Trade Guardian Engine
Data: OKX (price, OB, OI, funding) + CoinGecko (market cap, trend backup)
No pydantic models - plain dicts only to avoid all build issues on Render
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
import httpx
import asyncio
import time
import statistics
import os
import json

app = FastAPI(title="Trade Guardian Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

positions = {}

OKX   = "https://www.okx.com"
GECKO = "https://api.coingecko.com/api/v3"

# ── OKX helpers ───────────────────────────────────────────────────────────────

async def okx(path, params=None):
    url = OKX + path
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(url, params=params or {})
        r.raise_for_status()
        d = r.json()
        if d.get("code") != "0":
            raise Exception("OKX error: " + str(d.get("msg", "")))
        return d["data"]

async def gecko(path, params=None):
    url = GECKO + path
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(url, params=params or {},
                        headers={"Accept": "application/json"})
        r.raise_for_status()
        return r.json()

def to_okx_symbol(symbol: str) -> str:
    """BTCUSDT -> BTC-USDT-SWAP"""
    s = symbol.upper().replace("-SWAP", "").replace("-USDT", "")
    s = s.replace("USDT", "").replace("BUSD", "").replace("USD", "")
    return s + "-USDT-SWAP"

def to_gecko_id(symbol: str) -> str:
    mapping = {
        "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana",
        "BNB": "binancecoin", "XRP": "ripple", "ADA": "cardano",
        "DOGE": "dogecoin", "AVAX": "avalanche-2", "DOT": "polkadot",
        "MATIC": "matic-network", "LINK": "chainlink", "UNI": "uniswap",
        "LTC": "litecoin", "ATOM": "cosmos", "NEAR": "near",
        "ARB": "arbitrum", "OP": "optimism", "SUI": "sui",
        "APT": "aptos", "INJ": "injective-protocol",
    }
    base = symbol.upper().replace("USDT","").replace("-USDT-SWAP","").replace("-","")
    return mapping.get(base, base.lower())

async def get_price(okx_sym: str) -> float:
    data = await okx("/api/v5/market/ticker", {"instId": okx_sym})
    return float(data[0]["last"])

async def get_order_book(okx_sym: str):
    data = await okx("/api/v5/market/books", {"instId": okx_sym, "sz": "50"})
    return data[0]  # {bids, asks}

async def get_open_interest(okx_sym: str) -> float:
    data = await okx("/api/v5/public/open-interest", {"instId": okx_sym})
    return float(data[0]["oi"])

async def get_funding_rate(okx_sym: str) -> float:
    data = await okx("/api/v5/public/funding-rate", {"instId": okx_sym})
    return float(data[0]["fundingRate"])

async def get_candles(okx_sym: str, bar: str, limit: int = 50):
    """Returns list of [ts, open, high, low, close, vol, ...]"""
    data = await okx("/api/v5/market/candles",
                     {"instId": okx_sym, "bar": bar, "limit": limit})
    return list(reversed(data))  # oldest first

async def get_gecko_data(gecko_id: str):
    try:
        data = await gecko("/coins/" + gecko_id,
                           {"localization": "false", "tickers": "false",
                            "community_data": "false", "developer_data": "false"})
        mkt = data.get("market_data", {})
        return {
            "price_change_24h_pct": mkt.get("price_change_percentage_24h", 0),
            "market_cap": mkt.get("market_cap", {}).get("usd", 0),
            "total_volume": mkt.get("total_volume", {}).get("usd", 0),
        }
    except Exception:
        return {"price_change_24h_pct": 0, "market_cap": 0, "total_volume": 0}

# ── Analytics ─────────────────────────────────────────────────────────────────

def compute_body_expansion(candles):
    if len(candles) < 22:
        return 1.0, 1.0
    completed = candles[:-1]
    trigger   = completed[-1]
    bodies = [abs(float(k[4]) - float(k[1])) for k in completed[:-1]]
    vols   = [float(k[5]) for k in completed[:-1]]
    avg_body = statistics.mean(bodies[-20:]) if len(bodies) >= 20 else statistics.mean(bodies) if bodies else 1
    avg_vol  = statistics.mean(vols[-20:])   if len(vols)   >= 20 else statistics.mean(vols)   if vols   else 1
    bef = abs(float(trigger[4]) - float(trigger[1])) / avg_body if avg_body else 1.0
    vef = float(trigger[5]) / avg_vol if avg_vol else 1.0
    return round(bef, 3), round(vef, 3)

def classify_ob_imbalance(bids, asks, price):
    lim_up = price * 1.02
    lim_dn = price * 0.98
    bid_vol = sum(float(b[1]) for b in bids if float(b[0]) >= lim_dn)
    ask_vol = sum(float(a[1]) for a in asks if float(a[0]) <= lim_up)
    total = bid_vol + ask_vol
    if total == 0:
        return 0.0, False
    ratio = bid_vol / (ask_vol + 1e-9)
    wall  = max(bid_vol, ask_vol) > 3 * (total / 2)
    return round(ratio, 3), wall

def classify_oi_signal(price_chg, oi_chg):
    p, o = price_chg > 0, oi_chg > 0
    if p and o:   return "STRONG_INSTITUTIONAL"
    if p and not o: return "EXHAUSTION_SQUEEZE_END"
    if not p and o: return "AGGRESSIVE_SHORT_INFLOW"
    return "FADING_MOMENTUM"

def classify_mtf(c5m, c4h):
    def ema(data, span):
        k = 2.0 / (span + 1); e = data[0]
        for v in data[1:]: e = v * k + e * (1 - k)
        return e
    closes_4h = [float(k[4]) for k in c4h[:-1]]
    trend_4h  = "UNKNOWN"
    if len(closes_4h) >= 21:
        trend_4h = "BULLISH" if ema(closes_4h[-9:], 9) > ema(closes_4h[-21:], 21) else "BEARISH"
    closes_5m  = [float(k[4]) for k in c5m[:-1]]
    momentum_5m = "UNKNOWN"
    if len(closes_5m) >= 5:
        momentum_5m = "BULLISH" if closes_5m[-1] > closes_5m[-5] else "BEARISH"
    return momentum_5m, trend_4h

def compute_health(s):
    score = 50.0
    oi = s["oi_shift"]
    d  = s["direction"]
    if d == "LONG":
        if oi == "STRONG_INSTITUTIONAL":      score += 20
        elif oi == "EXHAUSTION_SQUEEZE_END":  score -= 25
        elif oi == "AGGRESSIVE_SHORT_INFLOW": score -= 20
    else:
        if oi == "AGGRESSIVE_SHORT_INFLOW":   score += 20
        elif oi == "EXHAUSTION_SQUEEZE_END":  score -= 20
        elif oi == "STRONG_INSTITUTIONAL":    score -= 15
    if d == "LONG":
        if s["trend_4h"] == "BULLISH" and s["momentum_5m"] == "BULLISH": score += 15
        elif s["trend_4h"] == "BEARISH": score -= 20
    else:
        if s["trend_4h"] == "BEARISH" and s["momentum_5m"] == "BEARISH": score += 15
        elif s["trend_4h"] == "BULLISH": score -= 20
    ob = s["ob_imbalance"]
    if d == "LONG":
        score += 10 if ob > 1.5 else (-15 if ob < 0.6 else 0)
    else:
        score += 10 if ob < 0.6 else (-15 if ob > 1.5 else 0)
    if s["liq_wall_flagged"]: score -= 15
    if s["cascade_signal"]:   score += 10
    if s["volatility_buffer"]: score -= 5
    return max(0.0, min(100.0, round(score, 1)))

def generate_verdict(s):
    h  = s["health_score"]
    oi = s["oi_shift"]
    d  = s["direction"]

    if h >= 75:   verdict = "STRONG HOLD - INSTITUTIONAL ACCUMULATION"
    elif h >= 55: verdict = "WATCH CLOSELY - LIQUIDITY WALL DETECTED" if s["liq_wall_flagged"] else "WATCH CLOSELY - MINUTE CONSOLIDATION"
    elif h >= 35: verdict = "TAKE PROFIT - MOMENTUM EXHAUSTED" if oi in ("EXHAUSTION_SQUEEZE_END","FADING_MOMENTUM") else "WATCH CLOSELY - STRUCTURE WEAKENING"
    else:         verdict = "ABORT POSITION - INSIDER OUTFLOW DETECTED"

    lines = []
    if d == "LONG":
        if oi == "STRONG_INSTITUTIONAL":      lines.append("Whales stacking longs. OI expanding with price. Institutional capacity intact.")
        elif oi == "EXHAUSTION_SQUEEZE_END":  lines.append("OI unwinding as price rises. Short squeeze exhaustion. New buyers thinning.")
        elif oi == "AGGRESSIVE_SHORT_INFLOW": lines.append("Fresh shorts entering. Downward pressure likely. Re-evaluate thesis.")
    else:
        if oi == "AGGRESSIVE_SHORT_INFLOW":   lines.append("Short sellers increasing conviction. OI confirming downside pressure.")
        elif oi == "EXHAUSTION_SQUEEZE_END":  lines.append("Short squeeze may be ending. OI declining means covering, not new shorts.")

    if s["trend_4h"] == "BEARISH" and d == "LONG":
        lines.append("HIGH RISK: 4H structure bearish. Counter-trend long into markdown.")
    elif s["trend_4h"] == "BULLISH" and d == "SHORT":
        lines.append("HIGH RISK: 4H trend bullish. Shorting into institutional buying.")

    if s["liq_wall_flagged"]:
        lines.append("Liquidity wall detected within 2% of price. Expect absorption or rejection.")

    pnl = s["pnl_pct"]
    if pnl > 2.0:   lines.append("Position +{:.2f}% profit. Consider partial close. Trail stop to breakeven.".format(pnl))
    elif pnl < -1.5: lines.append("Position {:.2f}% drawdown. Confirm thesis before averaging.".format(pnl))

    if s["volatility_buffer"]:
        lines.append("Funding rate elevated ({:.4f}%). Stop expanded 20%.".format(s["funding_rate"] * 100))

    if not lines:
        lines.append("Market structure neutral. Monitor for breakout on next 15m candle.")

    return verdict, " | ".join(lines)

# ── Background update ─────────────────────────────────────────────────────────

async def update_position(symbol):
    s = positions[symbol]
    okx_sym   = s["okx_sym"]
    gecko_id  = s["gecko_id"]
    try:
        price = await get_price(okx_sym)
        s["current_price"] = price

        ep = s["entry_price"]
        sz = s["position_size"]
        if s["direction"] == "LONG":
            s["pnl_usd"] = round((price - ep) * sz, 4)
            s["pnl_pct"] = round(((price - ep) / ep) * 100, 4)
        else:
            s["pnl_usd"] = round((ep - price) * sz, 4)
            s["pnl_pct"] = round(((ep - price) / ep) * 100, 4)

        ob = await get_order_book(okx_sym)
        ratio, wall = classify_ob_imbalance(ob["bids"], ob["asks"], price)
        s["ob_imbalance"]   = ratio
        s["liq_wall_flagged"] = wall

        oi_now = await get_open_interest(okx_sym)
        oi_chg = (oi_now - s["_prev_oi"]) / (s["_prev_oi"] + 1e-9)
        s["_prev_oi"]     = oi_now
        s["oi_delta_pct"] = round(oi_chg * 100, 3)

        gecko_data = await get_gecko_data(gecko_id)
        price_chg  = gecko_data["price_change_24h_pct"] / 100
        s["oi_shift"] = classify_oi_signal(price_chg, oi_chg)
        s["gecko_data"] = gecko_data

        c5m  = await get_candles(okx_sym, "5m",  50)
        c4h  = await get_candles(okx_sym, "4H",  50)
        c15m = await get_candles(okx_sym, "15m", 50)

        mom_5m, tr_4h = classify_mtf(c5m, c4h)
        s["momentum_5m"] = mom_5m
        s["trend_4h"]    = tr_4h
        s["mtf_coherence"] = "ALIGNED" if (
            (s["direction"] == "LONG"  and tr_4h == "BULLISH") or
            (s["direction"] == "SHORT" and tr_4h == "BEARISH")
        ) else "COUNTER-TREND"

        bef, vef = compute_body_expansion(c15m)
        s["body_expansion"]   = bef
        s["volume_expansion"] = vef
        s["cascade_signal"]   = bef > 1.30 and vef > 1.40

        fr = await get_funding_rate(okx_sym)
        s["funding_rate"]     = fr
        s["volatility_buffer"] = abs(fr) > 0.0008

        s["health_score"] = compute_health(s)
        s["live_verdict"], s["playbook"] = generate_verdict(s)
        s["signal_status"] = "LIVE"
        s["last_update"]   = time.time()

    except Exception as e:
        s["playbook"]      = "Data error: {}. Retrying...".format(str(e))
        s["signal_status"] = "ERROR"

async def monitor_loop(symbol):
    while symbol in positions:
        await update_position(symbol)
        await asyncio.sleep(60)

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    path = os.path.join(os.path.dirname(__file__), "monitor_dashboard.html")
    if os.path.exists(path):
        return FileResponse(path, media_type="text/html")
    return HTMLResponse("<h2>monitor_dashboard.html not found</h2>", status_code=404)

@app.post("/init")
async def init_position(body: dict):
    sym  = str(body.get("symbol", "")).upper()
    dirn = str(body.get("direction", "LONG")).upper()
    ep   = float(body.get("entry_price", 0))
    sz   = float(body.get("position_size", 0))

    if not sym or ep <= 0 or sz <= 0:
        raise HTTPException(400, "symbol, entry_price, position_size required and > 0")

    okx_sym  = to_okx_symbol(sym)
    gecko_id = to_gecko_id(sym)

    try:
        price = await get_price(okx_sym)
    except Exception as e:
        raise HTTPException(400, "Cannot fetch {} from OKX: {}".format(okx_sym, str(e)))

    positions[sym] = {
        "symbol": sym, "okx_sym": okx_sym, "gecko_id": gecko_id,
        "direction": dirn, "entry_price": ep, "position_size": sz,
        "started_at": time.time(), "last_update": time.time(),
        "current_price": price,
        "pnl_usd": 0.0, "pnl_pct": 0.0,
        "ob_imbalance": 0.0, "liq_wall_flagged": False,
        "oi_shift": "UNKNOWN", "oi_delta_pct": 0.0,
        "mtf_coherence": "UNKNOWN", "trend_4h": "UNKNOWN", "momentum_5m": "UNKNOWN",
        "body_expansion": 0.0, "volume_expansion": 0.0, "cascade_signal": False,
        "funding_rate": 0.0, "volatility_buffer": False,
        "health_score": 50.0,
        "live_verdict": "INITIALISING...",
        "playbook": "First data cycle in progress...",
        "signal_status": "PENDING",
        "gecko_data": {},
        "_prev_oi": 0.0,
    }

    asyncio.create_task(monitor_loop(sym))
    return {"status": "initialized", "symbol": sym, "okx_symbol": okx_sym}

@app.get("/state/{symbol}")
async def get_state(symbol: str):
    sym = symbol.upper()
    if sym not in positions:
        raise HTTPException(404, "Position not found")
    s = positions[sym]
    return {k: v for k, v in s.items() if not k.startswith("_")}

@app.delete("/close/{symbol}")
async def close_position(symbol: str):
    sym = symbol.upper()
    if sym in positions:
        del positions[sym]
        return {"status": "closed"}
    raise HTTPException(404, "Position not found")

@app.get("/health")
async def health():
    return {"status": "ok", "positions": list(positions.keys())}
