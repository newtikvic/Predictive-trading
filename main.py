"""
Trade Guardian Engine v2 - Multi-Asset Command Center
Up to 5 simultaneous positions
Data: OKX + CoinGecko. Zero pydantic.
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
import httpx, asyncio, time, statistics, os

app = FastAPI(title="Trade Guardian Engine v2")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MAX_POSITIONS = 5
positions = {}
OKX_BASE   = "https://www.okx.com"
GECKO_BASE = "https://api.coingecko.com/api/v3"

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
}

def to_okx(symbol):
    s = symbol.upper()
    for suf in ["USDT","BUSD","USD"]:
        if s.endswith(suf):
            return s[:-len(suf)] + "-USDT-SWAP"
    return s + "-USDT-SWAP"

def to_gecko(symbol):
    b = symbol.upper()
    for suf in ["USDT","BUSD","USD"]:
        if b.endswith(suf): b = b[:-len(suf)]
    b = b.replace("-SWAP","").replace("-","")
    return GECKO_MAP.get(b, b.lower())

async def okx_get(path, params=None):
    async with httpx.AsyncClient(timeout=10, follow_redirects=True) as c:
        r = await c.get(OKX_BASE + path, params=params or {})
        r.raise_for_status()
        d = r.json()
        if str(d.get("code")) != "0":
            raise ValueError("OKX {}: {}".format(d.get("code"), d.get("msg")))
        return d["data"]

async def gecko_get(path, params=None):
    async with httpx.AsyncClient(timeout=10, follow_redirects=True) as c:
        r = await c.get(GECKO_BASE + path, params=params or {}, headers={"Accept":"application/json"})
        r.raise_for_status()
        return r.json()

async def okx_price(inst):
    d = await okx_get("/api/v5/market/ticker", {"instId": inst})
    return float(d[0]["last"])

async def okx_book(inst):
    d = await okx_get("/api/v5/market/books", {"instId": inst, "sz": "50"})
    return d[0]

async def okx_oi(inst):
    d = await okx_get("/api/v5/public/open-interest", {"instId": inst})
    return float(d[0]["oi"])

async def okx_funding(inst):
    d = await okx_get("/api/v5/public/funding-rate", {"instId": inst})
    return float(d[0]["fundingRate"])

async def okx_candles(inst, bar, limit=60):
    d = await okx_get("/api/v5/market/candles", {"instId": inst, "bar": bar, "limit": str(limit)})
    return list(reversed(d))

async def gecko_data(gid):
    try:
        d = await gecko_get("/simple/price", {"ids": gid, "vs_currencies": "usd",
            "include_24hr_change": "true", "include_market_cap": "true"})
        coin = d.get(gid, {})
        return {"change_24h": coin.get("usd_24h_change", 0.0), "market_cap": coin.get("usd_market_cap", 0)}
    except Exception:
        return {"change_24h": 0.0, "market_cap": 0}

def ema_series(closes, span):
    if not closes: return []
    k = 2.0 / (span + 1)
    out = [closes[0]]
    for v in closes[1:]: out.append(v * k + out[-1] * (1.0 - k))
    return out

def ema_val(closes, span):
    s = ema_series(closes, span)
    return s[-1] if s else 0.0

def calc_rsi(closes, period=14):
    if len(closes) < period + 1: return 50.0
    gains = [max(closes[i]-closes[i-1], 0) for i in range(1, len(closes))]
    losses= [max(closes[i-1]-closes[i], 0) for i in range(1, len(closes))]
    ag = statistics.mean(gains[-period:])
    al = statistics.mean(losses[-period:])
    if al == 0: return 100.0
    return round(100.0 - 100.0 / (1.0 + ag/al), 2)

def calc_macd_bull(closes):
    if len(closes) < 35: return False
    e12 = ema_series(closes, 12)
    e26 = ema_series(closes, 26)
    ml  = [e12[i]-e26[i] for i in range(len(e26))]
    sl  = ema_series(ml, 9)
    return ml[-1] > sl[-1]

def calc_vol_ratio(candles):
    if len(candles) < 21: return 1.0
    vols = [float(k[5]) for k in candles]
    avg  = statistics.mean(vols[-20:-1]) if len(vols)>=20 else statistics.mean(vols[:-1])
    cur  = float(candles[-2][5])
    return round(cur/avg, 3) if avg else 1.0

def ob_split(bids, asks, price):
    lo, hi = price*0.98, price*1.02
    bv = sum(float(b[1]) for b in bids if float(b[0])>=lo)
    av = sum(float(a[1]) for a in asks if float(a[0])<=hi)
    t  = bv + av
    if t == 0: return 0.0, False
    ratio = bv/(av+1e-9)
    wall  = max(bv,av) > 3*(t/2)
    return round(ratio,3), wall

def body_exp(candles):
    if len(candles) < 22: return 1.0, 1.0
    done = candles[:-1]; trig = done[-1]
    bodies = [abs(float(k[4])-float(k[1])) for k in done[:-1]]
    vols   = [float(k[5]) for k in done[:-1]]
    ab = statistics.mean(bodies[-20:]) if len(bodies)>=20 else (statistics.mean(bodies) if bodies else 1)
    av = statistics.mean(vols[-20:])   if len(vols)>=20   else (statistics.mean(vols)   if vols   else 1)
    bef = abs(float(trig[4])-float(trig[1]))/ab if ab else 1.0
    vef = float(trig[5])/av if av else 1.0
    return round(bef,3), round(vef,3)

def analyze_tf(candles, direction):
    closes = [float(k[4]) for k in candles]
    if len(closes) < 10:
        return {"score":0,"aligned":False,"rsi":50,"macd_bull":False,
                "ema9":0,"ema21":0,"ema50":0,"price_above_ema21":False,
                "rsi_ok":False,"macd_ok":False,"vol_ratio":1.0,"price":0}
    price = closes[-1]
    e9, e21, e50 = ema_val(closes,9), ema_val(closes,21), ema_val(closes,50)
    rsi_v = calc_rsi(closes,14)
    macd_b = calc_macd_bull(closes)
    vr = calc_vol_ratio(candles)
    if direction == "LONG":
        p_ok = price > e21
        r_ok = rsi_v > 50
        m_ok = macd_b
    else:
        p_ok = price < e21
        r_ok = rsi_v < 50
        m_ok = not macd_b
    score = sum([p_ok, r_ok, m_ok])
    return {"price":round(price,6),"ema9":round(e9,6),"ema21":round(e21,6),"ema50":round(e50,6),
            "rsi":rsi_v,"macd_bull":macd_b,"price_above_ema21":p_ok,
            "rsi_ok":r_ok,"macd_ok":m_ok,"vol_ratio":vr,"score":score,"aligned":score>=2}

def mtf_alignment(tf5m, tf15m, tf1h, tf4h, direction):
    ac = sum(1 for tf in [tf5m,tf15m,tf1h] if tf["aligned"])
    if direction == "LONG":
        conflict = tf1h["score"]==0 and tf1h["rsi"]<45 and tf5m["score"]>=2
    else:
        conflict = tf1h["score"]==0 and tf1h["rsi"]>55 and tf5m["score"]>=2
    if conflict:
        status, color = "CONFLICT - HIGH RISK", "red"
    elif ac>=2 and tf15m["aligned"]:
        status, color = "IN ALIGNMENT", "green"
    elif ac>=2:
        status, color = "PARTIAL ALIGNMENT", "orange"
    elif ac==1:
        status, color = "WEAK - MONITOR", "orange"
    else:
        status, color = "OUT OF ALIGNMENT", "red"
    return {"status":status,"color":color,"aligned_count":ac,
            "scores":{"5m":tf5m["score"],"15m":tf15m["score"],"1h":tf1h["score"],"4h":tf4h["score"]}}

def struct_verdict(s, tf15m, tf1h):
    d      = s["direction"]
    pnl    = s["pnl_pct"]
    price  = s["current_price"]
    entry  = s["entry_price"]
    mtfc   = s["mtf"]["color"]
    e21_15 = tf15m["ema21"]
    e50_1h = tf1h["ema50"]
    r15    = tf15m["rsi"]
    r1h    = tf1h["rsi"]

    if d == "LONG":
        kl = max(e21_15, e50_1h) if e21_15>0 and e50_1h>0 else entry*0.985
        rsi_bad = r15<45 and r1h<45
        mom_ok  = r15>50 and tf15m["macd_bull"]
        dist    = round(((price-kl)/price)*100,2)
    else:
        kl = min(e21_15, e50_1h) if e21_15>0 and e50_1h>0 else entry*1.015
        rsi_bad = r15>55 and r1h>55
        mom_ok  = r15<50 and not tf15m["macd_bull"]
        dist    = round(((kl-price)/price)*100,2)

    if mtfc=="green" and mom_ok and pnl>-1.0:
        v,vc="STRUCTURE INTACT","green"
        adv="Next 3-5 candles projected to retest support and continue. Hold position."
        wadv=""
    elif mtfc=="green" and pnl>-2.5:
        v,vc="TREND WEAKENING","orange"
        adv="Pullback likely but trend not broken. Tighten stop loss."
        wadv="Wait for RSI to reclaim 50 on 15m before adding exposure."
    elif rsi_bad and pnl<-1.5:
        v,vc="STRUCTURE BROKEN","red"
        ce = max(2,int(abs(dist)/0.3)+1)
        me = ce*15
        if abs(dist)>2.0:
            adv="Do not wait: Structural shift detected. Key level ${:.4f} is {:.2f}% away.".format(kl,abs(dist))
            wadv="Est. recovery: {} candles (~{} mins). Exit now. Re-enter on 2 bullish closes on 15m.".format(ce,me)
        else:
            adv="Price approaching key level ${:.4f}. Watch for bounce.".format(kl)
            wadv="Wait for retest of ${:.4f} or 2 confirmed bullish closes on 15m.".format(round(kl,4))
    elif pnl<-3.0:
        v,vc="STRUCTURE BROKEN","red"
        adv="Trade thesis invalid. Price moved significantly against position."
        wadv="Wait for retest of ${:.4f} or 2 confirmed bullish closes on 15m.".format(round(kl,4))
    else:
        v,vc="TREND WEAKENING","orange"
        adv="Mixed signals. Reduce size or tighten stop."
        wadv="Wait for 15m MACD cross and RSI>50 before adding exposure."

    return {"verdict":v,"verdict_color":vc,"advice":adv,"wait_advice":wadv,"key_level":round(kl,6)}

def calc_health(s):
    sc = 50.0
    d, oi, mtf = s["direction"], s["oi_shift"], s["mtf"]
    if d=="LONG":
        if oi=="STRONG_INSTITUTIONAL":      sc+=15
        elif oi=="EXHAUSTION_SQUEEZE_END":  sc-=20
        elif oi=="AGGRESSIVE_SHORT_INFLOW": sc-=15
    else:
        if oi=="AGGRESSIVE_SHORT_INFLOW":   sc+=15
        elif oi=="EXHAUSTION_SQUEEZE_END":  sc-=15
        elif oi=="STRONG_INSTITUTIONAL":    sc-=10
    ac = mtf["aligned_count"]
    if ac==3: sc+=20
    elif ac==2: sc+=10
    elif ac==1: sc-=10
    else: sc-=20
    if mtf["color"]=="red":   sc-=10
    if s.get("liq_wall"):     sc-=10
    if s.get("cascade_signal"): sc+=8
    if s.get("vol_buffer"):   sc-=5
    return max(0.0,min(100.0,round(sc,1)))

def oi_sig(pc, oc):
    pu,ou = pc>0, oc>0
    if pu and ou:     return "STRONG_INSTITUTIONAL"
    if pu and not ou: return "EXHAUSTION_SQUEEZE_END"
    if not pu and ou: return "AGGRESSIVE_SHORT_INFLOW"
    return "FADING_MOMENTUM"

def calc_verdict(s):
    h   = s["health_score"]
    sv  = s.get("structural_verdict",{})
    mtf = s["mtf"]
    if sv.get("verdict")=="STRUCTURE BROKEN":
        return "ABORT POSITION - STRUCTURE BROKEN"
    if h>=72: return "STRONG HOLD - INSTITUTIONAL ACCUMULATION"
    if h>=55:
        return "WATCH CLOSELY - CONFLICT DETECTED" if mtf["color"]=="red" else "WATCH CLOSELY - CONSOLIDATION"
    if h>=35:
        return "TAKE PROFIT - MOMENTUM EXHAUSTED" if sv.get("verdict")=="TREND WEAKENING" else "WATCH CLOSELY - STRUCTURE WEAKENING"
    return "ABORT POSITION - INSIDER OUTFLOW DETECTED"

def calc_playbook(s):
    sv, mtf, d, oi = s.get("structural_verdict",{}), s["mtf"], s["direction"], s["oi_shift"]
    lines = []
    if d=="LONG":
        if oi=="STRONG_INSTITUTIONAL":       lines.append("Whales accumulating longs. OI expanding with price. Institutional backing intact.")
        elif oi=="EXHAUSTION_SQUEEZE_END":   lines.append("OI declining as price rises. Short squeeze exhaustion. Thin buyer base.")
        elif oi=="AGGRESSIVE_SHORT_INFLOW":  lines.append("Fresh shorts entering aggressively. Downside pressure building.")
    else:
        if oi=="AGGRESSIVE_SHORT_INFLOW":    lines.append("Short sellers adding conviction. OI confirms directional bias.")
        elif oi=="EXHAUSTION_SQUEEZE_END":   lines.append("Short squeeze may be ending. Monitor for reversal.")
    if mtf["status"]=="CONFLICT - HIGH RISK":
        lines.append("RULE 1 VIOLATED: 1H contradicts 5m. Counter-trend trap risk. Do not add size.")
    if not s.get("tf15m",{}).get("aligned"):
        lines.append("RULE 2: 15m unconfirmed. Wait for 15m breakout before scaling.")
    if sv.get("advice"):   lines.append(sv["advice"])
    if sv.get("wait_advice"): lines.append(sv["wait_advice"])
    if not lines: lines.append("Market structure neutral. Monitor 15m for breakout expansion.")
    return " | ".join(lines)

async def update(symbol):
    s = positions.get(symbol)
    if s is None: return
    inst, gid = s["okx_inst"], s["gecko_id"]
    try:
        price = await okx_price(inst)
        s["current_price"] = price
        ep, sz = s["entry_price"], s["position_size"]
        if s["direction"]=="LONG":
            s["pnl_usd"]=round((price-ep)*sz,4); s["pnl_pct"]=round(((price-ep)/ep)*100,4)
        else:
            s["pnl_usd"]=round((ep-price)*sz,4); s["pnl_pct"]=round(((ep-price)/ep)*100,4)

        book = await okx_book(inst)
        r, wall = ob_split(book["bids"], book["asks"], price)
        s["ob_imbalance"]=r; s["liq_wall"]=wall

        oi_now = await okx_oi(inst)
        prev   = s.get("_oi_prev", oi_now)
        oi_chg = (oi_now-prev)/(prev+1e-9)
        s["_oi_prev"]=oi_now; s["oi_delta_pct"]=round(oi_chg*100,3)

        gd = await gecko_data(gid)
        s["gecko_data"]=gd; s["oi_shift"]=oi_sig(gd["change_24h"]/100.0, oi_chg)

        c5m, c15m, c1h, c4h = await asyncio.gather(
            okx_candles(inst,"5m",60), okx_candles(inst,"15m",60),
            okx_candles(inst,"1H",60), okx_candles(inst,"4H",60))

        d_ = s["direction"]
        tf5m  = analyze_tf(c5m,  d_)
        tf15m = analyze_tf(c15m, d_)
        tf1h  = analyze_tf(c1h,  d_)
        tf4h  = analyze_tf(c4h,  d_)
        s["tf5m"]=tf5m; s["tf15m"]=tf15m; s["tf1h"]=tf1h; s["tf4h"]=tf4h
        s["mtf"] = mtf_alignment(tf5m,tf15m,tf1h,tf4h,d_)

        bef,vef = body_exp(c15m)
        s["body_expansion"]=bef; s["vol_expansion"]=vef
        s["cascade_signal"]=bef>1.30 and vef>1.40

        fr = await okx_funding(inst)
        s["funding_rate"]=fr; s["vol_buffer"]=abs(fr)>0.0008

        s["vol_ratio"]=calc_vol_ratio(c15m)
        s["health_score"]=calc_health(s)
        s["structural_verdict"]=struct_verdict(s,tf15m,tf1h)
        s["live_verdict"]=calc_verdict(s)
        s["playbook"]=calc_playbook(s)
        s["signal_status"]="LIVE"; s["last_update"]=time.time()
    except Exception as e:
        s["playbook"]="Data error: {}. Retrying...".format(str(e))
        s["signal_status"]="ERROR"

async def monitor_loop(symbol):
    while symbol in positions:
        await update(symbol)
        await asyncio.sleep(60)

@app.get("/")
async def root():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),"monitor_dashboard.html")
    if os.path.exists(p): return FileResponse(p,media_type="text/html")
    return HTMLResponse("<h2>monitor_dashboard.html not found</h2>",status_code=404)

@app.post("/init")
async def init_pos(request: Request):
    body=await request.json()
    symbol=str(body.get("symbol","")).strip().upper()
    direction=str(body.get("direction","LONG")).strip().upper()
    entry_price=float(body.get("entry_price",0))
    position_size=float(body.get("position_size",0))
    label=str(body.get("label",symbol))
    if not symbol: raise HTTPException(400,"symbol required")
    if entry_price<=0 or position_size<=0: raise HTTPException(400,"entry_price and position_size must be > 0")
    if direction not in ("LONG","SHORT"): raise HTTPException(400,"direction must be LONG or SHORT")
    if len(positions)>=MAX_POSITIONS and symbol not in positions:
        raise HTTPException(400,"Maximum {} positions reached".format(MAX_POSITIONS))
    inst=to_okx(symbol); gid=to_gecko(symbol)
    try:
        price=await okx_price(inst)
    except Exception as e:
        raise HTTPException(400,"Cannot fetch {} from OKX: {}".format(inst,str(e)))
    dtf={"score":0,"aligned":False,"rsi":50,"macd_bull":False,"ema9":0,"ema21":0,"ema50":0,
         "price_above_ema21":False,"rsi_ok":False,"macd_ok":False,"vol_ratio":1.0,"price":price}
    positions[symbol]={
        "symbol":symbol,"okx_inst":inst,"gecko_id":gid,"label":label,
        "direction":direction,"entry_price":entry_price,"position_size":position_size,
        "started_at":time.time(),"last_update":time.time(),"current_price":price,
        "pnl_usd":0.0,"pnl_pct":0.0,"ob_imbalance":0.0,"liq_wall":False,
        "oi_shift":"UNKNOWN","oi_delta_pct":0.0,
        "mtf":{"status":"INITIALISING","color":"grey","aligned_count":0,"scores":{"5m":0,"15m":0,"1h":0,"4h":0}},
        "structural_verdict":{"verdict":"INITIALISING","verdict_color":"grey","advice":"First cycle pending...","wait_advice":"","key_level":0},
        "tf5m":dtf,"tf15m":dtf,"tf1h":dtf,"tf4h":dtf,
        "body_expansion":0.0,"vol_expansion":0.0,"cascade_signal":False,
        "funding_rate":0.0,"vol_buffer":False,"vol_ratio":1.0,
        "health_score":50.0,"live_verdict":"INITIALISING...","playbook":"First data cycle in progress. Please wait ~60 seconds.",
        "signal_status":"PENDING","gecko_data":{},"_oi_prev":0.0,
    }
    asyncio.create_task(monitor_loop(symbol))
    return {"status":"initialized","symbol":symbol,"okx_instrument":inst}

@app.get("/all")
async def get_all():
    return {"count":len(positions),"max":MAX_POSITIONS,"positions":{
        sym:{"symbol":s["symbol"],"label":s["label"],"direction":s["direction"],
             "current_price":s["current_price"],"pnl_pct":s["pnl_pct"],"pnl_usd":s["pnl_usd"],
             "health_score":s["health_score"],"live_verdict":s["live_verdict"],
             "mtf_status":s["mtf"]["status"],"mtf_color":s["mtf"]["color"],
             "sv_verdict":s["structural_verdict"]["verdict"],"sv_color":s["structural_verdict"]["verdict_color"],
             "signal_status":s["signal_status"],"last_update":s["last_update"]}
        for sym,s in positions.items()}}

@app.get("/state/{symbol}")
async def get_state(symbol: str):
    sym=symbol.strip().upper()
    if sym not in positions: raise HTTPException(404,"No position for {}".format(sym))
    s=positions[sym]
    return {k:v for k,v in s.items() if not k.startswith("_")}

@app.delete("/close/{symbol}")
async def close_pos(symbol: str):
    sym=symbol.strip().upper()
    if sym in positions:
        del positions[sym]
        return {"status":"closed","symbol":sym}
    raise HTTPException(404,"No position for {}".format(sym))

@app.get("/health")
async def health():
    return {"status":"ok","active":len(positions),"max":MAX_POSITIONS}
