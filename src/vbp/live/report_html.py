from __future__ import annotations
"""Vygeneruje statický HTML dashboard z live_state (pro GitHub Pages).

Běží v CI po každém cronu: `python -m vbp.live.report_html docs/index.html`.
Čísla bere ze stejného summarize() jako textový report, plus rozepisuje kompletní
deník sázek (kdy zachyceno, na co, jak vsazeno, jak dopadlo) a graf bankrollu.
Self-contained.

Sázky, které kvůli starému bugu zůstaly bez výsledku a už se nedopočítají
(výkop starší než okno scores API), se z reportu vyhazují - viz `_is_dead`.
"""
import html
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path
from vbp.metrics import clv as clv_fn
from .report import summarize

# league_tier a book_type přeložené do lidské češtiny (výstup grotexity naming panelu)
_TIER_CZ = {"liquid": "Sledovaná liga", "neglected": "Okrajová liga"}
_BOOK_CZ = {"soft": "Běžná sázkovka", "exchange": "Sázková burza", "sharp": "Referenční trh"}
_BOOK_PILL = {"soft": "sázkovka", "exchange": "burza", "sharp": "referenční"}
_LEAGUE_CZ = {
    "soccer_brazil_campeonato": "Brazílie Série A",
    "soccer_sweden_superettan": "Švédsko Superettan",
}

# Sázka, která zůstala bez výsledku a jejíž výkop je starší než tolik dnů, se už
# nikdy nedopočítá (scores API vrací jen ~3 dny nazpět). Takové v reportu skrýváme.
_DEAD_AFTER_DAYS = 4

# Modelový vklad pro sloupec "kolik by to bylo v penězích".
STAKE_EUR = 10
# Modelový startovní kapitál pro kumulativní graf bankrollu.
BANKROLL_START_EUR = 500.0


def _pct(x: float) -> str:
    return f"{x * 100:+.2f} %".replace(".", ",")


def _sig(x: float) -> str:
    return f"{x:+.2f}".replace(".", ",")


def _eur(units: float) -> str:
    """Zisk/ztráta v jednotkách -> eura při STAKE_EUR na sázku."""
    return f"{units * STAKE_EUR:+.1f} €".replace(".", ",")


def _eur_abs(amount: float, signed: bool = False) -> str:
    """Částka v eurech (ne v jednotkách)."""
    s = f"{amount:+.2f}" if signed else f"{amount:.2f}"
    return s.replace(".", ",") + " €"


def _parse(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _fmt_dt(ts: str) -> str:
    d = _parse(ts)
    return d.strftime("%d.%m. %H:%M") if d else "-"


def _tip_label(b: dict) -> str:
    return {"H": b.get("home", "domácí"), "A": b.get("away", "hosté"), "D": "remíza"}[b["outcome"]]


def _is_dead(bet: dict, now: datetime) -> bool:
    """True = čekající sázka, jejíž zápas dávno proběhl a výsledek se už nedopočítá."""
    if bet["status"] != "pending":
        return False
    k = _parse(bet.get("kickoff", ""))
    return bool(k and k < now - timedelta(days=_DEAD_AFTER_DAYS))


def enrich_bets(store) -> list[dict]:
    """Ke každé sázce dolepí stav z linie: čeká / výhra / prohra + CLV + P/L (v jednotkách)."""
    lines = store.load_lines()
    out = []
    for b in store.load_bets():
        ln = lines.get(b["match_id"])
        settled = bool(ln and ln.get("settled") and ln.get("result"))
        rec = {**b, "status": "pending", "result": None, "clv": None, "pnl": None}
        if settled:
            won = ln["result"] == b["outcome"]
            rec["status"] = "won" if won else "lost"
            rec["result"] = ln["result"]
            rec["pnl"] = (b["price"] - 1.0) if won else -1.0
            if ln.get("pin_close"):
                rec["clv"] = clv_fn(b["price"], ln["pin_close"][b["outcome"]])
        out.append(rec)
    # nejdřív čekající (podle výkopu vzestupně), pak vyřízené (podle výkopu sestupně)
    out.sort(key=lambda r: (r["status"] != "pending", r.get("kickoff") or ""))
    return out


def build_context(store, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    rep = summarize(store)
    all_bets = enrich_bets(store)
    dead = sum(1 for b in all_bets if _is_dead(b, now))
    bets = [b for b in all_bets if not _is_dead(b, now)]  # deník bez mrtvých sázek
    done = [b for b in bets if b["status"] in ("won", "lost")]
    wins = sum(1 for b in done if b["status"] == "won")
    pnl = sum(b["pnl"] for b in done if b["pnl"] is not None)
    return {
        "generated": now,
        "total_bets": len(bets),          # jen živé (bez mrtvých) - ať sedí s deníkem
        "settled_bets": len(done),
        "dropped": dead,
        "by": rep["by"],
        "bets": bets,
        "done": len(done),
        "wins": wins,
        "losses": len(done) - wins,
        "pnl": pnl,
    }


def _bucket_rows_html(by: dict) -> str:
    if not by:
        return ('<tr><td colspan="8" class="muted">Zatím žádná dohraná sázka - '
                'čekáme na první odehrané zápasy.</td></tr>')
    out = []
    for (bt, tier), g in sorted(by.items()):
        lo, hi = g["clv_ci"]
        clv_cls = "pos" if g["mean_clv"] > 0 else "neg"
        ci = f"{_pct(lo)} až {_pct(hi)}" if g["n"] > 1 else "jen 1 sázka"
        pill = "soft" if bt == "soft" else "ex"
        money = g["roi"] * g["n"]                      # profit v jednotkách za skupinu
        money_cls = "pos" if money > 0 else ("neg" if money < 0 else "")
        out.append(
            f'<tr>'
            f'<td><span class="pill {pill}">{html.escape(_BOOK_CZ.get(bt, bt))}</span></td>'
            f'<td>{html.escape(_TIER_CZ.get(tier, tier))}</td>'
            f'<td class="n num">{g["n"]}</td><td class="n num">{g["wins"]}</td>'
            f'<td class="n num {clv_cls}">{_pct(g["mean_clv"])}</td>'
            f'<td class="num muted">{ci}</td>'
            f'<td class="n num">{_pct(g["roi"])}</td>'
            f'<td class="n num {money_cls}">{_eur(money)}</td></tr>')
    return "\n".join(out)


_STATUS = {
    "pending": ('<span class="pill wait">čeká</span>', ""),
    "won":     ('<span class="pill win">výhra</span>', "pos"),
    "lost":    ('<span class="pill loss">prohra</span>', "neg"),
}


def _log_rows_html(bets: list[dict]) -> str:
    if not bets:
        return '<tr><td colspan="12" class="muted">Zatím žádné sázky.</td></tr>'
    out = []
    for b in bets:
        badge, _ = _STATUS[b["status"]]
        clv = _pct(b["clv"]) if b["clv"] is not None else "-"
        clv_cls = "" if b["clv"] is None else ("pos" if b["clv"] > 0 else "neg")
        pnl = _sig(b["pnl"]) if b["pnl"] is not None else "-"
        eur = _eur(b["pnl"]) if b["pnl"] is not None else "-"
        pnl_cls = "" if b["pnl"] is None else ("pos" if b["pnl"] > 0 else "neg")
        pill = "soft" if b["book_type"] == "soft" else "ex"
        out.append(
            f'<tr>'
            f'<td class="num muted">{_fmt_dt(b.get("ts_detected", ""))}</td>'
            f'<td class="num muted">{_fmt_dt(b.get("kickoff", ""))}</td>'
            f'<td>{html.escape(_LEAGUE_CZ.get(b.get("league",""), b.get("league","")))}</td>'
            f'<td>{html.escape(b.get("home","?"))} – {html.escape(b.get("away","?"))}</td>'
            f'<td><b>{html.escape(_tip_label(b))}</b></td>'
            f'<td class="n num">{b["price"]:.2f}</td>'
            f'<td>{html.escape(b["book"])} <span class="pill {pill}">{_BOOK_PILL.get(b["book_type"],b["book_type"])}</span></td>'
            f'<td class="n num">{_pct(b["edge"])}</td>'
            f'<td>{badge}</td>'
            f'<td class="n num {clv_cls}">{clv}</td>'
            f'<td class="n num {pnl_cls}">{pnl}</td>'
            f'<td class="n num {pnl_cls}">{eur}</td>'
            f'</tr>')
    return "\n".join(out)


def bankroll_points(
    bets: list[dict],
    stake: float = STAKE_EUR,
    start: float = BANKROLL_START_EUR,
) -> list[dict]:
    """Kumulativní bankroll po dohraných sázkách (se startovním bodem)."""
    done = [b for b in bets if b["status"] in ("won", "lost")]
    done.sort(key=lambda b: (b.get("kickoff") or "", b.get("ts_detected") or ""))
    points: list[dict] = [{
        "i": 0, "date": "start", "label": "start", "match": "", "tip": "",
        "price": None, "status": None, "pnl": 0.0, "bank": start,
    }]
    cum = 0.0
    for i, b in enumerate(done, 1):
        pnl_e = b["pnl"] * stake
        cum += pnl_e
        tip = {"H": b.get("home"), "A": b.get("away"), "D": "remíza"}[b["outcome"]]
        ko = b.get("kickoff") or ""
        points.append({
            "i": i,
            "date": ko[:10],
            "label": f"{ko[8:10]}.{ko[5:7]}." if len(ko) >= 10 else ko,
            "match": f"{b.get('home', '?')} – {b.get('away', '?')}",
            "tip": tip or "",
            "price": b["price"],
            "status": b["status"],
            "pnl": round(pnl_e, 2),
            "bank": round(start + cum, 2),
        })
    return points


def _render_bankroll_section(bets: list[dict]) -> str:
    """SVG graf kumulativního bankrollu + minitiles, ve stylu hlavního dashboardu."""
    points = bankroll_points(bets)
    if len(points) <= 1:
        return (
            '<section id="bankroll">\n'
            '    <div class="eyebrow">Bankroll</div>\n'
            '    <h2 class="serif" style="margin-top:6px">Vývoj bankrollu</h2>\n'
            '    <p class="lead-in">Model: start <b>'
            f'{BANKROLL_START_EUR:.0f} €</b>, stejný vklad <b>{STAKE_EUR:.0f} €</b> na každou sázku.</p>\n'
            '    <div class="note">Zatím žádná dohraná sázka - graf se objeví po prvním výsledku.</div>\n'
            '  </section>'
        )

    W, H = 960, 400
    pad_l, pad_r, pad_t, pad_b = 64, 28, 28, 52
    cw, ch = W - pad_l - pad_r, H - pad_t - pad_b
    banks = [p["bank"] for p in points]
    ymin = math.floor((min(banks) - 20) / 20) * 20
    ymax = math.ceil((max(banks) + 20) / 20) * 20
    if ymax <= ymin:
        ymax = ymin + 40
    n = len(points)
    start = BANKROLL_START_EUR

    def x_at(i: int) -> float:
        return pad_l if n <= 1 else pad_l + (i / (n - 1)) * cw

    def y_at(bank: float) -> float:
        return pad_t + (1 - (bank - ymin) / (ymax - ymin)) * ch

    coords = [(x_at(i), y_at(p["bank"])) for i, p in enumerate(points)]
    path_d = "M " + " L ".join(f"{x:.2f},{y:.2f}" for x, y in coords)
    y_bottom = y_at(ymin)
    area_d = (
        path_d
        + f" L {coords[-1][0]:.2f},{y_bottom:.2f} L {coords[0][0]:.2f},{y_bottom:.2f} Z"
    )

    y_grid: list[str] = []
    for t in range(int(ymin), int(ymax) + 1, 20):
        y = y_at(t)
        y_grid.append(
            f'<line class="cgrid" x1="{pad_l}" y1="{y:.2f}" x2="{W - pad_r}" y2="{y:.2f}"/>'
        )
        y_grid.append(
            f'<text class="caxis" x="{pad_l - 10}" y="{y + 4:.2f}" text-anchor="end">{t}</text>'
        )

    seen: set[str] = set()
    x_lab_svg: list[str] = []
    for i, p in enumerate(points):
        if p["date"] == "start":
            lab, key = "start", "start"
        elif p["date"] not in seen:
            seen.add(p["date"])
            lab, key = p["label"], p["date"]
        else:
            continue
        x = x_at(i)
        x_lab_svg.append(
            f'<text class="caxis" x="{x:.2f}" y="{H - pad_b + 20}" text-anchor="middle">'
            f'{html.escape(lab)}</text>'
        )

    circles: list[str] = []
    for i, p in enumerate(points):
        x, y = coords[i]
        cls = "pt-start" if i == 0 else ("pt-win" if p["status"] == "won" else "pt-loss")
        price = "" if p["price"] is None else f'{p["price"]:.2f}'
        circles.append(
            f'<circle class="pt {cls}" cx="{x:.2f}" cy="{y:.2f}" r="4.5" '
            f'data-i="{i}" data-bank="{p["bank"]}" data-pnl="{p["pnl"]}" '
            f'data-match="{html.escape(p["match"], quote=True)}" '
            f'data-tip="{html.escape(p["tip"] or "", quote=True)}" '
            f'data-price="{price}" data-status="{p["status"] or "start"}" '
            f'data-date="{html.escape(p["date"], quote=True)}"/>'
        )

    y_start = y_at(start)
    final = points[-1]["bank"]
    pnl = final - start
    pnl_cls = "pos" if pnl >= 0 else "neg"
    roi = pnl / start * 100 if start else 0.0
    roi_str = f"{roi:+.1f}".replace(".", ",")
    mid_y = pad_t + ch / 2
    n_bets = len(points) - 1
    wins = sum(1 for p in points if p["status"] == "won")
    losses = sum(1 for p in points if p["status"] == "lost")

    return f'''  <section id="bankroll">
    <div class="eyebrow">Bankroll</div>
    <h2 class="serif" style="margin-top:6px">Vývoj bankrollu</h2>
    <p class="lead-in">Modelový průběh jmění: start <b>{start:.0f} €</b>, stejný vklad
      <b>{STAKE_EUR:.0f} €</b> na každou sázku · {n_bets} dohraných ({wins}–{losses}).
      Není to reálný účet - jen převod papírových jednotek na peníze.</p>
    <div class="tiles" style="margin-bottom:16px">
      <div class="tile"><div class="k num">{start:.0f} €</div><div class="l">start</div></div>
      <div class="tile"><div class="k num {pnl_cls}">{_eur_abs(final)}</div><div class="l">teď</div></div>
      <div class="tile"><div class="k num {pnl_cls}">{_eur_abs(pnl, signed=True)}</div>
        <div class="l">P/L ({roi_str} %)</div></div>
      <div class="tile"><div class="k num">{_eur_abs(min(banks))}</div><div class="l">minimum</div></div>
      <div class="tile"><div class="k num">{_eur_abs(max(banks))}</div><div class="l">maximum</div></div>
    </div>
    <div class="chart-card" id="bank-card">
      <div class="chart-title">Kumulativní jmění po sázkách</div>
      <div class="tooltip" id="bank-tip"></div>
      <svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="Graf bankrollu">
        <defs>
          <linearGradient id="bank-ag" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="var(--teal)" stop-opacity="0.28"/>
            <stop offset="100%" stop-color="var(--teal)" stop-opacity="0.02"/>
          </linearGradient>
        </defs>
        {''.join(y_grid)}
        <line class="start-line" x1="{pad_l}" y1="{y_start:.2f}" x2="{W - pad_r}" y2="{y_start:.2f}"/>
        <text class="start-label" x="{W - pad_r}" y="{y_start - 6:.2f}" text-anchor="end">start {start:.0f} €</text>
        <path class="area" d="{area_d}"/>
        <path class="cline" d="{path_d}"/>
        {''.join(circles)}
        {''.join(x_lab_svg)}
        <text class="caxis" x="16" y="{mid_y:.0f}" text-anchor="middle"
              transform="rotate(-90 16 {mid_y:.0f})">Bankroll (€)</text>
      </svg>
      <div class="legend">
        <span><i class="dot s"></i> Start</span>
        <span><i class="dot w"></i> Výhra</span>
        <span><i class="dot l"></i> Prohra</span>
        <span class="muted">— — startovní kapitál</span>
      </div>
    </div>
  </section>
  <script>
  (function () {{
    var tip = document.getElementById("bank-tip");
    var card = document.getElementById("bank-card");
    if (!tip || !card) return;
    function fmt(n) {{
      return n.toFixed(2).replace(".", ",") + " €";
    }}
    function position(e) {{
      var r = card.getBoundingClientRect();
      var x = e.clientX - r.left + 14;
      var y = e.clientY - r.top + 14;
      var tw = tip.offsetWidth || 220;
      var th = tip.offsetHeight || 100;
      if (x + tw > r.width - 8) x = e.clientX - r.left - tw - 12;
      if (y + th > r.height - 8) y = e.clientY - r.top - th - 8;
      tip.style.left = x + "px";
      tip.style.top = y + "px";
    }}
    card.querySelectorAll("circle.pt").forEach(function (el) {{
      el.addEventListener("mouseenter", function (e) {{
        el.classList.add("active");
        var st = el.dataset.status;
        var bank = fmt(Number(el.dataset.bank));
        var body;
        if (st === "start") {{
          body = '<div class="t">Startovní kapitál</div>' +
            '<div class="row"><span class="m">Bankroll</span><span><b>' + bank + "</b></span></div>";
        }} else {{
          var pnl = Number(el.dataset.pnl);
          var pnlS = (pnl >= 0 ? "+" : "") + Math.abs(pnl).toFixed(2).replace(".", ",") + " €";
          var res = st === "won"
            ? '<span class="pos">výhra</span>'
            : '<span class="neg">prohra</span>';
          var price = el.dataset.price || "-";
          body = '<div class="t">' + el.dataset.match + "</div>" +
            '<div class="m">' + el.dataset.date + " · tip: " + el.dataset.tip +
            " @ " + price + "</div>" +
            '<div class="row"><span class="m">Výsledek</span><span>' + res + "</span></div>" +
            '<div class="row"><span class="m">P/L sázky</span><span class="' +
            (pnl >= 0 ? "pos" : "neg") + '">' + pnlS + "</span></div>" +
            '<div class="row"><span class="m">Bankroll</span><span><b>' + bank + "</b></span></div>";
        }}
        tip.innerHTML = body;
        tip.classList.add("show");
        position(e);
      }});
      el.addEventListener("mousemove", position);
      el.addEventListener("mouseleave", function () {{
        el.classList.remove("active");
        tip.classList.remove("show");
      }});
    }});
  }})();
  </script>'''


def render_html(ctx: dict) -> str:
    gen = ctx["generated"].strftime("%d. %m. %Y %H:%M UTC")
    record = f'{ctx["wins"]}-{ctx["losses"]}' if ctx["done"] else "-"
    pnl_cls = "pos" if ctx["pnl"] > 0 else ("neg" if ctx["pnl"] < 0 else "")
    pnl_str = _sig(ctx["pnl"]) + " j." if ctx["done"] else "-"
    eur_str = _eur(ctx["pnl"]) if ctx["done"] else "-"
    return _TEMPLATE.format(
        gen=gen,
        total_bets=ctx["total_bets"],
        settled_bets=ctx["settled_bets"],
        dropped=ctx["dropped"],
        record=record,
        pnl_str=pnl_str,
        pnl_cls=pnl_cls,
        eur_str=eur_str,
        stake_eur=STAKE_EUR,
        bucket_rows=_bucket_rows_html(ctx["by"]),
        log_rows=_log_rows_html(ctx["bets"]),
        bankroll_section=_render_bankroll_section(ctx["bets"]),
    )


_TEMPLATE = """<!doctype html>
<html lang="cs">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Vyplácí se porovnávat kurzy - živý deník</title>
<style>
  :root{{
    --bg:#f7f4ee;--panel:#fffdf8;--ink:#2b2620;--ink-soft:#6b6357;
    --line:#e4ddd0;--line-strong:#d2c8b6;--rust:#b5462a;--teal:#2f6f6a;--ochre:#c98a2b;
    --rust-soft:#f0d9d1;--teal-soft:#d3e6e3;--ochre-soft:#f2e4c8;--wait-soft:#e8e2d4;
    --shadow:0 1px 2px rgba(60,48,30,.06),0 8px 24px rgba(60,48,30,.05);
  }}
  @media (prefers-color-scheme:dark){{
    :root{{--bg:#1a1815;--panel:#221f1a;--ink:#ece5d8;--ink-soft:#a49a89;
    --line:#332e26;--line-strong:#453e33;--rust:#e06b4a;--teal:#57a39c;--ochre:#dda63f;
    --rust-soft:#3a271f;--teal-soft:#1f322f;--ochre-soft:#352a17;--wait-soft:#2c2820;
    --shadow:0 1px 2px rgba(0,0,0,.3),0 8px 24px rgba(0,0,0,.35);}}
  }}
  *{{box-sizing:border-box}}
  body{{margin:0;background:var(--bg);color:var(--ink);
    font-family:system-ui,-apple-system,"Segoe UI",sans-serif;line-height:1.55;-webkit-font-smoothing:antialiased}}
  .wrap{{max-width:1080px;margin:0 auto;padding:clamp(20px,5vw,56px) clamp(16px,4vw,32px) 80px}}
  .serif{{font-family:Charter,"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif}}
  .num{{font-variant-numeric:tabular-nums;font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace}}
  .eyebrow{{font-size:.72rem;letter-spacing:.13em;text-transform:uppercase;color:var(--ink-soft);font-weight:600}}
  h1{{font-size:clamp(1.9rem,4.5vw,2.6rem);line-height:1.1;margin:.35em 0 .1em;text-wrap:balance;font-weight:600}}
  h2{{font-size:1.26rem;margin:0 0 .2em;font-weight:600}}
  .sub{{color:var(--ink-soft);font-size:1.02rem;max-width:64ch}}
  header{{border-bottom:1px solid var(--line);padding-bottom:26px;margin-bottom:30px}}
  .stamp{{margin-top:14px;font-size:.82rem;color:var(--ink-soft)}}
  .stamp b{{color:var(--teal)}}
  section{{margin:38px 0}}
  .lead-in{{color:var(--ink-soft);max-width:70ch;margin:.2em 0 1.3em}}
  .tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px}}
  .tile{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px;box-shadow:var(--shadow)}}
  .tile .k{{font-size:2.1rem;line-height:1;font-weight:600}}
  .tile .l{{font-size:.83rem;color:var(--ink-soft);margin-top:7px}}
  .tile.good .k{{color:var(--teal)}}
  .tbl-wrap{{overflow-x:auto;border:1px solid var(--line);border-radius:12px;box-shadow:var(--shadow)}}
  table{{border-collapse:collapse;width:100%;background:var(--panel);font-size:.88rem}}
  th,td{{padding:9px 13px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}}
  thead th{{font-size:.68rem;letter-spacing:.05em;text-transform:uppercase;color:var(--ink-soft);font-weight:600;position:sticky;top:0;background:var(--panel)}}
  tbody tr:last-child td{{border-bottom:none}}
  tbody tr:hover td{{background:var(--bg)}}
  td.n{{text-align:right}}
  .pos{{color:var(--teal);font-weight:600}}
  .neg{{color:var(--rust);font-weight:600}}
  .muted{{color:var(--ink-soft)}}
  .pill{{display:inline-block;font-size:.68rem;padding:1px 8px;border-radius:20px;font-weight:600;letter-spacing:.02em}}
  .pill.soft{{background:var(--teal-soft);color:var(--teal)}}
  .pill.ex{{background:var(--ochre-soft);color:var(--ochre)}}
  .pill.win{{background:var(--teal-soft);color:var(--teal)}}
  .pill.loss{{background:var(--rust-soft);color:var(--rust)}}
  .pill.wait{{background:var(--wait-soft);color:var(--ink-soft)}}
  .note{{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--line-strong);
    border-radius:8px;padding:14px 18px;font-size:.9rem;color:var(--ink-soft);margin-top:14px}}
  .note b{{color:var(--ink)}}
  dl.gloss{{margin:14px 0 0;display:grid;gap:8px;font-size:.88rem}}
  dl.gloss div{{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}}
  dl.gloss dt{{font-weight:600;color:var(--ink);min-width:160px}}
  dl.gloss dd{{margin:0;color:var(--ink-soft);max-width:64ch}}
  ul.caveats{{list-style:none;padding:0;margin:0;display:grid;gap:10px}}
  ul.caveats li{{padding-left:22px;position:relative;color:var(--ink-soft);font-size:.92rem;max-width:74ch}}
  ul.caveats li::before{{content:"";position:absolute;left:4px;top:.62em;width:6px;height:6px;border-radius:50%;background:var(--rust)}}
  footer{{margin-top:48px;padding-top:20px;border-top:1px solid var(--line);color:var(--ink-soft);font-size:.8rem}}
  code{{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:.86em;background:var(--line);padding:1px 6px;border-radius:5px}}
  .chart-card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 10px 12px;
    box-shadow:var(--shadow);position:relative}}
  .chart-title{{font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;color:var(--ink-soft);
    font-weight:600;margin:0 0 8px 12px}}
  svg.chart{{width:100%;height:auto;display:block}}
  .cgrid{{stroke:var(--line);stroke-width:1}}
  .caxis{{fill:var(--ink-soft);font-size:11px;font-family:system-ui,-apple-system,"Segoe UI",sans-serif}}
  .area{{fill:url(#bank-ag)}}
  .cline{{fill:none;stroke:var(--teal);stroke-width:2.4;stroke-linejoin:round;stroke-linecap:round}}
  .start-line{{stroke:var(--ink-soft);stroke-width:1.2;stroke-dasharray:5 5;opacity:.7}}
  .start-label{{fill:var(--ink-soft);font-size:11px;font-family:system-ui,-apple-system,"Segoe UI",sans-serif}}
  .pt{{stroke:var(--panel);stroke-width:1.5;cursor:pointer}}
  .pt:hover,.pt.active{{r:7}}
  .pt-start{{fill:var(--ochre)}}
  .pt-win{{fill:var(--teal)}}
  .pt-loss{{fill:var(--rust)}}
  .tooltip{{position:absolute;pointer-events:none;z-index:5;background:var(--panel);border:1px solid var(--line-strong);
    border-radius:10px;padding:10px 12px;min-width:200px;box-shadow:var(--shadow);font-size:.84rem;line-height:1.4;
    opacity:0;transform:translateY(4px);transition:opacity .12s;color:var(--ink)}}
  .tooltip.show{{opacity:1;transform:none}}
  .tooltip .t{{font-weight:650;margin-bottom:4px}}
  .tooltip .m{{color:var(--ink-soft);font-size:.8rem}}
  .tooltip .row{{display:flex;justify-content:space-between;gap:16px;margin-top:3px}}
  .legend{{display:flex;flex-wrap:wrap;gap:14px 20px;margin:14px 8px 0;font-size:.82rem;color:var(--ink-soft)}}
  .legend span{{display:inline-flex;align-items:center;gap:6px}}
  .dot{{width:9px;height:9px;border-radius:50%;display:inline-block}}
  .dot.w{{background:var(--teal)}}
  .dot.l{{background:var(--rust)}}
  .dot.s{{background:var(--ochre)}}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="eyebrow">Value Betting Predictor &middot; živý test</div>
    <h1 class="serif">Vyplácí se porovnávat kurzy? Živý deník sázek</h1>
    <p class="sub">Systém prochází kurzy u víc sázkových kanceláří a vsadí tam, kde je nejlepší
      kurz aspoň o&nbsp;3&nbsp;% výhodnější, než odpovídá férové pravděpodobnosti (tu bereme z Pinnacle,
      nejostřejší kanceláře na trhu). Nesází se skutečné peníze - každá sázka je jen na papíře a jen se
      zaznamenává, jak by dopadla. Stránka se přepisuje sama po každém automatickém běhu.</p>
    <p class="stamp">Aktualizováno: <b>{gen}</b> &middot; automaticky, po každém běhu</p>
  </header>

  <section>
    <div class="eyebrow">Souhrn</div>
    <h2 class="serif" style="margin-top:6px">Kde jsme teď</h2>
    <p class="lead-in">Sázka se zapíše ve chvíli, kdy zápas ještě nezačal - systém nikdy nevidí výsledek
      dopředu. Bilanci a zisk počítáme jen z už odehraných zápasů. Konečný verdikt bude dávat smysl
      až u zhruba 100 dohraných sázek u běžných sázkovek.</p>
    <div class="tiles">
      <div class="tile"><div class="k num">{total_bets}</div><div class="l">sázek zapsáno celkem</div></div>
      <div class="tile good"><div class="k num">{settled_bets}</div><div class="l">už dohráno</div></div>
      <div class="tile"><div class="k num">{record}</div><div class="l">výher - proher</div></div>
      <div class="tile"><div class="k num {pnl_cls}">{pnl_str}</div><div class="l">zisk / ztráta v jednotkách</div></div>
      <div class="tile"><div class="k num {pnl_cls}">{eur_str}</div><div class="l">v penězích při {stake_eur} € na sázku</div></div>
    </div>
    <div class="note"><b>Co znamená „j.":</b> zisk je v jednotkách, kde jedna jednotka = jeden vklad
      (na každou sázku vsázíme stejně). Kladné číslo = na papíře jsme vydělali tolik vkladů, záporné =
      tolik jsme prodělali. Poslední dlaždice to samé přepočítává na peníze, kdybychom na každou sázku
      vsadili {stake_eur} €. <b>Pozor:</b> při takhle malém počtu sázek je zisk hlavně náhoda, ne signál -
      spolehlivější je náskok u výkopu níže.</div>
    <div class="note">Kvůli staré chybě systém do 4.&nbsp;8.&nbsp;2026 zápasy nevyhodnocoval, takže dřívější
      sázky zůstaly bez výsledku a už se nedopočítají (staré výsledky nejsou k dispozici). Takových sázek
      bylo {dropped} a v deníku je proto nezobrazujeme.</div>
  </section>

{bankroll_section}

  <section>
    <div class="eyebrow">Deník</div>
    <h2 class="serif" style="margin-top:6px">Kdy a na co systém vsadil</h2>
    <p class="lead-in">Všechny sázky - nahoře ty, u kterých se ještě čeká na výsledek, dole dohrané.
      „Zachyceno" = kdy systém sázku našel, „výkop" = začátek zápasu, „tip" = na koho jsme vsadili.</p>
    <dl class="gloss">
      <div><dt>Kurz</dt><dd>nejlepší cena, kterou jsme napříč sázkovkami na daný tip našli.</dd></div>
      <div><dt>Náskok při sázce</dt><dd>o kolik byl náš kurz výhodnější než férová pravděpodobnost z Pinnacle ve chvíli, kdy jsme sázeli.</dd></div>
      <div><dt>Náskok u výkopu</dt><dd>totéž měřené proti kurzu těsně před výkopem - klíčové číslo, kladné = chytili jsme lepší cenu, než jaká byla nakonec. (Odborně „CLV", closing line value - nemá nic společného s customer lifetime value z marketingu.)</dd></div>
      <div><dt>Zisk</dt><dd>kolik jsme na sázce vydělali/prodělali; v jednotkách (výhra = kurz-1, prohra = -1) a vedle v penězích při {stake_eur} € na sázku.</dd></div>
    </dl>
    <div class="tbl-wrap">
      <table>
        <thead><tr>
          <th>Zachyceno</th><th>Výkop</th><th>Liga</th><th>Zápas</th><th>Tip</th>
          <th class="n">Kurz</th><th>Kde sázíme</th><th class="n">Náskok při sázce</th><th>Stav</th>
          <th class="n">Náskok u výkopu</th><th class="n">Zisk</th><th class="n">Zisk ({stake_eur} €)</th>
        </tr></thead>
        <tbody>
{log_rows}
        </tbody>
      </table>
    </div>
  </section>

  <section>
    <div class="eyebrow">Přehled</div>
    <h2 class="serif" style="margin-top:6px">Jak si náskok u výkopu vede podle místa a ligy</h2>
    <p class="lead-in">Každá sázka patří do skupiny podle dvou věcí: <b>kde</b> jsme kurz vzali a <b>jak
      sledovaná</b> je daná liga. Proto jsou tady dva samostatné sloupce. Verdikt padne, až bude celé
      95% rozpětí náskoku u výkopu u běžných sázkovek nad nulou. Dokud je sázek málo, ber to jen jako
      kontrolu, ne důkaz.</p>
    <div class="tbl-wrap">
      <table>
        <thead><tr><th>Kde sázíme</th><th>Liga</th><th class="n">Sázek</th><th class="n">Výher</th>
          <th class="n">Náskok u výkopu (průměr)</th><th>Rozpětí (95 %)</th><th class="n">ROI</th>
          <th class="n">Zisk ({stake_eur} €)</th></tr></thead>
        <tbody>
{bucket_rows}
        </tbody>
      </table>
    </div>
    <dl class="gloss">
      <div><dt>Běžná sázkovka</dt><dd>klasická kancelář pro veřejnost (Bet365, Unibet, Fortuna...); reaguje pomaleji, tam hledáme výhodné kurzy.</dd></div>
      <div><dt>Sázková burza</dt><dd>místo, kde sázíš proti jiným hráčům, ne proti kanceláři (Betfair).</dd></div>
      <div><dt>Sledovaná liga</dt><dd>velký, hlídaný trh (brazilská Série A) - chyby v kurzech mizí rychle.</dd></div>
      <div><dt>Okrajová liga</dt><dd>menší soutěž, které kanceláře dávají míň pozor (švédská Superettan) - víc prostoru pro výhodný kurz.</dd></div>
      <div><dt>Náskok u výkopu (průměr)</dt><dd>průměrně o kolik byl náš kurz lepší než kurz těsně před výkopem; tohle rozhoduje o verdiktu (odborně CLV).</dd></div>
      <div><dt>ROI a Zisk ({stake_eur} €)</dt><dd>kolik bychom reálně vydělali z vsazeného (ROI v %, vedle v eurech při {stake_eur} € na sázku). Vysoké číslo tu klidně vznikne z pár šťastných výher na vysoký kurz - proto je u malého počtu sázek spíš náhoda než signál.</dd></div>
    </dl>
  </section>

  <section>
    <div class="eyebrow">Na co si dát pozor</div>
    <h2 class="serif" style="margin-top:6px">Co drží závěr při zemi</h2>
    <ul class="caveats">
      <li>„Kurz před výkopem" je poslední cena, kterou jsme stihli odečíst - ne přesně minutu před začátkem.</li>
      <li>Papírové sázky nepočítají s tím, že v praxi dostaneš o kousek horší kurz a že sázkovky
        úspěšné hráče omezují nebo vyhazují - potvrzujeme tím, že výhoda <b>existuje</b>, ne že se dá
        do nekonečna škálovat.</li>
      <li>Výpočet rozpětí bere sázky jako nezávislé, což úplně nejsou (víc tipů na tentýž zápas,
        stejná sázka u víc kanceláří) - rozpětí tak vypadá užší, než ve skutečnosti je. Ber závěr s rezervou.</li>
    </ul>
  </section>

  <footer>
    github.com/radim-tales/value-betting-predictor &middot; kurzy z The Odds API &middot;
    stránka se generuje sama po každém běhu, needitovat ručně.
  </footer>
</div>
</body>
</html>
"""


def main():
    import sys
    from .config import BETS_FILE, LINES_FILE
    from .store import Store
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs/index.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    ctx = build_context(Store(BETS_FILE, LINES_FILE))
    out.write_text(render_html(ctx), encoding="utf-8")
    print(f"[report_html] wrote {out} ({ctx['settled_bets']}/{ctx['total_bets']} settled, "
          f"record {ctx['wins']}-{ctx['losses']}, pnl {ctx['pnl']:+.2f}, dropped {ctx['dropped']})")


if __name__ == "__main__":
    main()
