from __future__ import annotations
"""Vygeneruje statický HTML dashboard z live_state (pro GitHub Pages).

Běží v CI po každém cronu: `python -m vbp.live.report_html docs/index.html`.
Čísla bere ze stejného summarize() jako textový report, plus rozepisuje kompletní
deník sázek (kdy nalezeno, na co, jak vsazeno, jak dopadlo). Self-contained.
"""
import html
from datetime import datetime, timezone
from pathlib import Path
from vbp.metrics import clv as clv_fn
from .report import summarize

_TIER_CZ = {"liquid": "likvidní", "neglected": "zanedbaná"}
_BT_CZ = {"soft": "soft", "exchange": "burza", "sharp": "sharp"}
_LEAGUE_CZ = {
    "soccer_brazil_campeonato": "Brazílie Série A",
    "soccer_sweden_superettan": "Superettan",
}


def _pct(x: float) -> str:
    return f"{x * 100:+.2f} %".replace(".", ",")


def _sig(x: float) -> str:
    return f"{x:+.2f}".replace(".", ",")


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
    lines = store.load_lines()
    bets = enrich_bets(store)
    done = [b for b in bets if b["status"] in ("won", "lost")]
    wins = sum(1 for b in done if b["status"] == "won")
    pnl = sum(b["pnl"] for b in done if b["pnl"] is not None)
    kicked = sum(1 for v in lines.values()
                 if (_parse(v.get("kickoff", "")) or now) < now)
    return {
        "generated": now,
        "total_bets": rep["total_bets"],
        "settled_bets": rep["settled_bets"],
        "kicked_off": kicked,
        "by": rep["by"],
        "bets": bets,
        "done": len(done),
        "wins": wins,
        "losses": len(done) - wins,
        "pnl": pnl,
    }


def _bucket_rows_html(by: dict) -> str:
    if not by:
        return ('<tr><td colspan="6" class="muted">Zatím žádná settled sázka - '
                'čekáme na první dohrané zápasy.</td></tr>')
    out = []
    for (bt, tier), g in sorted(by.items()):
        lo, hi = g["clv_ci"]
        clv_cls = "pos" if g["mean_clv"] > 0 else "neg"
        ci = f"[{_pct(lo)} ; {_pct(hi)}]" if g["n"] > 1 else "jen 1 sázka"
        pill = "soft" if bt == "soft" else "ex"
        out.append(
            f'<tr><td><span class="pill {pill}">{html.escape(_BT_CZ.get(bt, bt))}</span> '
            f'{html.escape(_TIER_CZ.get(tier, tier))}</td>'
            f'<td class="n num">{g["n"]}</td><td class="n num">{g["wins"]}</td>'
            f'<td class="n num {clv_cls}">{_pct(g["mean_clv"])}</td>'
            f'<td class="num">{ci}</td>'
            f'<td class="n num">{_pct(g["roi"])}</td></tr>')
    return "\n".join(out)


_STATUS = {
    "pending": ('<span class="pill wait">čeká</span>', ""),
    "won":     ('<span class="pill win">výhra</span>', "pos"),
    "lost":    ('<span class="pill loss">prohra</span>', "neg"),
}


def _log_rows_html(bets: list[dict]) -> str:
    if not bets:
        return '<tr><td colspan="10" class="muted">Zatím žádné sázky.</td></tr>'
    out = []
    for b in bets:
        badge, _ = _STATUS[b["status"]]
        clv = _pct(b["clv"]) if b["clv"] is not None else "-"
        clv_cls = "" if b["clv"] is None else ("pos" if b["clv"] > 0 else "neg")
        pnl = _sig(b["pnl"]) if b["pnl"] is not None else "-"
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
            f'<td>{html.escape(b["book"])} <span class="pill {pill}">{_BT_CZ.get(b["book_type"],b["book_type"])}</span></td>'
            f'<td class="n num">{_pct(b["edge"])}</td>'
            f'<td>{badge}</td>'
            f'<td class="n num {clv_cls}">{clv}</td>'
            f'<td class="n num {pnl_cls}">{pnl}</td>'
            f'</tr>')
    return "\n".join(out)


def render_html(ctx: dict) -> str:
    gen = ctx["generated"].strftime("%d. %m. %Y %H:%M UTC")
    lost_data = max(0, ctx["kicked_off"] - ctx["settled_bets"])
    record = f'{ctx["wins"]}-{ctx["losses"]}' if ctx["done"] else "-"
    pnl_cls = "pos" if ctx["pnl"] > 0 else ("neg" if ctx["pnl"] < 0 else "")
    pnl_str = _sig(ctx["pnl"]) + " j." if ctx["done"] else "-"
    return _TEMPLATE.format(
        gen=gen,
        total_bets=ctx["total_bets"],
        settled_bets=ctx["settled_bets"],
        kicked_off=ctx["kicked_off"],
        lost_data=lost_data,
        record=record,
        pnl_str=pnl_str,
        pnl_cls=pnl_cls,
        bucket_rows=_bucket_rows_html(ctx["by"]),
        log_rows=_log_rows_html(ctx["bets"]),
    )


_TEMPLATE = """<!doctype html>
<html lang="cs">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Value Betting Predictor - live</title>
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
  .sub{{color:var(--ink-soft);font-size:1.02rem;max-width:62ch}}
  header{{border-bottom:1px solid var(--line);padding-bottom:26px;margin-bottom:30px}}
  .stamp{{margin-top:14px;font-size:.82rem;color:var(--ink-soft)}}
  .stamp b{{color:var(--teal)}}
  section{{margin:38px 0}}
  .lead-in{{color:var(--ink-soft);max-width:66ch;margin:.2em 0 1.3em}}
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
  ul.caveats{{list-style:none;padding:0;margin:0;display:grid;gap:10px}}
  ul.caveats li{{padding-left:22px;position:relative;color:var(--ink-soft);font-size:.92rem;max-width:72ch}}
  ul.caveats li::before{{content:"";position:absolute;left:4px;top:.62em;width:6px;height:6px;border-radius:50%;background:var(--rust)}}
  footer{{margin-top:48px;padding-top:20px;border-top:1px solid var(--line);color:var(--ink-soft);font-size:.8rem}}
  code{{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:.86em;background:var(--line);padding:1px 6px;border-radius:5px}}
  .legend{{display:flex;flex-wrap:wrap;gap:16px;margin:14px 0 4px;font-size:.82rem;color:var(--ink-soft)}}
  .legend span b{{color:var(--ink);font-weight:600}}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="eyebrow">Value Betting Predictor &middot; forward paper-trading</div>
    <h1 class="serif">Line-shopping edge, živý deník</h1>
    <p class="sub">Systém bere nejlepší cenu napříč knihami, kde překoná fér Pinnacle o&nbsp;≥3&nbsp;%,
      a paper-tradinguje ji naživo. Tahle stránka se generuje po každém běhu cronu z reálného stavu.</p>
    <p class="stamp">Aktualizováno: <b>{gen}</b> &middot; automaticky z <code>live_state/</code></p>
  </header>

  <section>
    <div class="eyebrow">Souhrn</div>
    <h2 class="serif" style="margin-top:6px">Kde jsme teď</h2>
    <p class="lead-in">Sázka vzniká, když výsledek ještě neexistuje (anti-leak = čas). Bilance a P/L
      jsou jen za dohrané sázky; jednotka = 1× sázka (flat staking). Verdikt padne až u ~100 settled
      soft sázek.</p>
    <div class="tiles">
      <div class="tile"><div class="k num">{total_bets}</div><div class="l">sázek nalezeno celkem</div></div>
      <div class="tile good"><div class="k num">{settled_bets}</div><div class="l">vyhodnoceno (dohráno)</div></div>
      <div class="tile"><div class="k num">{record}</div><div class="l">bilance výhra-prohra</div></div>
      <div class="tile"><div class="k num {pnl_cls}">{pnl_str}</div><div class="l">zisk/ztráta (flat 1j.)</div></div>
    </div>
    <div class="note">Do 4.&nbsp;8.&nbsp;2026 harness kvůli bugu nesettloval; ~{lost_data} dohraných sázek je
      z 3denního okna scores API nenávratně pryč. P/L za tak malé n je čirý šum - řídící metrika je CLV, ne zisk.</div>
  </section>

  <section>
    <div class="eyebrow">Deník</div>
    <h2 class="serif" style="margin-top:6px">Kdy a na co systém vsadil</h2>
    <p class="lead-in">Všechny nalezené sázky, čekající nahoře, dohrané dole. „Nalezeno" = kdy to systém
      zachytil, „výkop" = začátek zápasu. Tip je strana, na kterou padla nejlepší cena.</p>
    <div class="legend">
      <span><b>Edge</b> = o kolik nejlepší cena překonává fér Pinnacle</span>
      <span><b>CLV</b> = výnos kurzu vůči zavírací linii</span>
      <span><b>P/L</b> = zisk v jednotkách (kurz-1 při výhře, -1 při prohře)</span>
    </div>
    <div class="tbl-wrap">
      <table>
        <thead><tr>
          <th>Nalezeno</th><th>Výkop</th><th>Liga</th><th>Zápas</th><th>Tip</th>
          <th class="n">Kurz</th><th>Kniha</th><th class="n">Edge</th><th>Stav</th>
          <th class="n">CLV</th><th class="n">P/L</th>
        </tr></thead>
        <tbody>
{log_rows}
        </tbody>
      </table>
    </div>
  </section>

  <section>
    <div class="eyebrow">Agregace</div>
    <h2 class="serif" style="margin-top:6px">CLV podle typu knihy</h2>
    <p class="lead-in">Verdikt = 90% bootstrap CI soft-knih celé nad nulou. Dokud je n malé, ber to jako
      sanity check, ne signál.</p>
    <div class="tbl-wrap">
      <table>
        <thead><tr><th>Typ &middot; liga</th><th class="n">n</th><th class="n">výhry</th>
          <th class="n">CLV</th><th>95% CI (CLV)</th><th class="n">ROI</th></tr></thead>
        <tbody>
{bucket_rows}
        </tbody>
      </table>
    </div>
  </section>

  <section>
    <div class="eyebrow">Mantinely</div>
    <h2 class="serif" style="margin-top:6px">Co drží verdikt při zemi</h2>
    <ul class="caveats">
      <li>„close" = poslední snapshot před výkopem (proxy, ne přesně -5 min).</li>
      <li>Paper-trading ignoruje slippage a limity/bany knih - potvrzuje <b>existenci</b> edge, ne jeho
        škálovatelnost.</li>
      <li>Bootstrap CI bere sázky jako nezávislé, což nejsou (H/D/A téhož zápasu + tatáž sázka u víc
        knih) - CI je opticky užší, ber verdikt s rezervou.</li>
    </ul>
  </section>

  <footer>
    github.com/radim-tales/value-betting-predictor &middot; The Odds API v4 &middot;
    stránka se generuje automaticky po každém cronu, nesahat ručně.
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
          f"record {ctx['wins']}-{ctx['losses']}, pnl {ctx['pnl']:+.2f})")


if __name__ == "__main__":
    main()
