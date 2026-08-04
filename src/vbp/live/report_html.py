from __future__ import annotations
"""Vygeneruje statický HTML dashboard z live_state (pro GitHub Pages).

Běží v CI po každém cronu: `python -m vbp.live.report_html docs/index.html`.
Čísla bere ze stejného summarize() jako textový report, takže Pages i terminál
ukazují totéž. Stránka je self-contained (žádné externí zdroje).
"""
import html
from datetime import datetime, timezone
from pathlib import Path
from .report import summarize, _settled_bet_rows

_TIER_CZ = {"liquid": "likvidní", "neglected": "zanedbaná"}
_BT_CZ = {"soft": "soft", "exchange": "burza", "sharp": "sharp"}


def _pct(x: float) -> str:
    return f"{x * 100:+.2f} %".replace("+", "+").replace(".", ",")


def _kicked_off(lines: dict, now: datetime) -> int:
    n = 0
    for v in lines.values():
        try:
            if datetime.fromisoformat(v["kickoff"].replace("Z", "+00:00")) < now:
                n += 1
        except (KeyError, ValueError, AttributeError):
            pass
    return n


def build_context(store, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    rep = summarize(store)
    lines = store.load_lines()
    rows = _settled_bet_rows(store)
    return {
        "generated": now,
        "settled_bets": rep["settled_bets"],
        "total_bets": rep["total_bets"],
        "total_lines": len(lines),
        "kicked_off": _kicked_off(lines, now),
        "by": rep["by"],
        "rows": rows,
    }


def _bucket_rows_html(by: dict) -> str:
    if not by:
        return ('<tr><td colspan="6" style="color:var(--muted)">Zatím žádná settled sázka '
                '- čekáme na první dohrané zápasy.</td></tr>')
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


def _settled_list_html(rows: list[dict]) -> str:
    if not rows:
        return ('<tr><td colspan="6" style="color:var(--muted)">Zatím nic vyhodnoceného.</td></tr>')
    out = []
    for r in sorted(rows, key=lambda x: x["clv"], reverse=True):
        clv_cls = "pos" if r["clv"] > 0 else "neg"
        won = "výhra" if r["won"] else "prohra"
        won_cls = "pos" if r["won"] else "muted"
        out.append(
            f'<tr><td>{html.escape(_TIER_CZ.get(r["league_tier"], r["league_tier"]))}</td>'
            f'<td>{html.escape(r["home"])} – {html.escape(r["away"])}</td>'
            f'<td>{html.escape(r["outcome"])}</td>'
            f'<td class="n num">{r["price"]:.2f}</td>'
            f'<td>{html.escape(r["book"])}</td>'
            f'<td class="n num {clv_cls}">{_pct(r["clv"])}</td></tr>')
    return "\n".join(out)


def render_html(ctx: dict) -> str:
    gen = ctx["generated"].strftime("%d. %m. %Y %H:%M UTC")
    lost = max(0, ctx["kicked_off"] - ctx["settled_bets"])
    return _TEMPLATE.format(
        gen=gen,
        total_bets=ctx["total_bets"],
        settled_bets=ctx["settled_bets"],
        kicked_off=ctx["kicked_off"],
        lost=lost,
        bucket_rows=_bucket_rows_html(ctx["by"]),
        settled_rows=_settled_list_html(ctx["rows"]),
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
    --rust-soft:#f0d9d1;--teal-soft:#d3e6e3;--ochre-soft:#f2e4c8;
    --shadow:0 1px 2px rgba(60,48,30,.06),0 8px 24px rgba(60,48,30,.05);
  }}
  @media (prefers-color-scheme:dark){{
    :root{{--bg:#1a1815;--panel:#221f1a;--ink:#ece5d8;--ink-soft:#a49a89;
    --line:#332e26;--line-strong:#453e33;--rust:#e06b4a;--teal:#57a39c;--ochre:#dda63f;
    --rust-soft:#3a271f;--teal-soft:#1f322f;--ochre-soft:#352a17;
    --shadow:0 1px 2px rgba(0,0,0,.3),0 8px 24px rgba(0,0,0,.35);}}
  }}
  *{{box-sizing:border-box}}
  body{{margin:0;background:var(--bg);color:var(--ink);
    font-family:system-ui,-apple-system,"Segoe UI",sans-serif;line-height:1.55;-webkit-font-smoothing:antialiased}}
  .wrap{{max-width:940px;margin:0 auto;padding:clamp(20px,5vw,56px) clamp(16px,4vw,32px) 80px}}
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
  .lead-in{{color:var(--ink-soft);max-width:64ch;margin:.2em 0 1.3em}}
  .tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:14px}}
  .tile{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px;box-shadow:var(--shadow)}}
  .tile .k{{font-size:2.1rem;line-height:1;font-weight:600}}
  .tile .l{{font-size:.83rem;color:var(--ink-soft);margin-top:7px}}
  .tile.good .k{{color:var(--teal)}}
  .tile.bad .k{{color:var(--rust)}}
  .tbl-wrap{{overflow-x:auto;border:1px solid var(--line);border-radius:12px;box-shadow:var(--shadow)}}
  table{{border-collapse:collapse;width:100%;background:var(--panel);font-size:.9rem}}
  th,td{{padding:10px 14px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}}
  thead th{{font-size:.7rem;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-soft);font-weight:600}}
  tbody tr:last-child td{{border-bottom:none}}
  td.n{{text-align:right}}
  .pos{{color:var(--teal);font-weight:600}}
  .neg{{color:var(--rust);font-weight:600}}
  .muted{{color:var(--ink-soft)}}
  .pill{{display:inline-block;font-size:.72rem;padding:2px 9px;border-radius:20px;font-weight:600}}
  .pill.soft{{background:var(--teal-soft);color:var(--teal)}}
  .pill.ex{{background:var(--ochre-soft);color:var(--ochre)}}
  .note{{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--line-strong);
    border-radius:8px;padding:14px 18px;font-size:.9rem;color:var(--ink-soft);margin-top:12px}}
  .note b{{color:var(--ink)}}
  ul.caveats{{list-style:none;padding:0;margin:0;display:grid;gap:10px}}
  ul.caveats li{{padding-left:22px;position:relative;color:var(--ink-soft);font-size:.92rem;max-width:70ch}}
  ul.caveats li::before{{content:"";position:absolute;left:4px;top:.62em;width:6px;height:6px;border-radius:50%;background:var(--rust)}}
  footer{{margin-top:48px;padding-top:20px;border-top:1px solid var(--line);color:var(--ink-soft);font-size:.8rem}}
  code{{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:.86em;background:var(--line);padding:1px 6px;border-radius:5px}}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="eyebrow">Value Betting Predictor &middot; forward paper-trading</div>
    <h1 class="serif">Line-shopping edge, live test</h1>
    <p class="sub">Realtime validace jediného přeživšího edge z výzkumu: brát nejlepší cenu napříč
      knihami, kde překoná fér Pinnacle o&nbsp;≥3&nbsp;%. Běží na GitHub Actions cronu, tahle stránka
      se generuje po každém běhu.</p>
    <p class="stamp">Aktualizováno: <b>{gen}</b> &middot; automaticky z <code>live_state/</code></p>
  </header>

  <section>
    <div class="eyebrow">Stav sběru</div>
    <h2 class="serif" style="margin-top:6px">Kde jsme teď</h2>
    <p class="lead-in">Settled = zápas dohrán a výsledek dopočítán. Verdikt padne, až bude ~100 settled
      soft sázek.</p>
    <div class="tiles">
      <div class="tile"><div class="k num">{total_bets}</div><div class="l">nasbíraných sázek celkem</div></div>
      <div class="tile good"><div class="k num">{settled_bets}</div><div class="l">settled (vyhodnoceno)</div></div>
      <div class="tile"><div class="k num">{kicked_off}</div><div class="l">zápasů po výkopu</div></div>
      <div class="tile"><div class="k num">~100</div><div class="l">settled soft sázek pro verdikt</div></div>
    </div>
    <div class="note">Do 4.&nbsp;8.&nbsp;2026 harness kvůli bugu nesettloval; ~{lost} dohraných sázek je
      z 3denního okna scores API nenávratně pryč. Od té doby se settluje 1×/den spolehlivě.</div>
  </section>

  <section>
    <div class="eyebrow">Zatím naměřeno</div>
    <h2 class="serif" style="margin-top:6px">CLV podle typu knihy</h2>
    <p class="lead-in">Metrika = CLV, výnos vůči zavírací (fér) linii. Verdikt = 90% CI soft-knih celé
      nad nulou. Dokud je n malé, ber to jako sanity check, ne signál.</p>
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
    <div class="eyebrow">Detail</div>
    <h2 class="serif" style="margin-top:6px">Vyhodnocené sázky</h2>
    <div class="tbl-wrap">
      <table>
        <thead><tr><th>Liga</th><th>Zápas</th><th>Tip</th><th class="n">Kurz</th>
          <th>Kniha</th><th class="n">CLV</th></tr></thead>
        <tbody>
{settled_rows}
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
    github.com/radim-tales/value-betting-predictor (private) &middot; The Odds API v4 &middot;
    stránka se generuje automaticky, nesahat ručně.
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
    print(f"[report_html] wrote {out} ({ctx['settled_bets']}/{ctx['total_bets']} settled)")


if __name__ == "__main__":
    main()
