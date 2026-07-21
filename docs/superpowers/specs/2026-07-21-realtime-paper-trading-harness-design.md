# Real-time paper-trading validační harness (v1)

**Datum:** 2026-07-21
**Stav:** Návrh odsouhlasen, připraven k implementačnímu plánu
**Kontext:** Navazuje na value-betting-predictor. Backtest (Plán A) + LLM vrstva (Plán B) dokázaly, že predikcí z historických features Pinnacle neporazíš. Čtyři analýzy + PoC ukázaly jediný reálný, statisticky významný edge: **line-shopping / value proti měkkým knihám** (na historii CLV +3,65 %, 95% CI [+2,2 %; +5,1 %]). Ten ale nejde čistě backtestovat na denních CSV - proto realtime paper-trading.

## 1. Cíl

Poctivě **dopředu** změřit, jestli line-shopping value edge (kladné CLV proti Pinnacle) drží v realtime. **Žádné reálné sázky, žádný bankroll, žádné doporučování** - čistě měření. Odpovídá na otázku:

> Existuje ten edge dopředu, a **kde** žije - u soft-bookmakerů vs na burzách, v likvidní vs zanedbané lize?

Toto je **validační nástroj**, ne obchodní systém. Bankroll/Kelly/doporučování je záměrně mimo v1 (viz §9).

## 2. Co znamená úspěch

Primární metrika = **běžící mean CLV s bootstrap konfidenčním intervalem**, rozdělená podle:

- **typu knihy:** sharp / soft-bookmaker / burza (exchange),
- **ligy:** likvidní / zanedbaná.

**Edge potvrzen**, když **CLV soft-bookmakerů má 90% bootstrap CI nad nulou** přes dostatečný počet usazených sázek (konkrétní práh N a doba běhu se dolní v plánu; řádově stovky sázek / týdny).

Sekundárně: realizované ROI a objem (počet value sázek za týden a ligu). ROI je diagnostika, ne verdikt - na malém N je šum (dokázáno v analýzách).

Rozpad podle typu knihy je zásadní: PoC ukázal value na **matchbook (burza)**, což může být strukturální levnost (komise místo marže), ne mispricing. Oddělením soft-bookmaker vs burza měříme skutečnou otázku odděleně od strukturálního efektu.

## 3. Architektura

Stateless Python skript spouštěný **GitHub Actions cronem 3× denně** (pevné časy), headless, zdarma. Stav se persistuje **commitem JSON/JSONL logů zpět do repa** (git-friendly, žádná externí databáze; jeden cron běh naráz → žádné souběžné zápisy).

Jeden běh má tři odpovědnosti:

1. **Poll** - stáhne kurzy sledovaných lig (1 likvidní + 1 zanedbaná), přes `vbp.devig` spočítá Pinnacle-fair pravděpodobnost (pravda), najde nejlepší cenu napříč všemi knihami, zaloguje **nové** value sázky (dedup) s tagem typu knihy.
2. **Snapshot open/close** - u každého zápasu drží první viděnou Pinnacle linii (open) a průběžně přepisuje poslední před výkopem (close = proxy pro CLV; free tier neumožňuje přesné -5 min).
3. **Settle** - pro doběhlé zápasy stáhne výsledek (`/scores` endpoint The Odds API), označí won/lost, spočítá ROI a CLV.

Kvůli free tieru The Odds API (~16 kreditů/den) je polling řídký: 2 ligy × 3×/den = 6 kreditů/den, s rezervou. Hustší polling / víc lig / přesnější close = placený tier, až kdyby v1 potvrdil edge.

## 4. Komponenty (jednotky s jednou odpovědností)

- **`odds_client`** - tenký klient The Odds API: `fetch_odds(sport, regions)`, `fetch_scores(sport)`, `list_sports()`. Vrací syrový JSON. Loguje spotřebu kreditů z hlaviček.
- **`adapter`** - transformace API eventu na `{book: {"H","D","A"}}` + klasifikace typu knihy (sharp/soft/exchange) přes pevný číselník book keys. Osekané semínko = `realtime_poc.py`.
- **`value`** - detekce value: `devig(Pinnacle)` = pravda, nejlepší cena napříč knihami, EV ≥ práh (default edge 3 %, kurz 1.6-8.0). Vrací value kandidáty s tagem knihy. Reuse `vbp.devig`.
- **`store`** - persistence: čtení/zápis `bets.jsonl` a `lines.json`, dedup sázek, update open/close/result. Čistý JSON I/O, žádná logika edge.
- **`settle`** - párování doběhlých zápasů s výsledky, výpočet won/lost + CLV (přes uložený close proxy) + ROI. Reuse `vbp.metrics`.
- **`report`** - z logů vypíše mean CLV + bootstrap CI × typ knihy × liga, ROI, objem, pokrytí Pinnacle. Spustitelný i lokálně. Reuse `vbp.metrics` (roi, clv, bootstrap_roi_ci).
- **`run`** - orchestruje jeden cron běh: poll → snapshot → settle → commit. Entrypoint pro GitHub Actions.

## 5. Datový model (JSON v repu, adresář `runs/live/`)

- **`bets.jsonl`** - append-only, jeden řádek = jedna virtuální value sázka:
  `{match_id, league, league_tier, home, away, kickoff, outcome, book, book_type, price, edge, pin_fair_open, ts_detected}`
- **`lines.json`** - dict keyed by `match_id`, per zápas:
  `{league, home, away, kickoff, pin_open: {H,D,A}, pin_close: {H,D,A}, result, settled}`

Report se generuje výhradně z těchto dvou souborů. `book_type ∈ {sharp, soft, exchange}` dle pevného číselníku (Pinnacle=sharp; B365/BW/WH/... = soft; Betfair/Matchbook/... = exchange).

## 6. Data flow (jeden cron běh)

```
pro každou sledovanou ligu:
    events = odds_client.fetch_odds(sport, regions)
    pro každý event:
        books = adapter.event_to_books(event)          # {book: H/D/A} + typy
        pokud je Pinnacle v books:
            store.update_line(match_id, pin_open|pin_close dle času vs kickoff)
            pro každý value kandidát z value.find(books):
                store.add_bet(...) pokud ještě není (dedup na match_id+outcome+book)
scores = odds_client.fetch_scores(sport)               # doběhlé zápasy
settle.settle_finished(store, scores)                  # result -> won/lost, CLV, ROI
commit JSON logů zpět do repa
```

Open vs close: pokud `now < kickoff - buffer`, aktualizuj `pin_open` jen když ještě není; vždy přepiš `pin_close` na aktuální (poslední před výkopem se stane finálním close). CLV = `pin_close_fair[outcome] * bet_price - 1`.

## 7. Anti-leak a poctivost (zabudované)

- **Anti-leak = čas.** Value detekuješ z aktuálně viditelných kurzů; výsledek v tu chvíli neexistuje. Žádný leak budoucnosti z principu.
- **Report explicitně uvádí caveaty:** (a) „close" je proxy (poslední snapshot, ne -5 min); (b) paper-trading **ignoruje slippage** (nejlepší cena, kterou jsi viděl, nemusí být dostupná při reálné sázce) a **absenci banů** → harness potvrzuje **existenci** edge, ne jeho **škálovatelnost**.

## 8. Tech & reuse

- **Python 3.11+**, běží na GitHub Actions (cron workflow).
- **The Odds API** (free tier, klíč přes GitHub Actions secret `ODDS_API_KEY`).
- **Reuse z vbp:** `devig` (Shin), `metrics` (roi, clv, bootstrap_roi_ci). Realtime cesta potřebuje jen **numpy + scipy + requests** (ne pandas/sklearn) - lehká.
- **Testy** jedou na **uložených fixture JSON odpovědích API** (žádná síť ani kredit v testech), stejný princip jako FakeLLM v Plánu B. Pokrývají: adapter (transformace + klasifikace knih), value (detekce + práh), store (dedup, open/close update), settle (won/lost, CLV), report (render metrik).

## 9. YAGNI (záměrně mimo v1)

Bankroll / frakční Kelly / staking; doporučování „vsaď tohle"; reálné sázení; over/under a jiné trhy (jen 1X2 / h2h); víc než 2 ligy; web dashboard; scraping / LLM vrstva; placený API tier; přesný -5 min close snapshot. Vše až kdyby v1 potvrdil, že edge dopředu drží.

## 10. Otevřené body do implementačního plánu

- Konkrétní výběr 2 lig (aktuálně aktivních dle `list_sports` + dostatečné Pinnacle pokrytí; PoC ukázal, že Pinnacle posílá pozdě a u některých zápasů chybí).
- Přesné cron časy (3× denně) vzhledem k typickým časům výkopů zvolených lig.
- Práh N usazených sázek a doba běhu pro závěr „edge potvrzen".
- Formát fixture JSON pro testy (osekané reálné API odpovědi).
- Buffer okna pro open vs close přepis.
- Číselník book keys → book_type (sharp/soft/exchange) dle knih, co The Odds API reálně vrací.
