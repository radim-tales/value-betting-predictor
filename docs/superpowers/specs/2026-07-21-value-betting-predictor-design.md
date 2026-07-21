# Value-betting predictor s učícím se playbookem

**Datum:** 2026-07-21
**Stav:** Návrh odsouhlasen (po oponentuře 3 externích modelů), připraven k implementačnímu plánu

## 1. Cíl

Praktický fotbalový prediktor, který se přes vlastní, průběžně přepisovaný textový "playbook" snaží v čase zlepšovat v hledání **value** na trhu 1X2 (domácí / remíza / hosté). Value = agentův odhad pravděpodobnosti výsledku je vyšší než férová (od marže očištěná) pravděpodobnost implikovaná kurzem.

Projekt je primárně **praktický prediktor**, ne akademický experiment ani hřiště na architekturu. Multiagentní rozpad rolí je prostředek, ne cíl.

### Co znamená úspěch

Primární metrika je **CLV (Closing Line Value)** - sázíme na **otevírací** kurz a měříme, jestli jsme dlouhodobě porazili **zavírací** linii. CLV má řádově nižší varianci než ROI na malém vzorku a je to skutečný ukazatel dovednosti; sedí i na "praktický prediktor" (sázíš dřív, než trh dozraje), místo beznadějného honění zavíracího kurzu.

Doplňkově:

- **ROI** simulovaných sázek (flat stake) **s bootstrap konfidenčním intervalem**; ROI bez CI je dekorace.
- **Kalibrace:** Brier score a log-loss na **všech** tipech (ne jen na value sázkách) + kalibrační křivka. Value betting stojí na kalibraci.
- Srovnání proti **baselinům** (viz §6) - nejen "vždy favorit", ale i silné deterministické modely a noise baseline.
- **Ablace playbooku** (viz §6) - důkaz, že "učení přes text" reálně přidává hodnotu.
- **Held-out verdikt** na předem zamčeném locked-test splitu je jediný závazný výsledek; křivka učení je jen diagnostika, ne důkaz.

## 2. Princip validity (nejdůležitější omezení)

**Žádný leak budoucnosti, ani z dat, ani z paměti modelu.** Oponentura odhalila, že "žádný web" nestačí:

- **Leak paměti modelu (kritický).** Claude byl trénován na skutečných výsledcích, tabulkách a statistikách. Jméno týmu + kalendářní datum = prakticky open-book test na memorované výsledky. **Řešení: anonymizace.** Týmy se agentovi předávají jako `Team_A` / `Team_B` (konzistentně napříč sezonou i v playbooku), do promptu nejde ani kalendářní datum, ani jméno ligy/sezony - jen relativní čas ("kolo N", "N dní odpočinku"). Ověření leaku: A/B běh reálná jména vs anonymizace; skokový propad výkonu = leak byl reálný.
- **Leak z dat.** Agent dostává výhradně balíček pre-match faktů z dat se **striktně starším datem** než tipovaný zápas (`date < target_date`; při více zápasech téhož dne se celý den ignoruje, protože nemáme spolehlivé časy výkopů). Testovatelný invariant.
- **Whitelist sloupců, ne blacklist.** Feature builder smí číst jen: datum, týmy, historické výsledky a pre-match kurzy. Nikdy pozápasové sloupce (střely, karty, poločas...). Leakage unit testy to hlídají v CI.
- **Výsledky** odhaluje až deterministický oracle, po odevzdání tipů kola.

Toto rozhodnutí je nadřazené "realističnosti". Bez něj je měření bezcenné.

## 3. Rozsah (scope první verze)

**V rozsahu:**

- Trh **1X2** (H/D/A).
- **Jedna** dobře pokrytá evropská liga z `football-data.co.uk`, se sezonami, které mají **otevírací i zavírací** kurzy (nutné pro CLV).
- **Anchor model** (deterministický statistický baseline, bez trhu) + **LLM korektor** + deterministický harness.
- **Bloková** reflexe playbooku.
- Experimentální design se splity train / validation / locked-test a s **plnou baterií ablací** (§6).

**Záměrně mimo rozsah (YAGNI, možná pozdější rozšíření):**

- Over/under 2.5 a další trhy, dvojtipy, přesný výsledek.
- Kelly staking (první verze jen flat stake).
- Česká Fortuna Liga a nižší soutěže (chybí čistá historická kurzová data; stretch, až jádro poběží).
- Line shopping / více bookmakerů (fixujeme jeden zdroj kurzu, viz §7).
- Grafické UI.

## 4. Komponenty

Kreativita je koncentrovaná do jediné LLM role; vše, co musí být přesné, je deterministický kód.

### 4.1 Feature builder (deterministický kód)

Ze surových CSV (přes whitelist sloupců) postaví ke každému zápasu **pre-match balíček** platný k jeho datu, počítaný jen ze zápasů s dřívějším datem:

- Forma posledních N zápasů (celkově, zvlášť doma / venku).
- Pozice a body v tabulce (k začátku dne zápasu).
- Klouzavé góly vstřelené/obdržené, gólová bilance.
- Série (výher/proher/remíz).
- Dny odpočinku od posledního zápasu.
- (H2H se do v1 **nezařazuje** - malý vzorek, přetahuje se přes sezony a soupisky, láká k narativnímu overfittingu. Kandidát na pozdější ablačně ověřené rozšíření.)

Feature builder **neposílá kurz do predikčního promptu.** Kurz zpracuje zvlášť pro value filtr: **devig** (očištění marže) metodou, která respektuje favorite-longshot bias - **Shin nebo power method**, ne prostá normalizace `1/kurz`; zvolená metoda se zafixuje a citlivost se loguje.

Anonymizace (Team_A/B, relativní čas) se aplikuje na výstup před předáním LLM.

### 4.2 Anchor model (deterministický kód)

Deterministický statistický model, který z pre-match faktů (**bez trhu**) vyrobí výchozí pravděpodobnosti H/D/A. Kandidáti: **Elo s modelem remíz** nebo multinomiální/ordinální regrese nad stejnými features (konkrétní volba v plánu). Anchor plní tři role najednou:

1. Dává LLM kalibrovaný výchozí bod (agent nemusí střílet absolutní čísla z hlavy → míň halucinací).
2. Je sám o sobě **silným baselinem** k poražení.
3. Drží kurz mimo predikční prompt.

### 4.3 Prediktor / korektor (LLM — jádro, jediná kreativní role)

Jeden agent, jeden sdílený playbook, dva režimy:

- **Koriguj (tipuj):** dostane pre-match balíček (anonymizovaný, bez kurzu) + výchozí pravděpodobnosti z anchor modelu + aktuální playbook. Vrátí **bodovou korekci** anchor pravděpodobností (např. `H: -3%, D: +1%, A: +2%`), ne absolutní odhad z hlavy. **Nejdřív vrátí čísla ve validovaném JSON, teprve pak (volitelně) zdůvodnění** - aby textový příběh netlačil čísla k líbivosti. Value filtr (§4.4) je až za tímto.
- **Reflektuj (blokově):** ne po každém kole, ale po **bloku** (~3-5 kol / min. N vyhodnocených sázek). Dostane **agregovaný report + kalibrační summary** (overconfidence, bias na remízy, výkon po skupinách), ne příběhy jednotlivých zápasů. Přepíše playbook s omezením: každá změna musí uvést metrický důvod.

**Playbook** je **strukturovaný** (ne volný deník), s omezenou délkou i max počtem pravidel:

```
## Priors (stabilní)
## Rules (max N, každé s počítadlem support/oppose)
## Hypotheses (kandidáti, TTL, čekají na potvrzení)
## Banned (zahozené heuristiky)
## Notes (efemérní, mažou se po K blocích)
```

Lifecycle pravidla: nové = kandidát → povýší na potvrzené až po M blocích s čistě pozitivní evidencí → jinak drop. Je to jediné, co přetrvává mezi koly (model se netrénuje do vah).

### 4.4 Value filter (deterministický kód)

Z korigovaných pravděpodobností a férového (devig) kurzu vybere sázky s disciplínou:

- **Minimální edge** (práh v procentních bodech, laděný jen na validaci - viz §6).
- **Max 1 sázka na zápas** = argmax edge (jinak by při špatné kalibraci prošlo H i D).
- **Filtr kurzu** do rozumného rozsahu (např. mimo extrémní favority a longshoty), drop řádků s chybějícím kurzem.
- Volitelně potlačit value v prvních K kolech sezony (nestabilní tabulka).

### 4.5 Oracle / rozhodčí (deterministický kód)

Po odevzdání tipů kola odhalí výsledky a spočítá:

- **CLV** (primární): otevírací vsazený kurz vs zavírací férová linie.
- **ROI** (flat stake) s **bootstrap CI**; zásah/nezásah; počet sázek; průměrný a mediánový kurz.
- **Kalibrace:** Brier, log-loss, kalibrační křivka na všech tipech.
- **Stratifikace:** ROI zvlášť pro H/D/A a po kurzových skupinách; P/L bez top 1-3 výher (kontrola, jestli výsledek nestojí na pár longshotech).
- **Slippage stres test:** ROI při zhoršení kurzu o 1 / 2 / 3 %.
- **Audit log** každé sázky: vstupní balíček, anchor P, prompt hash, raw LLM odpověď, parsované + normalizované P, férová tržní P, value rozhodnutí, výsledek, P/L, verze playbooku.

Žádný LLM. Musí být přesné a reprodukovatelné.

### 4.6 Orchestrátor / loop (deterministický kód)

Walk-forward přes kola sezony, reflexe po blocích:

```
pro každé kolo R v sezoně (chronologicky):
    balíček = feature_builder(historie < datum kola R)         # anonymizovaný, bez kurzu
    anchor_P = anchor_model(balíček)
    korekce = LLM.koriguj(balíček, anchor_P, playbook)          # JSON čísla, pak důvod
    tipy    = anchor_P + korekce
    sázky   = value_filter(tipy, devig(otevírací kurzy R))
    oracle(sázky, výsledky R, zavírací kurzy R)                 # CLV, ROI, kalibrace, log
    pokud konec bloku:
        playbook = LLM.reflektuj(playbook, agregovaný_report_bloku)
uložit metriky + snapshot playbooku po každém bloku
```

Ošetřit failure modes: timeout, malformed JSON, odmítnutí, rate limit → schema validace + retry + v krajním případě skip s logem; orchestrátor **nikdy tiše nedomýšlí** pravděpodobnosti.

## 5. Data flow (jedno kolo)

```
historie (< datum kola, whitelist) ─▶ feature builder ─▶ balíček (anonym., bez kurzu)
                                                              │
                                              anchor model ◀──┤
                                                    │         │
                                              anchor_P ──┐    │
                                  playbook(t) ───────────┼────▶ LLM: koriguj ─▶ korekce (JSON)
                                                         │                          │
                                         tipy = anchor_P + korekce ◀────────────────┘
                                                         │
                              devig(otevírací kurz) ─▶ value_filter ─▶ navržené sázky
                                                                            │
výsledky + zavírací kurz ─▶ oracle ─▶ CLV / ROI±CI / kalibrace / log ◀──────┘

na konci bloku:  agregovaný report ─▶ LLM: reflektuj ─▶ playbook(t+1)
```

## 6. Experimentální design (proti p-hackingu a iluzi učení)

**Splity dat (předem daný, nikdy neměnit po nahlédnutí):**

- **Train/learn** sezona: playbook se učí.
- **Validation** sezona: ladí se value práh, délka playbooku, cadence a prompt.
- **Locked test** sezona: spustí se **jednou**, žádná změna po nahlédnutí. Hlavní test = jiná sezona **téže ligy** (časový přenos); jiná liga jen jako volitelný robustness check.

**Zamčený config před během:** liga, sezony, bookmaker/zdroj kurzu, devig metoda, práh, stake, N formy, délka playbooku, cadence reflexe, model, temperature → do `config.yaml` + hash do reportu. Cokoli později = explicitně "exploratory".

**Baseliny (porazit ty levné, ne jen "vždy favorit"):**

- Tržní implikované pravděpodobnosti (kalibrace / log-loss benchmark).
- **Anchor model samotný** (bez LLM).
- Silný deterministický model (logistická regrese / Elo / Dixon-Coles).
- **Noise baseline:** tržní P + gaussovský šum přes value filtr - když se agent neliší od šumu, nemá edge.
- Naivní: vždy favorit, vždy domácí.

**Ablace playbooku (důkaz, že učení přes text něco dělá):** naučený playbook vs prázdný vs zamrzlý seed vs statický ručně psaný vs **bez reflexe** (statický LLM). Teprve když naučený porazí tyto kontroly na locked testu, dá se mluvit o učení.

**Akceptační kritéria (předem daná, ať se výsledek nedopoví zpětně):** např. na locked testu ROI > 0 i po 1% slippage, bootstrap CI ne extrémně široké, Brier/log-loss lepší než tržní baseline, a min. počet sázek (např. ≥100). Konkrétní hodnoty se zafixují v plánu.

## 7. Vyhodnocení

- **Primární verdikt:** CLV a výkon na **locked testu**, s bootstrap CI, proti baselinům a ablacím.
- **Sekundární:** ROI±CI, kalibrace (Brier/log-loss/křivka), stratifikace H/D/A a kurzové skupiny, slippage.
- **Diagnostika (ne důkaz):** křivka učení (klouzavé metriky, první vs poslední třetina) - je to in-sample adaptace, ne out-of-sample edge.

## 8. Technologie

- **Python** (pandas na data; anchor model přes lehkou statistiku / scikit-learn nebo vlastní Elo).
- **Data:** `football-data.co.uk` CSV; vybraná liga/sezony musí mít otevírací i zavírací kurzy 1X2. **Jeden fixní zdroj kurzu** (např. Pinnacle nebo tržní průměr); nikdy ex-post nejlepší kurz napříč bookmakery.
- **LLM:** Claude. Konkrétní model (case: levnější na korekci-batch, silnější na reflexi) a odhad nákladů se doladí v plánu.
- **Perzistence & reprodukovatelnost:** playbook = verzovaný strukturovaný markdown; metriky/logy CSV/JSON; **kompletní audit log** (viz §4.5). Reprodukovatelnost neznamená znovu volat LLM (Claude není bitově deterministický), ale **přehrát z uložených raw odpovědí**; logovat model ID, temperature, prompt hash, timestamp.

## 9. Vědomá rizika a omezení

- **Porazit trh je extrémně těžké** i s CLV framingem. I "nepropadl proti baselinům" je legitimní zjištění (funkční laboratoř > falešně pozitivní playbook-román).
- **Učení přes text může být iluze** - proto ablace jako tvrdý test, ne jako ozdoba.
- **Spurious lekce a recency bias** - bloková reflexe, evidence counters a lifecycle pravidel to tlumí, nezaručují.
- **Variance i s CLV** - jedna sezona = málo sázek; proto CI a min. N. Rozšíření o další sezony je připravená cesta.
- **Feature set je retail-level** (forma/tabulka) - proti sharp trhu je to slabé; v1 testuje **mechaniku učení playbooku**, ne že porazí Pinnacle z CSV formy. Explicitně to tak pojmenovat v závěrech.
- **Náklady LLM** - korekce (N zápasů batchem v jednom promptu) + bloková reflexe výrazně sníží počet volání oproti "2 volání/kolo"; odhad a strop v plánu.

## 10. Otevřené body do implementačního plánu

- Konkrétní liga a train/validation/locked-test sezony (musí mít open+close kurzy).
- Volba anchor modelu (Elo s modelem remíz vs multinomiální/ordinální regrese).
- Devig metoda (Shin vs power) a fixní zdroj kurzu.
- Default value práh (laděný na validaci), flat stake, K úvodních kol bez sázek, rozsah povolených kurzů.
- Délka a max počet pravidel playbooku, přesná cadence bloku, formát promptu (batch JSON schema).
- Konkrétní model Claude + temperature + odhad nákladů celého běhu (vč. ablací).
- Konkrétní hodnoty akceptačních kritérií.
- Jak přesně value filtr naloží s korekcí, po níž P nesečtou na 1 (normalizovat vs odmítnout).
