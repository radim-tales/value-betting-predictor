# Value-betting predictor s učícím se playbookem

**Datum:** 2026-07-21
**Stav:** Návrh odsouhlasen, připraven k implementačnímu plánu

## 1. Cíl

Praktický fotbalový prediktor, který se přes vlastní, průběžně přepisovaný textový "playbook" snaží v čase zlepšovat v hledání **value** na trhu 1X2 (domácí / remíza / hosté). Value = agentův odhad pravděpodobnosti výsledku je vyšší než pravděpodobnost implikovaná zavíracím kurzem (po odečtení marže sázkovky).

Projekt je primárně **praktický prediktor**, ne akademický experiment ani hřiště na architekturu. Multiagentní rozpad rolí je prostředek, ne cíl.

### Co znamená úspěch

Měřítko není hrubá úspěšnost tipů (tipovat favority je snadné a bezcenné), ale **ekonomika proti trhu**:

- **ROI** simulovaných sázek (flat stake) na navržené value příležitosti.
- **Hit-rate** jako druhotná metrika.
- Srovnání proti dvěma baselinům: **vždy vsadit favorita** podle kurzu a **samotný trh** (implikované pravděpodobnosti z kurzu).
- **Křivka učení:** srovnání výkonu na začátku vs na konci walk-forward loopu.
- **Held-out test generalizace:** finální playbook se pustí na dosud neviděnou sezonu/ligu **bez dalšího učení**; drží si výkon = důkaz, že se agent naučil něco přenositelného, ne že se jen namemoroval na jednu sadu zápasů.

## 2. Princip validity (nejdůležitější omezení)

**Žádný leak budoucnosti.** Zápasy z historie se sice už odehrály, ale agent se musí chovat, jako by stál před nimi. Proto:

- Agent **nikdy** nedostane přístup k webu ani k vlastním "znalostem" o konkrétních zápasech/týmech, protože by si nevyhnutelně přitáhl budoucnost (výsledek, pozápasovou analýzu, koncovou tabulku).
- Agent dostává **výhradně** balíček pre-match faktů dopočítaný z dat, jejichž datum je **striktně starší** než datum tipovaného zápasu.
- Skutečné výsledky odhaluje až deterministický oracle, a to až po odevzdání tipů daného kola.

Toto rozhodnutí je nadřazené "realističnosti" sběru informací. Bez něj by celé měření bylo bezcenné.

## 3. Rozsah (scope první verze)

**V rozsahu:**

- Trh **1X2** (H/D/A).
- **Jedna** dobře pokrytá evropská liga z `football-data.co.uk` (výsledky + zavírací kurzy zdarma v CSV).
- Walk-forward jednou sezonou pro učení + jedna neviděná sezona/liga pro held-out test.
- Jeden LLM agent ve dvou režimech (tipuj / reflektuj) + deterministický harness.

**Záměrně mimo rozsah (YAGNI, možná pozdější rozšíření):**

- Over/under 2.5 a další trhy, dvojtipy, přesný výsledek.
- Kelly staking (první verze jen flat stake).
- Česká Fortuna Liga a nižší soutěže (chybí čistá historická kurzová data; stretch cíl, až jádro poběží).
- Více lig / sezon najednou, křížová validace.
- Grafické UI.

## 4. Komponenty

Kreativita je koncentrovaná do jediné LLM role; vše, co musí být přesné a nepodplatitelné, je deterministický kód.

### 4.1 Feature builder (deterministický kód)

Ze surových CSV historie ligy postaví ke každému zápasu **pre-match balíček** platný k jeho datu. Obsah balíčku (počítáno jen ze zápasů s dřívějším datem):

- Forma posledních N zápasů (celkově, zvlášť doma / venku).
- Aktuální pozice a body v tabulce.
- Vstřelené/obdržené góly (klouzavě), gólová bilance.
- Série (výher/proher/remíz).
- Vzájemné zápasy obou týmů (H2H) z historie.
- Dny odpočinku od posledního zápasu.
- **Implikované pravděpodobnosti z kurzu** (H/D/A) po odečtení marže (overround), včetně samotného kurzu.

**Tvrdá časová hranice:** do balíčku nesmí vstoupit žádný zápas s datem >= datum tipovaného zápasu. Toto je testovatelný invariant.

### 4.2 Prediktor / reflektor (LLM — jádro, jediná kreativní role)

Jeden agent, jeden sdílený playbook, dva režimy:

- **Tipuj:** dostane pre-match balíček kola + aktuální playbook. Pro každý zápas vrátí tři pravděpodobnosti (H/D/A, součet ~1) a stručné zdůvodnění. Deterministický kód pak označí sázku jako navrženou jen tam, kde agentův odhad překročí **value práh** nad implikovanou pravděpodobností trhu (konkrétní hodnota prahu je parametr, default doladíme v plánu).
- **Reflektuj:** po odhalení výsledků kola dostane přehled (co vyšlo/nevyšlo, které value sázky prošly) a **přepíše playbook**. Playbook má **omezenou délku** → nutí k destilaci ponaučení, ne k hromadění balastu. Agent si v něm smí vést i vlastní poznámky, jak mu které přístupy vycházejí.

**Playbook** = jeden verzovaný textový/markdown dokument. Jediné, co přetrvává mezi koly. Je to motor učení (model se netrénuje do vah).

### 4.3 Oracle / rozhodčí (deterministický kód)

Po odevzdání tipů kola odhalí skutečné výsledky a spočítá:

- Value a zásah/nezásah každé navržené sázky.
- ROI (flat stake), hit-rate, počet value sázek, průměrný kurz vsazených sázek.
- Log každé sázky (zápas, tip, kurz, odhad pravděpodobnosti, výsledek, zisk/ztráta).

Nikdy nepoužívá LLM — musí být přesné a reprodukovatelné.

### 4.4 Orchestrátor / loop (deterministický kód)

Walk-forward strojem přes kola sezony:

```
pro každé kolo R v sezoně (v časovém pořadí):
    balíček = feature_builder(historie < datum kola R)
    tipy    = LLM.tipuj(balíček, playbook)
    sázky   = value_filter(tipy, kurzy kola R)
    výsledek = oracle(sázky, skutečné výsledky kola R)   # záznam metrik
    playbook = LLM.reflektuj(playbook, výsledek)          # nová verze playbooku
uložit metriky + verzi playbooku po každém kole
```

Po každém kole ukládá metriky i **snapshot playbooku**, aby šla vidět evoluce strategie. Celý běh musí být reprodukovatelný ze seedu + vstupních dat + logů.

## 5. Data flow (jedno kolo)

```
historie (< datum kola) ─▶ feature builder ─▶ pre-match balíček
                                                    │
              playbook(t) ─────────────────────────▶ LLM: tipuj ─▶ tipy (H/D/A pravděpodobnosti)
                                                                          │
                                                    value_filter ◀────────┘  (+ kurzy kola)
                                                          │
                                                    navržené sázky
                                                          │
skutečné výsledky kola ─▶ oracle ─▶ ROI / hit-rate / log ◀┘
                              │
                              └─▶ LLM: reflektuj ─▶ playbook(t+1)
```

## 6. Vyhodnocení

- **Primární:** ROI navržených sázek vs baseline "vždy favorit" a vs trh.
- **Sekundární:** hit-rate, počet a objem value sázek, průměrný vsazený kurz.
- **Křivka učení:** klouzavé ROI / hit-rate; srovnání první vs poslední třetiny loopu.
- **Held-out test:** finální (zamrzlý) playbook se pustí na neviděnou sezonu/ligu bez další reflexe; porovná se výkon s learning fází i s baseliny na téže held-out sadě.

## 7. Technologie

- **Python** (pandas na zpracování dat, čistá pipeline bez frameworkových závislostí navíc).
- **Data:** `football-data.co.uk` CSV (výsledky + zavírací kurzy 1X2). Konkrétní liga a sezony se vyberou v implementačním plánu (kritérium: úplnost kurzů a dost kol).
- **LLM:** Claude. Konkrétní model a odhad nákladů se doladí v implementačním plánu.
- **Perzistence:** playbook = jeden verzovaný markdown soubor; metriky a logy sázek do strukturovaných souborů (CSV/JSON). Reprodukovatelnost přes seed.

## 8. Vědomá rizika a omezení

- **Porazit zavírací kurz je extrémně těžké.** Realistický výsledek první verze může být mírné mínus. I "nepropadl výrazně proti trhu" je legitimní pozitivní zjištění.
- **Spurious lekce.** LLM reflexe může vygenerovat falešné vzorce ("tým v červeném vyhrává"). Omezená délka playbooku a held-out test jsou pojistky, ne záruka.
- **Variance.** Jedna sezona = málo zápasů; závěry ber s rezervou. Rozšíření o další sezony je připravená cesta, ne přepis.
- **Náklady LLM.** Každé kolo = 2 volání LLM (tipuj + reflektuj) × počet kol × počet běhů; odhad a strop se řeší v plánu.

## 9. Otevřené body do implementačního plánu

- Výběr konkrétní ligy a learning/held-out sezon.
- Default hodnota value prahu a velikost flat stake.
- Přesná délková hranice playbooku a formát pre-match balíčku předávaného do promptu.
- Volba konkrétního modelu Claude + odhad nákladů celého běhu.
- Formát logů a jednoduchý report/shrnutí na konci běhu.
