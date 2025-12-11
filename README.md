# Autonómne jazdiaci agent pre hru Trackmania – Diplomová práca
**Školiteľ:** Ing. Alexander Šimko, PhD.

Tento repozitár obsahuje zdrojový kód a súvisiace materiály k bakalárskej a diplomovej práci zameranej na vývoj autonómneho jazdiaceho agenta pre počítačovú hru Trackmania.  

Projekt nadväzuje na bakalársku prácu, v ktorej bol navrhnutý a implementovaný agent využívajúci učenie posilňovaním (RL). V rámci diplomovej práce vzniká nová vetva založená na evolučných algoritmoch a neuroevolúcii, ktorá umožňuje porovnať RL prístup s evolučným učením neurónových sietí.

---

## Prehľad

Cieľom projektu je vytvoriť agenta schopného samostatne prechádzať trate v hre Trackmania bez zásahu človeka. Agent sa učí pomocou experimentovania a spätnej väzby z prostredia, pričom dôraz je kladený na:

- reálne časové obmedzenia (interakcia s bežiacou hrou v reálnom čase),
- reprezentáciu trate a stavového priestoru,
- návrh odmeny / hodnotenia jazdy,
- architektúru učenia (RL vs. evolučné algoritmy).

Repozitár aktuálne obsahuje **dve hlavné vetvy prístupu**:

1. **Agent s učením posilňovaním (RL, PPO)**  
   - neurónová sieť sa trénuje pomocou algoritmu Proximal Policy Optimization (PPO),
   - tréning prebieha v simulovanom prostredí nad mapou Trackmanie,
   - politika sa učí end-to-end z pozorovaní na akcie (plyn, brzda, zatáčanie).

2. **Evolučný agent (neuroevolúcia)**  
   - populácia jedincov reprezentuje politiky v podobe neurónových sietí,
   - genetický algoritmus evolvuje váhy siete na základe kvality jazdy (multi-metrické hodnotenie),
   - prostredie je spúšťané sekvenčne pre každého jedinca, výsledkom je „fitness“ a štatistiky jazdy,
   - experimentuje sa s viacnásobnými metrikami (progress, čas, prejdená vzdialenosť, kolízie…).

---

## Štruktúra projektu

### Hlavné moduly (spoločné pre RL aj GA)

- `Actor.py` – logika agenta a výber akcií (rozhranie medzi politikou a prostredím)
- `Car.py` – simulácia fyziky vozidla a jeho pohybu
- `Driver.py` – riadi interakciu agenta s prostredím (napojenie na Trackmaniu)
- `Enviroment.py` – prostredie/obálka Trackmanie pre učenie a testovanie agenta
- `Map.py` – spracovanie a reprezentácia trate (bloky, checkpointy, geometria)
- `Vizualizer.py` – vizualizácia výsledkov a správania agenta
- `Plugins/get_data_driver` – skripty pre zber a spracovanie dát z hry
- `Maps/` – použité mapy tratí
- `Meshes/` – súbory s geometriou trate
- `logs/` – logy z tréningov a experimentov

### RL (Reinforcement Learning) časť

- `Training.py` – tréningový proces pre RL agenta (PPO a súvisiaca RL slučka)
- `ppo_racing_game.zip` – pôvodná implementácia PPO (Proximal Policy Optimization) pre racing hru
- `agent_driver.mp4` – ukážka jazdy RL agenta

### Evolučný / GA (neuroevolúcia) agent

- `Individual.py` – reprezentácia jedinca:
  - chromozóm = parametre (váhy, biasy) neurónovej siete,
  - pomocné metódy na hodnotenie, porovnávanie a prácu s multi-metrickým hodnotením.
- `EvolutionPolicy.py` – politika agenta založená na neurónovej sieti:
  - mapuje stavový vektor (senzorické vstupy) na akcie,
  - používa sa pri evaluácii jednotlivcov v prostredí.
- `EvolutionTrainer.py` – genetický algoritmus a tréningová slučka:
  - inicializácia populácie,
  - vyhodnocovanie jedincov v prostredí Trackmanie,
  - selekcia, kríženie, mutácia, elitizmus,
  - logovanie metrík (čas, progress, distance, stavy epizód).
- `GADriver.py` – nástroje na spúšťanie a prehrávanie jazdy jedincov:
  - načítanie uložených generácií,
  - vizuálna demonstrácia najlepších agentov.

---

## Denník práce

**Týždeň 1 (22. 9. – 28. 9. 2025):**  
- Začiatok semestra

**Týždeň 2 (29. 9. – 5. 10. 2025):**  
- Prvá prednáška

**Týždeň 3 (6. 10. – 12. 10. 2025):**  
- Úvodné stretnutie so školiteľom

**Týždeň 4 (13. 10. – 19. 10. 2025):**  
- Konzultovanie reinforcment learning vs. genetický algoritmus

**Týždeň 5 (20. 10. – 26. 10. 2025):**  
- Programovanie genetického algoritmu - základ

**Týždeň 6 (27. 10. – 2. 11. 2025):**  
- Programovanie genetického algoritmu - funkčný prototyp

**Týždeň 7 (3. 11. – 9. 11. 2025):**  
- Multikriteriálne hodnotenie jedincov

**Týždeň 8 (10. 11. – 16. 11. 2025):**  
- Trénovanie agenta

**Týždeň 9 (17. 11. – 23. 11. 2025):**  
- Zrušené stretnutie

**Týždeň 10 (24. 11. – 30. 11. 2025):**  
- Trénovanie agenta

**Týždeň 11 (1. 12. – 7. 12. 2025):**  
- Písanie práce

**Týždeň 12 (8. 12. – 14. 12. 2025):**  
- Pŕiprava prezentácie, písanie práce

---

## Plány do budúcna

- S použitím miltikriteriálneho trénovania natrénovať čo najlepšieho jedinca.
- Napísanie diplomovej práce
