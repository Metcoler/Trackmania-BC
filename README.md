# Trackmania-BC

Tento repozitár obsahuje zdrojový kód a súvisiace materiály k bakalárskej / diplomovej práci zameranej na vývoj autonómneho jazdiaceho agenta pre počítačovú hru Trackmania s využitím učenia posilňovaním.

## Prehľad

Cieľom projektu je vytvoriť agenta schopného samostatne prechádzať trate v hre Trackmania bez zásahu človeka. Agent sa učí pomocou experimentovania a spätnej väzby z prostredia, pričom dôraz je kladený na reálne časové obmedzenia, reprezentáciu trate a architektúru učenia.

## Štruktúra projektu

- `Actor.py` – logika agenta a výber akcií
- `Car.py` – simulácia fyziky vozidla a jeho pohybu
- `Driver.py` – riadi interakciu agenta s prostredím
- `Enviroment.py` – prostredie Trackmanie pre učenie
- `Map.py` – spracovanie a reprezentácia trate
- `Training.py` – tréningový proces a RL slučka
- `Vizualizer.py` – vizualizácia výsledkov a chovania agenta
- `Plugins/get_data_driver` – skripty pre zber a spracovanie dát
- `Maps/` – použité mapy trate
- `Meshes/` – súbory s geometriou trate
- `logs/` – logy z tréningov
- `agent_driver.mp4` – ukážka jazdy agenta
- `ppo_racing_game.zip` – implementácia PPO (Proximal Policy Optimization)
- `Diplomová práca - kostra.pdf` – návrh štruktúry diplomovej práce

## Denník práce

**Týždeň 1:**  
- Prvá prednáška

**Týždeň 2:**  
- Konzultácie zadania práce

**Týždeň 3:**  
- Zhromažďovanie zdrojov

**Týždeň 4:**  
- Triedenie zdrojov na vysokú, strednú, nízku prioritu 

**Týždeň 5:**  
- Prečítanie si troch článkov z vysokej priority 

**Týždeň 6:**  
- Prečítanie si troch článkov z vysokej priority 

**Týždeň 7:**  
- Prečítanie si troch článkov z vysokej priority 

**Týždeň 8:**  
- Prečítanie si troch článkov z vysokej priority

**Týždeň 9:**  
- Veľká noc

**Týždeň 10:**  
- Rozbehanie prototypu z bakalárskej práce

**Týždeň 11:**  
- Konzultácie ohľadom prezentácie

