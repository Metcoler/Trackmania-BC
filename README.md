# Trackmania-BC

Tento repozitár obsahuje zdrojový kód a súvisiace materiály k bakalárskej / diplomovej práci zameranej na vývoj autonómneho jazdiaceho agenta pre počítačovú hru Trackmania s využitím učenia posilňovaním.


**Školiteľ:** Ing. Alexander Šimko PhD.



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
- A Benchmark Environment for Offline Reinforcement Learning in Racing Games
- A comparison of Different Machine Learning Techniques to Develop the AI of a Virtual Racing Game
- A Driving Model in the Realistic 3D Game Trackmania Using Deep Reinforcement Learning

**Týždeň 6:**  
- Prečítanie si troch článkov z vysokej priority
- Deep Reinforcement Learning in Real-Time Environments
- End-to-end Autonomous Driving - Challenges and Frontiers
- End-to-End Driving in a Realistic Racing Game with Deep Reinforcement Learning

**Týždeň 7:**  
- Prečítanie si troch článkov z vysokej priority
- End-to-End Race Driving with Deep Reinforcement Learning
- Game AI Pro Chapter 39 Representing and Driving a Race Track for AI Controlled Vehicles
- Improving Trackmania Reinforcement Learning Performance - A Comparison of Sophy and Trackmania AI

**Týždeň 8:**  
- Prečítanie si dvoch článkov z vysokej priority
- Neural Network versus Behavior Based Approach in Simulated Car Racing Game
- Simulated Autonomous Driving Using Reinforcement Learning - A Comparative Study on Unity’s ML-Agents Framework

**Týždeň 9:**  
- Veľká noc

**Týždeň 10:**  
- Rozbehanie prototypu z bakalárskej práce

**Týždeň 11:**  
- Tvorba prezentácie
- Konzultácie ohľadom prezentácie

**Týždeň 12:**  
- Kostra práce
- Prezentácia
- Tento github

## Plány do budúcna

- Asynchrónna inferencia rešpektujúca podmienky reálneho času (real-time reinforcement learning)
- Prechod z Pythonu na C# pre dosiahnutie skutočnej asynchrónnosti (obe vlákna budú skutočne paralelné)
- Automatizovaný tréning s logovaním metrík a priebežným vyhodnocovaním výkonu agenta
- Zadanie celej trate agentovi vopred – nespoliehať sa len na lokálnu (lidarovú) informáciu
  - → v súlade s tým, ako fungujú profesionálni pretekári (poznajú trať dopredu)
- Predtréning modelu na jednoduchej, plne synchrónnej hre na získanie základného jazdného správania

