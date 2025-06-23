> Ambito dell'*unsupervised learning*: dati non etichettati da organizzare, dargli struttura (potrebbe non essere utile all'utente, è soggettiva)

![[09 - Clustering 2025-06-22 16.27.35.excalidraw]]

## K-means
Richiede assunzione del numero di raggruppamenti (*cluster*) desiderati
1. assegnazione di punti detti *centroidi*, uno per cluster 
2. analisi di ogni punto, classificati per vicinanza ai centroidi -> si formano i cluster
3. aggiornamento centroidi -> spostati più "centrali" dei raggruppamenti creati
4. di nuovo analisi dei punti in base a distanza dai centroidi (siccome sono stati aggiornati, classificazione dei punti può cambiare)
5. repeat passi 3-4 finché sistema converge -> centroidi stabilizzati, non si spostano più

**Aggiornamento centroidi:** prese coordinate di tutti i punti del cluster, il nuovo centroide avrà come coordinate la media delle coordinate dei punti

### Algoritmo
![[Pasted image 20250622164144.png|400]]

K-means tecnica molto semplice -> implementazioni ottimizzate

Inizializzando centroidi casualmente potrebbero configurarsi in modo non ottimale
![[Pasted image 20250622171807.png|400]]
-> caso 1: punti tutti circa equidistanti dai centroidi 

Metrica per valutare bontà dei cluster: *somma distanze tra punti e rispettivo centroide* (funzione di costo) -> obiettivo: minimizzarla (distanza minore = centroidi più rappresentativi dei cluster)
![[Pasted image 20250622172157.png|300]]

### Algoritmo migliorato: random initialization
![[Pasted image 20250622172458.png|400]]

> **N.B:** casi più complessi (dati distribuiti con continuità) potrebbero non convergere (continui spostamenti dei centroidi) -> si può scegliere numero max. iterazioni

-> **Problematicità:** K-means prende tutti i punti dentro a un certo raggio di distanza dal centroide e li mette in un cluster (crea strutture circolari) -> comportamento non ideale con conformazioni di dati molto allungati o "movimentati"
![[kmeansNoisyMoonsScatter-1.png|250]]

***

## Hierarchical Agglomerative Clustering
Parte a raggruppare elementi dal basso:
1. per ogni punto creo un cluster
2. ogni cluster cerca l'altro *cluster più vicino* -> uniti 
3. si ripete punto 2 fino ad ottenere un unico grande cluster

![[Pasted image 20250622173140.png|400]]

"Tagliando" dendrogramma a varie altezze aumenta man mano numero di cluster 
- decidere numero cluster desiderato, tagliare all'altezza corrispondente
- se non si sa numero di cluster desiderato: capire quando proporzione delle altezze che servono per unire nuovi cluster è molto maggiore che per unire i cluster precedenti (rapporto tra nuovo "passo" da compiere rispetto ai passi precedenti)
![[09 - Clustering 2025-06-22 17.45.09.excalidraw|250]]

### Misura distanza tra cluster
- **single link:** distanza minima tra un elemento in un cluster e un elemento nell'altro cluster
- **complete link:** distanza massima tra un elemento in un cluster e un elemento nell'altro cluster
- **average:** distanza media tra tutti gli elementi in un cluster e tutti quelli dell'altro
![[Pasted image 20250622175040.png|200]]
-> il sistema unisce sempre i cluster più **vicini** a prescindere dalla distanza scelta

***

## DBSCAN Algorithm
> Density-Based Clustering Methods

- Lavora con dati che hanno forme arbitrare (distribuzione di punti anche non circolari)
- Gestisce rumore
- Una sola scansione dei punti
- Necessita parametri di densità

-> si imposta *Epsilon* (diametro della circonferenza) + **densità** *MinPoints* (minimo numero di punti affinché un punto diventi core point)

3 possibili ruoli dei punti:
- *core point:* se dentro alla circonferenza ha almeno MinPoints -> core point vicini (a distanza ≤ Eps) messi nello stesso cluster
- *noise point:* punto "isolato", nella circonferenza ci sono meno di MinPoints -> scartato
- *border point:* si trova "sul bordo", ci sono meno di MinPoints nella sua circonferenza ma uno di questi punti è un core point

![[09 - Clustering 2025-06-23 09.05.56.excalidraw]]

### Pseudo-algoritmo
- scelgo punto $p$
- prendo tutti i punti a distanza Eps da $p$ e li conto
- se $p$ noise point -> label: noise
- se $p$ border/core point -> formo un cluster
- visito tutti i vicini del core point:
	  - se non sono stati ancora assegnati a un cluster, li assegno al cluster appena creato
	  - se a loro volta sono core points, visiterò anche i loro vicini
- continuo finché non ci sono più core points a distanza Eps dal cluster
- scelgo altro punto $p'$ non ancora visitato e ripeto procedura

> **N.B:** non è necessario definire numero di cluster desiderati

-> metodologia noise resistant: non crea singoli cluster (potenzialmente tantissimi) per ogni punto di noise

**Problematicità:** se *densità* dei punti del dataset è *variabile*, avrei bisogno di Eps diversi -> difficile tarare parametro Eps -> soluzione pratica: normalizzare, portare i dati vicino all'origine con std. dev. = 1

> [!Osservazione]
> Non sempre i cluster trovati dagli algoritmi (struttura trovata nei dati) sono quelli che ci si aspettava/desiderati.
> Es: clusterizzazione di volti -> invece di suddividere per persone, potrebbe suddividere per orientazione del volto oppure presenza di cappellino in testa/occhiali, etc. 
> Nell'esercitazione: vengono dati in input solo pixel senza altre indicazioni, si basa su colore e posizione dei pixel 



