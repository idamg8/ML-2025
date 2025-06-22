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

***

## Hierarchical Agglomerative Clustering
Parte a raggruppare elementi dal basso:
1. per ogni punto creo un cluster
2. ogni cluster cerca l'altro cluster più vicino -> uniti 
3. si ripete punto 2 fino ad ottenere un unico grande cluster

![[Pasted image 20250622173140.png|400]]

