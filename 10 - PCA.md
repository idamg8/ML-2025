## Dimensionality Reduction
Es: misurazioni fatte in due sistemi di misura diversi e con sensibilità diverse o con errori di approssimazione (dati ridondanti) -> dati delle misurazioni su spazio 2D, vogliamo *ridurre a 1D* (trovando retta che meglio approssima i dati) e calcolare *errore* di rappresentazione -> ottengo punti su retta

![[Pasted image 20250623103022.png|400]]

-> strategia applicabile anche da 3D a 2D (dati proiettati su piano che meglio li rappresenta) o dimensionalità ancora più alte

Ridurre dimensionalità (es. a 2D) utile per *plottare dati e trovare similarità fra punti* (es. dati rappresentati da 100 feature ridotte a 2, plottati, punti vicini rappresentano oggetti simili)

> Condizionare (contesto AI generativa): dire al sistema di generare un nuovo punto in una certa area del feature space (es: suddiviso in zona tavoli, sedie, lampade, ecc. -> per generare lampada: dire al sistema di trovare punto che finisca in quell'area del feature space)
> ![[Pasted image 20250623104244.png|300]]

## Principal Component Analysis (PCA)
Obiettivo: ridurre più possibile errore di rappresentazione dei dati proiettati sullo spazio a dimensionalità inferiore
-> da n dimensioni a k dimensioni: trovare k vettori su cui proiettare i dati per minimizzare errore di proiezione

![[10 - PCA 2025-06-23 10.47.40.excalidraw|400]]
> **N.B:** PCA $\neq$ Linear Regression -> non trova retta che meglio approssima dati, ma retta con minor errore

### Preprocessing
-> è raccomandato effettuare normalizzazione
![[Pasted image 20250623105126.png|400]]

### Algoritmo
Si crea *matrice di covarianza*: mette a confronto i dati, mostra quali punti sono simili ad altri (rappresenta variazione di ogni variabile rispetto alle altre)
-> analizzata con svd (singular value decomposition) che fattorizza la matrice in prodotto di 3 matrici:
1. U = **autovettori** -> importanti: sfruttati per identificare direzioni (vettori) più importanti per rappresentare i dati 
2. S = autovalori
3. V = autovettori "rigirati"

![[Pasted image 20250623105839.png|400]]

In U sono contenuti $n$ vettori con le direzioni principali utili per rappresentare i dati -> si prendono $k$ vettori più importanti e si definisce matrice Ureduce (nuova base) -> si prendono tutti i dati, li si moltiplica per la base di vettori in Ureduce -> si ottiene proiezione dei dati sulla nuova base
![[Pasted image 20250623110139.png|400]]

### Ricostruzione da rappresentazione compressa
![[Pasted image 20250623110410.png|400]]
> **N.B:** tutti i punti su stessa linea di ortogonalità rispetto alla retta, avranno la stessa proiezione (condensati su un unico punto)

***

## Scegliere numero di componenti principali
Come scegliere numero di dimensioni del nuovo spazio?

-> Calcolare errore tra x e approssimazione di x (per tutti i dati) diviso fattore di "grandezza" delle feature -> buon $k$ porta ad avere *varianza sotto una certa soglia* (≤ 0.01 - 1%)

-> Strategia più veloce: sfruttare autovalori di svd 

![[10 - PCA 2025-06-23 11.09.57.excalidraw]]

**Vantaggi:**
- Se training set grande e con gran quantità di feature, ridimensionare feature space velocizza Supervised Learning (meno feature da tenere in considerazione)
	  - riduzione spazio per memorizzare dati
	  - algoritmo di addestramento velocizzato
- Visualizzazione (2/3 dimensioni -> dati plottabili)

***

## Dimensionality Reduction con Reti Neurali
Simulato collo di bottiglia -> rete parte da dimensionalità alta, poi la porto a dimensionalità minore (layer con meno neuroni) -> utilizzo poi rete inversa (speculare) per ricostruire il dato iniziale

![[Pasted image 20250623112047.png|400]]

- Encoder porta dati da feature space originario a ridotto
- Decoder riporta dati a feature space originario

Se in output non escono dati "uguali" agli input, si aggiornano pesi per minimizzare l'errore di ricostruzione