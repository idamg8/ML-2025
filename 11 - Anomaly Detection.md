> Molto utile e interessante a livello aziendale, nel contesto di Unsupervised Learning (dati non etichettati)

Dataset, si aggiunge nuovo esempio $x_{\text{}}$ -> capire se $x_{test}$ è anomalo rispetto agli altri del dataset

- Prendere sottoinsieme di dati non anomali, analizzare zona in cui si distribuiscono
- Su questa distribuzione definire modello $p(x)$, capire dove si posiziona nuovo esempio $x_{test}$

## Distribuzione gaussiana (normale)
Distribuzione "a campana"

![[Pasted image 20250623123748.png|500]]

![[Pasted image 20250623123932.png|400]]
> - media -> muove curva gaussiana 
> - deviazione std. -> stringe/allarga la campana

![[Pasted image 20250623124057.png|500]]

Se calcolando $p(x_{test})$ esso si posiziona "in coda" alla gaussiana, probabilmente è anomalo.

### Algoritmo
Calcolare distribuzioni gaussiane (definite da media e varianza) per ognuna delle $n$ feature dei dati:
- se esempio "normale" ottengo valore "alto" -> esempio classificato normale solo se tutte le feature sono "normali"
- se esempio anomalo $p(x)$ scende -> valori anomali di alcune feature sono bassi (vanno verso 0), essendo inseriti in una produttoria abbassano il risultato finale
![[Pasted image 20250623124637.png|500]]

**Algoritmo:**
1. scegliere $n$ feature $x_{i}$ indicative di esempi anomali
2. calcolare medie $\mu_{1},\dots,\mu_{n}$ e deviazioni standard $\sigma_{1},\dots,\sigma_{n}$ per ogni feature $$\mu_{j}=\frac{1}{m}\sum_{i=1}^m x_{j}^{(i)} \quad\quad \sigma_{j}^2=\frac{1}{m}\sum_{i=1}^m(x_{j}^{(i)}-\mu_{j})^2$$
3. dato nuovo esempio $x$ calcolare $p(x)$ $$p(x)=\prod_{j=1}^np(x_{j};\mu_{j};\sigma_{j}^2)=\prod_{j=1}^n\frac{1}{\sqrt{ 2\pi \sigma_{j} }}\exp\left( -\frac{(x_{j}-\mu_{j})^2}{2\sigma_{j}^2} \right)$$
4. anomalia se $p(x)<\varepsilon$ valore di soglia

### Esempio
![[11 - Anomaly Detection 2025-06-23 14.01.25.excalidraw]]

***

## Anomaly detection vs. Supervised Learning
**AD)** pochissimi elementi anomali (classificati come y=1, classe positiva), tantissimi elementi normali (y=0, classe negativa) 
**SL)** gran numero di esempi di entrambe le classi

**AD)** tanti tipi diversi di anomalie -> difficile per un algoritmo imparare a riconoscerle dai pochi esempi disponibili; nuove anomalie potrebbero essere completamente diverse dalle precedenti
**SL)** abbastanza elementi della classe positiva per imparare come saranno i prossimi esempi (probabilmente simili a quelli precedenti)

***

## Scegliere feature da usare
Plottare distribuzione delle feature -> se non sono gaussiane 2 strategie:
- cambiare metodo 
- eseguire trasformazioni non lineari per ottenere distribuzioni gaussiane (es. log, radici)

> Alcuni sistemi di anomaly detection funzionano con reti neurali; es:
> Rete addestrata *solo* con immagini di volti -> molto brava in compressione e decompressione solo di immagini di volti -> errore di ricostruzione basso
> Se viene data in input immagine non di un volto (es. lecca-lecca): output sarà "volto di lecca-lecca" -> errore di ricostruzione alto -> anomalia

