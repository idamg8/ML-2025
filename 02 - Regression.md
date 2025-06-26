-> trovare retta che modella, rappresenta i dati 
-> interrogare modello per fare previsioni sui dati, in base ad apprendimento su una base di dati preesistenti

Dare al modello: dati in input e dato che voglio restituisca come output
-> dati in input e desired output devono essere correlati semanticamente (es. dimensione appartamento -> prezzo)

Per fare training: fornisco al modello esempi di training come coppia input, desiderato

### Notazione
- m = numero esempi di training
- x = variabili di input/feature
- y = target/output value
- (x,y) = coppia (input, desiderato), esempio di training
  -> $(x^{(i)},y^{(i)})$: i-esimo esempio di training; i = 1, ...

![[Pasted image 20250617115650.png|500]]

***

# Linear Regression
## Con una variabile
### Ipotesi
$$h_{\theta}(x_{1})=\theta_{0}+\theta_{1}x_{1}$$
$x_{1}$: variabile
$\theta_{0}, \theta_1$: parametri
$\theta=\begin{bmatrix} \theta_{0} \\ \theta_{1} \end{bmatrix}$
$x=\begin{bmatrix} 1 \\ x_{1} \end{bmatrix}$
-> facendo prodotto scalare tra i due vettori otteniamo l'ipotesi sopra: $h_{\theta}=\theta^Tx$

> *Retta parametrica:* si muove rispetto ai parametri (quando li cambiamo, definiamo una nuova retta)

**Obiettivo**: trovare parametri migliori per approssimare i dati 
-> retta più vicina ai dati: mediamente errore tra distanza punto della retta-esempio di training sia minore possibile

### Problema minimizzazione
Dati dei valori, calcolo la funzione e la confronto coi valori di ground truth del training set -> voglio minimizzare l'errore 

### Funzione di costo (errore quadratico medio)
$$J(\theta_{0}, \theta_{1})=\frac{1}{2m}\sum_{i=1}^m(h_{\theta}(x^{(i)})-y^{(i)})^2$$
Trovare: $\displaystyle \min_{\theta_{0},\theta_{1}}J(\theta_{0},\theta_{1})$

> -> elevazione al quadrato: errori di precisione possono essere positivi o negativi, potrei annullare il risultato -> elevando al quadrato diventano tutti positivi
> Non metto il modulo perché la sua derivata è più difficile da maneggiare

Parto con dei parametri (anche casuali inizialmente), testo bontà del modello con funzione di costo, cerco di migliorare il risultato manipolando i parametri e dando la nuova ipotesi alla funzione di costo
-> 0: modello ha errore nullo rispetto al ground truth
-> $+\infty$: errore massimo tra predizione e ground truth

### Ipotesi semplificata: un parametro
-> sole rette passanti per l'origine, rimuovendo $\theta_{0}$
$h_{\theta}(x_{1})=\theta_{1}x_{1}$
$J(\theta_{1})=\frac{1}{2m}\sum_{i=1}^m(h_{\theta}(x^{(i)})-y^{(i)})^2$
$\displaystyle \min_{\theta_{1}}J(\theta_{1})$

### Grafico funzione di costo
Con una particolare configurazione dei parametri, testo bontà modello con funzione di costo
-> vedo qual è l'errore con questi parametri, inserisco punto nel grafico in corrispondenza di valore del parametro-risultato funzione di costo
![[Pasted image 20250617123250.png|400]]
Grafico della funzione di costo convesso: vogliamo *trovare il punto di minimo* (minimo errore possibile)

#### Esempio con 2 parametri: grafico tridimensionale
![[Pasted image 20250617144636.png|400]]

**Linee di livello:** rappresentazione alternativa per grafici tridimensionali, mostrano i punti con stesso valore della funzione di costo 
![[Pasted image 20250617145100.png|600]]
-> minimo in corrispondenza del cerchio più interno

***

## Gradient Descent Algorithm
Algoritmo generale per trovare punto di minimo di una funzione: trovare parametri della funzione che la minimizzano -> in questo caso funzione di costo per linear regression
- scegliere parametri iniziali (casuali; spesso inizializzati a 0)
- modificarli man mano per scendere verso il minimo -> ridurre la funzione di costo J

> *"Mi guardo intorno, vedo in quale direzione intorno a me il terreno è più in pendenza verso il basso, faccio un passo in quella direzione e ripeto fino ad arrivare a un punto di minimo (intorno a me c'è piano o tutto in salita)"*

Per linear regression funziona bene perché funzione convessa; in casi più complessi, in base al punto di partenza possiamo finire in "valli" (minimi) diverse -> non è garantito arrivare a minimo assoluto
![[Pasted image 20250617150209.png|500]]
$$\text{repeat until convergence}\space\{\theta_{j}:=\theta_{j}-\alpha\frac{\partial}{\partial\theta_{j}}J(\theta_{0},\theta_{1}) \text{ for j = 0, 1} \space\}$$
> Nuovo parametro = vecchio parametro - derivata funzione di costo rispetto al parametro

Dimensione del "passo" -> parametro $\alpha≥0$ spesso adattivo nel tempo (se fosse sempre troppo grande rischio di saltare il punto di minimo)
-> $\alpha=0$ rimango con parametro identico (annullo la sottrazione della derivata)

![[Pasted image 20250617151859.png|500]]

> Significato geometrico derivata: pendenza retta

![[Pasted image 20250617152224.png|600]]

> **N.B:** strategia utile ma utilizzabile solo con funzioni derivabili

### Gradient Descent funziona correttamente?
Calcolare funzione di costo a ogni riapplicazione di GD, e plottare risultati
-> vogliamo grafico con andamento decrescente (funzione di costo deve essere sempre minore a ogni iterazione)

**Problema**: funzione di costo potrebbe non scendere a ogni iterazione, potrebbe non convergere -> può essere causato da $\alpha$ troppo grande 
-> anche $\alpha$ troppo piccolo è problematico (convergenza troppo lenta)

![[Pasted image 20250617154447.png|600]]

***

## Linear Regression multivariabile
### Notazione
- n = numero feature
- $x^{(i)}$ = feature in input dell'i-esimo esempio di training
- $x_{j}^{(i)}$ = valore j-esima feature dell'i-esimo esempio di training

A ogni feature $x_{i}$ associo un parametro $\theta_{i}$ + $\theta_{0}$ libero -> *modello interpretabile dall'umano* (ogni feature rappresenta una caratteristica, ogni parametro indica quanto contribuisce quella feature) -> soluzione *intuitiva*

### Ipotesi
$$h_{\theta}(x)=\theta_{0}+\theta_{1}x_{1}+\dots+\theta_{n}(x_{n})$$
$x_{1},\dots,x_{n}$: variabili
$\theta_{0},\dots,\theta_n$: parametri
$\theta=\begin{bmatrix} \theta_{0} \\ \theta_{1} \\ \dots \\ \theta_{n} \end{bmatrix}$
$x=\begin{bmatrix} 1 \\ x_{1} \\ \dots \\ x_{n}\end{bmatrix}$
-> facendo prodotto scalare tra i due vettori otteniamo l'ipotesi sopra: $h_{\theta}=\theta^Tx$

### Funzione di costo
$$J(\theta_{0}, \theta_{1},\dots,\theta_{n})=\frac{1}{2m}\sum_{i=1}^m(h_{\theta}(x^{(i)})-y^{(i)})^2$$
**Obiettivo:** $\displaystyle \min_{\theta_{0},\theta_{1},\dots,\theta_{n}}J(\theta_{0},\theta_{1},\dots,\theta_{n})$

### Gradient Descent per Linear Regression multivariabile
Come per GD con una variabile, ma ora faccio update simultaneo non di 2 parametri $\theta_{0},\theta_{1}$ ma di n parametri $\theta_{0},\dots,\theta_{n}$ derivando volta per volta funzione di costo per uno degli n parametri.

Otteniamo in generale: 
$$\theta_{j}:=\theta_{j}-\alpha \frac{1}{m}\sum_{i=1}^m(h_{\theta}(x^{(i)})-y^{(i)})x_{j}^{(i)}$$
**Esempio:** $\theta_{0}:=\theta_{0}-\alpha \frac{1}{m}\sum_{i=1}^m(h_{\theta}(x^{(i)})-y^{(i)})x_{0}^{(i)}$

***

## Feature Normalization
Feature hanno diversi ordini di grandezza (es. metri quadri, numero camere/piani, età della casa...) -> dati molto grandi o piccoli in valore assoluto -> problema con funzione di costo

Se feature molto diverse: coefficienti molto diversi fra loro -> funzione di costo potrebbe assumere conformazione molto "allungata" per alcuni parametri

Normalizzare: far sì che le feature si muovano all'incirca nello stesso range, si ottiene miglior performance -> tutte le feature divise per un *fattore di normalizzazione*

![[Pasted image 20250617161534.png|500]]
### Mean Normalization
Tiene in considerazione *media* e *deviazione standard* (quanto i dati si discostano dalla media) delle feature

-> attenzione focalizzata su una feature alla volta, si calcola media e deviazione standard per la singola feature
$$x'_{i}=\frac{x_{i}-\mu_{i}}{\sigma_{i}}$$
> nuovo valore = vecchio valore - media / standard deviation

### MinMax Normalization
-> attenzione una feature alla volta in maniera indipendente
$$x'_{i}=\frac{x_{i}-min(x)}{max(x)-min(x)}$$

-> sposta tutte le feature nel range [0,1]

**IMPORTANTE:** applicare *stessa funzione di normalizzazione sia a training set che a test set*!
-> se usiamo per es. MinMax, usare lo stesso min e max sia per training set che per test set, non min e max specifici per i due set -> funzione di normalizzazione si basa su due parametri (min, max), cambiandone anche solo uno il risultato della funzione sarà certamente diverso

![[Pasted image 20250617163729.png|600]]

### Denormalizzazione della predizione
Dopo normalizzazione, modello restituisce in output dati normalizzati (es. tra 0 e 1) -> normalizzazione inversa per renderli più fruibili (es. prezzo appartamento comprensibile immediatamente dall'umano)

- MinMax: $x'_{i}=\frac{x_{i}-min(x)}{max(x)-min(x)}$ -> $x_{i}=x'_{i}(max(x)-min(x))+min(x)$
- Mean: $x'_{i}=\frac{x_{i}-\mu_{i}}{\sigma_{i}}$ -> $x_{i} = x'_{i} \cdot \sigma_{i}+\mu_{i}$

![[Pasted image 20250617172928.png]]

## Polynomial Regression
Modelli basati su curve descritte da polinomi 
Ipotesi parametrizzate con $\theta$ come per LR, ma feature possono essere di grado superiore a 1
Parametri si trovano ancora con funzione di costo e GD

-> utili, modellano più precisamente dati, ma più difficili da manipolare: non è scontato trovare l'ipotesi più interessante, soprattutto ad alta dimensionalità (problema decisionale: quale ipotesi è la migliore?)

Ipotesi polinomiale può essere *trattata come Linear Regression* con opportuna sostituzione di variabili
![[Pasted image 20250617173806.png|400]]
Con elevamento a potenza le feature possono muoversi in range troppo diversi-> importante *normalizzare*

## Coefficiente di correlazione di Pearson
Si ottiene tramite formula che mette in relazione due feature (vettori)
$$\rho=\frac{1}{N-1}\sum_{i=1}^N\left(\frac{X_{i}-\mu_{X}}{\sigma_{X}}\right)\left(\frac{Y_{i}-\mu_{Y}}{\sigma_{Y}}\right)$$
Coefficiente sta in $[-1,1]$ -> se dati stanno su una retta (indipendentemente dall'inclinazione) coefficiente = 1: correlazione lineare
Con sistemi più complessi (es. reti neurali) sono utili anche feature con contributo non lineare

-> cerca di capire se due feature sono correlate: spesso usato per mettere a confronto feature e target -> capire quali feature sono più correlate all'output (scegliere per un problema di tenere solo le feature più correlate)

>**N.B:** con più feature è necessario avere molti più esempi di training (non basta aggiungerne uno per ogni feature aggiuntiva; fattore circa x10) -> scegliere feature più correlate utile per avere meno esempi di training 

