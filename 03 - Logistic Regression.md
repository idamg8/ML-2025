Logistic Regression -> *classificatore* 

## Classificazione binaria
Ipotesi ha valori tra 0 e 1 (non tutti i valori reali come per LR) -> 2 classi
Sfrutta idea della LR inserendola in una funzione sigmoidale

**Funzione sigmoid:** 
$$g(z)=\frac{1}{1+e^{-z}}$$
![[Pasted image 20250618112353.png|300]]
- $z$ molto grande: $e^{-z}$ tende a 0 -> $g(z)=\frac{1}{1+0}=1$ -> asintoto orizzontale a 1
- $z$ molto piccolo: $e^{-z}$ diventa molto grande -> $g(z)=\frac{1}{1+\infty}=0$ -> asintoto orizzontale a 0
- $z=0\to g(z)=\frac{1}{1+1}=\frac{1}{2}=0.5$
Quindi sigmoid mappa tutti i valori in $[0,1]$.

Per Logistic Regression si usa versione della sigmoide parametrizzata con $\theta_{0},\dots,\theta_{n}$ -> funzione composta
$$h_{\theta}(x)=g(\theta^Tx)=\frac{1}{1+e^{-\theta^Tx}}$$
> **N.B:** $\theta^Tx$ -> prodotto scalare tra parametri $\theta$ e feature x -> $\theta_{0}+\theta_{1}x_{1}+\dots+\theta_{n}x_{n}$; un parametro per feature + $\theta_{0}$ libero

***

## Decision Boundary
Discriminare tra classe 0/1 -> *valore di discriminazione (soglia decisionale) arbitrario 0.5* (spostabile per esigenze diverse)

**IMPORTANTE:** $z > 0: g(z) > 0.5$ e viceversa -> $g(\theta^Tx)>0.5$ quando $\theta^Tx>0$

Usando algoritmo di Logistic Regression possiamo ottenere i parametri $\theta$; con $\theta^Tx=0$ otteniamo $h_{\theta}(x)=0.5$

### Esempio
$h_{\theta}(x)=g(\theta_{0}+ \theta_{1}x_{1}+\theta_{2}x_{2})$ con $\theta=[\theta_{0},\theta_{1},\theta_{2}]^T,x=[1,x_{1},x_{2}]^T$
LR -> $\theta=[-3,1,2]$
$h_{\theta}(x)=0.5\to \theta^Tx=0\to-3+x_{1}+2x_{2}=0$ -> possiamo disegnare la **retta decisionale**: $x_{1}=3-2x_{2}$
Punti sopra la retta: valore > 0.5 -> classe 1; punti sotto la retta: valore < 0.5 -> classe 0 
![[Pasted image 20250618152456.png|300]]
***

## Funzione di costo
> **N.B:** $\log = \ln$ nelle slide

Trovare $\theta$ che verifica $\begin{cases} h_{\theta}(x) &≥ 0.5 &\text{if } y=1\\ h_{\theta}(x) &≤ 0.5 &\text{if } y=0 \end{cases}$
$$J(\theta)=\frac{1}{m}\sum_{i=1}^mCost(h_{\theta}(x),y)$$
$$Cost(h_{\theta}(x),y)=-y\log(h_{\theta}(x))-(1-y)\log(1-h_{\theta}(x))$$

Funzione *derivabile e convessa* -> GD per trovare parametri ottimali
Alternanza dei termini in base all'esempio di training considerato: 
- $y=0$ si annulla il primo termine $-y\log(h_{\theta}(x))$ -> rimane $-\log(1-h_{\theta}(x))$
- $y=1$ si annulla il secondo termine $-(1-y)\log(1-h_{\theta}(x))$ -> rimane $-\log(h_{\theta}(x))$

![[Pasted image 20250618153803.png|600]]
![[images.png|200]]

1. $-\log(h_{\theta}(x))$ quando $h_{\theta}(x))=1$ ovvero l'esempio in analisi è nella classe 1 -> $y=1$ come preso in ipotesi: il modello ottiene il risultato corretto -> errore nullo (funzione di costo in corrispondenza di 1 è pari a 0)
   Se l'ipotesi dà risultato vicino a 0 con $y=1$ allora l'errore è molto alto -> funzione di costo in prossimità di 0 tende a $+\infty$
2. $-\log(1-h_{\theta}(x))$ simmetrico: abbiamo posto $y=0$ -> esempio di training "negativo"
   - se $h_{\theta}(x)=0$ anche funzione di costo pari a 0 -> errore nullo
   - se ipotesi dà risultato vicino a 1, funzione di costo tende a $+\infty$

***
![[Pasted image 20250618155600.png|500]]
***

## No-linear Decision Boundary
Se ci sono feature di grado superiore a 1: da retta otteniamo curva -> decision boundary non lineare

### Esempio
![[Pasted image 20250618160340.png]]
-> con questi parametri otteniamo equazione circonferenza centrata in 0 e di raggio 1

Soluzione quasi mai utilizzata: problema decisionale per trovare funzione interna a $g$

***

## Classificazione multiclasse
Problema risolvibile con composizione di più Logistic Regression 

**One-vs-all approach:** scompongo problema in problemi più semplici
-> es. 3 classi: considero la prima classe contro le altre due unificate, poi la seconda contro le altre due, infine la terza: mi riduco a 3 *problemi binari* -> compongo i risultati
-> per trovare classe di nuovo elemento x: scegliere il classificatore che massimizza $h_{\theta}(x)$
