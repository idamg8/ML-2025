> [!Curiosità storica]
> Reti neurali importanti a livello mondiale dal 2012: professoressa di Stanford creato database enorme di immagini (ImageNet), competizione a livello mondiale -> distribuiti dati di training con immagini + label, alla fine rilasciate immagini di test -> inviare risultati e vedere quale modello fa previsioni con errore minore.
> Conferenza a Firenze con workshop dedicato a questa competizione -> università di Toronto prima in classifica con gap di tanti punti percentuali rispetto ad altre università -> AlexNet rete neurale, finora rimaste in background (teoria già esistente dagli anni '90) perché molto onerose da calcolare, ora addestramento affidato a *schede grafiche* invece che CPU -> molto efficienti con somme e prodotti, performance enormemente migliori per addestramento!

Logistic Regression: $h_{\theta}(x)=g(\theta^Tx)=\frac{1}{1+e^{-\theta^Tx}}$ 

-> per reti neurali cambio notazione:
- $\theta=\begin{bmatrix} \theta_{1}\\ \theta_{2}\\ \dots\\ \theta_{n}\end{bmatrix}$ invece che $\theta_{i}$ diventano $w_{i}$ **pesi**
- $x=\begin{bmatrix} x_{1}\\ x_{2}\\ \dots\\ x_{n}\end{bmatrix}$
- $b=\theta_{0}$ **bias** -> unico termine senza feature associata

$$h_{\theta}(x)=g(\theta^Tx+b)=\frac{1}{1+e^{-(\theta^Tx+b)}}$$

![[07 - Neural Networks 2025-06-21 12.03.31.excalidraw|400]]

![[Pasted image 20250621121321.png|400]]

Sigmoide: 
- output 1 se argomento della funzione g(z) è alto
- output 0 se argomento della funzione g(z) è basso (< 0)
-> mima funzionamento neuroni: si attivano e inviano impulso solo se a loro arriva forte impulso

Rete neurale in forma più basilare = Logistic Regression
MA: nelle reti neurali la *funzione di attivazione* può cambiare (non necessariamente sigmoid)

Inizialmente parametri scelti casualmente, poi migliorati con GD

Per casi più complessi di classificazione si possono *effettuare più logistic regression*, ognuna associata ad un neurone e con i suoi parametri (pesi degli archi, bias associato al neurone)

![[Screenshot 2025-06-21 alle 15.31.06.png|500]]
> in questo caso i due neuroni neri effettuano logistic regression, neurone finale arancione mette insieme i risultati -> vengono creati decision boundary definiti da 2+ rette o piani

**Obiettivo:** usare reti più piccole possibili 

Aumentando profondità della rete (aggiungendo livelli di neuroni), essa migliora sempre di più la creazione di decision boundary 
![[Pasted image 20250621154742.png|500]]

>-> ridurre mano a mano dimensionalità sui vari layer man mano che scendo in profondità: primi livelli vicini a input layer con molti neuroni, andando verso output layer sempre meno neuroni
![[Screenshot 2025-06-21 alle 15.42.44.png|300]]

**Rete collegata (densa):** tutti i neuroni sono collegati a tutti i successivi

>Dato un problema, che rete neurale utilizzare? Problema ancora non risolto -> cercare problemi simili usati da altri, modificare opportunamente

- Rete neurale senza funzione di attivazione, anche se complessa, si comporta come rete neurale a un solo neurone.
- Anche se stessa architettura di due reti neurali, con *pesi diversi potrebbero risolvere problemi completamente diversi* -> stesso problema con architetture diverse richiede pesi diversi

![[Pasted image 20250621160041.png]]
> Tot. parametri: 21 (12 pesi + 4 bias hidden layer, 4 pesi + 1 bias output layer)
> **Operatore point-wise:** applica stessa funzione a ogni componente di matrice/vettore

-> rete neurale funziona con *operazioni su matrici* (GPU specializzate in questo, più veloci delle CPU)

**---> Esempi interessanti su slide!**

***

## Addestramento rete neurale
1. Inizializzazione randomica dei pesi
2. Algoritmo di **Back Propagation:** (scalabile per numero di layer e neuroni)
   - Forward Step: inserisco elemento di training nella rete (input), eseguo operazioni con pesi e bias fino all'output, calcolando l'errore con la funzione di costo
   - Backward Step: errore calcolato viene propagato all'indietro e serve ad aggiornare i pesi se l'errore è molto grande -> funzione di costo dipende dai parametri (pesi, bias)
     -> $J(w_{1},w_{2},w_{3},\dots,b_{1},b_{2}\dots)=\frac{1}{2m}\sum_{i=1}^m(h_{w,b}(x^{(i)})-y^{i})$ -> applico GD per ogni parametro $w,b$

***

## Funzioni di attivazione
- Sigmoide: importanza storica e per Logistic Regression, ma ora poco usata 

- **Rectified Linear Unit (ReLu):** usata per *layer intermedi*; se valore < 0 restituisce 0 (neurone non si attiva), se impulso > 0 restituisce valore dell'impulso in entrata
  es. R(-10) = 0 | R(-5) = 0 | R(5) = 5 | R(7) = 7
![[07 - Neural Networks 2025-06-21 17.23.36.excalidraw|400]]
	-> **Problema:** non derivabile in 0, ma valore esattamente 0 molto improbabile, quindi usata ugualmente

Rete neurale per classificazione binaria: sigmoide all'ultimo layer (output 0/1)

Rete neurale per regressione, all'ultimo layer sigmoide non appropriata; strategie:
- no funzione di attivazione
- se ho certezza che valore di predizione > 0: ReLu 
  (es. stimare numero prodotti venduti il prossimo mese)

***

## Loss Functions
Funzioni di costo applicate alle reti neurali 

1. Classificazione binaria: Binary Cross-Entropy Loss 
$$L(W,b)=\frac{1}{m}\sum_{i=1}^m-y^{(i)}\log\left(h_{W,b}(x^{(i)})\right)-(1-y^{(i)})\log(1-h_{W,b}(x^{(i)}))$$
2. Regressione: Mean Squared Error Loss
$$L(W,b)=\frac{1}{m}\sum_{i=1}^m\left(h_{W,b}(x^{(i)})-y^{(i)}\right)^2$$

-> uguali a funzioni di costo per Logistic o Linear Regression, ma con *ipotesi dipendenti da pesi e bias*: reti neurali possono **applicarsi sia a classificazione che regressione** anche con la stessa architettura, ma con funzioni di costo differenti
-> calcolo funzione di costo, ne calcolo la derivata per il Gradient Descent -> aggiornare i parametri: si cambiano se la derivata ha un valore, se la derivata è 0 il parametro rimane lo stesso (motivo per cui sigmoid abbandonata nel tempo -> valori prima di $\approx 6$ e dopo $\approx 6$ è costante, derivata 0)

>**N.B:** derivata funzione rispetto a un parametro -> quanto la funzione varia in funzione della variazione del parametro

***

## Classificazione multiclasse
Reti neurali nativamente estendibili a problemi multiclasse -> output layer con più neuroni, uno per classe (si attiva solo quello della classe corrispondente all'input)

![[Pasted image 20250621174509.png]]

Importante: decidere prima dell'addestramento quale label usare per ogni classe (es. pedestrian = [1 0 0 0])

## Softmax Layer
Vogliamo neuroni dell'output layer coordinati tra loro -> restituiscono informazioni di probabilità sulla classe di appartenenza dell'input: somma delle probabilità di ogni classe = 1 (100%)

Nel layer finale non c'è una funzione di attivazione collegata a ogni neurone, ma i risultati di ogni neurone del layer vengono combinati:
$$z^{[i]}=W^{[i]}a^{[i-1]}+b^{[i]}$$
	con $a^{[i-1]}$ risultati del layer precedente moltiplicati per i pesi $W^{[i]}$
$$Softmax:a^{[i]}=\frac{e^{z^{[i]}}}{\sum_{J=1}^Ne^{z_{J}^{[i]}}}$$
### Esempio
![[07 - Neural Networks 2025-06-21 18.39.41.excalidraw]]

## Cross-Entropy Loss
$$L(W,b)=-\sum_{i=1}^my^{(i)}\log\left(h_{W,b}(x^{(i)})\right)$$
![[Pasted image 20250621185045.png|250]]
> simile alla Binary Cross-Entropy Loss vista sopra, ma si concentra *solo sulla classe positiva* (le negative sono potenzialmente infinite)

### Esempio
![[Pasted image 20250621184637.png|400]]
> le classi negative sono poste a 0 nel ground truth $y$ -> nel prodotto si eliminano le classi negative, rimane solo quella positiva

Errore calcolato solo sulla classe positiva, vogliamo che sia più vicino possibile a 0 -> far salire la probabilità della classe positiva significa far scendere le probabilità delle classi negative (siccome la somma deve essere sempre = 1) 



