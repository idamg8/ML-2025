> [!Curiosità]
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
- output 0 se argomento della funzione g(z) è basso (0 o inferiore)
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

