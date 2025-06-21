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
