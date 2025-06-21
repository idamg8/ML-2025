## Holdout Method
Dataset diviso in:
- Training set
  1. Training Set vero e proprio
  2. **Validation Set:** usato per valutare come sta andando l'addestramento, fare tuning prima di arrivare ai test col Test Set; es: indecisione tra più ipotesi (caso Regressione Polinomiale), le testo sul Validation Set, mantengo quella con performance migliore (es. f-score) 
- Test set: test finale delle performance del modello addestrato 

![[Pasted image 20250620173303.png|400]]

-> spesso si fanno tuning con validation set; una volta trovata ipotesi migliore si riaddestra il modello con il Training Set completo (reinserendo il Validation Set -> modello addestrato su tutti i dati) e infine test con Test Set

**Problema:** performance del modello basata su Validation Set che è porzione piccola di dati (potrebbero esserci bias, dati più semplici del caso generale, casi più difficili non considerati per la valutazione se non stanno nel Validation Set -> alta performance misurata potrebbe non rispecchiare performance effettiva del modello)

## K-Fold Cross Validation
-> vogliamo validare modello nel modo più *fair* possibile

Training Set diviso in $k$ (solitamente 3/5/10) segmenti; esecuzione $k$ esperimenti con ogni volta Validation Set formato da un segmento diverso -> $k$ modelli con performance diverse, la performance finale sarà la media $mean(P_{1},\dots,P_{k})$

-> stesso problema si può presentare col Test Set (cross validation potenzialmente anche con Test Set: strategia non utilizzata perché quantità di esperimenti proibitiva)

***

## Regularization

![[Pasted image 20250620174614.png|600]]

**Underfitting:** modello non rappresenta con abbastanza precisione i dati
**Overfitting:** se ci sono troppe feature, ipotesi può modellare molto bene i dati ma non riuscire a generalizzare nuovi esempi
-> problema sia per classificazione che per regressione

### Classificazione

![[Pasted image 20250620175332.png|400]]

- Underfitting: performance basse in training e test set
- Overftting: con decision boundary troppo frastagliato è possibile che piccolo scostamento dei dati non li renda più riconoscibili (es. volto con/senza occhiali, capelli diversi)
  -> modello più espressivo dei dati, *non generalizza*
  -> basse performance in test set

## Risolvere overfitting
-> Ridurre numero feature: spesso difficile

-> **Regolarizzazione:** mantiene tutte le feature ma cerca di ridimensionare alcune feature che potrebbero risultare preponderanti (e in mancanza di esse, il modello potrebbe non funzionare come previsto)

### Esempio
Articoli di giornale in un certo periodo storico menzionano spesso Trump e sono articoli di politica -> parametro associato a Trump molto alto, risulta preponderante nella classificazione di articoli politici/non politici (ma se un articolo parla di politica senza menzionare Trump, non classificato correttamente)

## Ridge regression
-> Si aggiunge termine di regolarizzazione alla funzione di costo (es. di Linear Regression)
$$J(\theta)=\frac{1}{2m}\left[\sum_{i=1}^m(h_{\theta}(x^{(i)})-y^{(i)})^2+\lambda \sum_{j=1}^n\theta_{j}^2\right]$$
> $\lambda$ coefficiente di regolarizzazione

![[Pasted image 20250620182730.png]]

Obiettivo: minimizzare funzione J, quindi
- **Coefficiente $\lambda$ basso ($\approx 0$):** B si cancella, bontà funzione di costo dipende solo da bontà del modello predittivo indipendentemente dai $\theta$ -> potrebbe esserci overfitting -> *basso errore su training set, alto su validation*
- **Coefficiente $\lambda$ alto:** termine B sale estremamente -> tenere parametri $\theta$ più possibile vicini a 0 per minimizzare la sommatoria  (evitare overfitting su feature specifiche) -> *errore alto sia su training che su validation* (si ignora il termine A che effettivamente valuta bontà del sistema)
![[Pasted image 20250620185140.png|400]]
> Variance: overfitting
> Bias: underfitting

***

## Diagnostic
-> strumenti diagnostici

### Learning Curve
-> mettere a confronto errore e dimensione del Training Set, calcolare 2 funzioni di costo (una per Training Set e una per Validation - fisso)
$$J_{train}(\theta)=\frac{1}{m_{train}}\left[\sum_{i=1}^{m_{train}}Cost(h_{\theta}(x,y))\right]$$
$$J_{vs}(\theta)=\frac{1}{m_{vs}}\left[\sum_{i=1}^{m_{vs}}Cost(h_{\theta}(x,y))\right]$$
![[Pasted image 20250620190121.png|400]]

> Addestro modello con sempre più esempi di training 

Curva verde:
- pochi dati training: possono esserci problemi di riconoscimento in validation
- molto dati training: diminuisce errore in validation

Curva blu:
formula di costo calcolata in base a esempi di training -> se sistema ha pochi dati, avrà alta performance ed errore di predizione bassissimo; aggiungendo dati aumenta errore (aumenta variabilità del dataset)

### Underfitting = high bias
![[Pasted image 20250621103929.png]]

Errore molto alto sia in training che in validation -> piccolo gap fra i due errori
-> aggiungere esempi al training set non aiuta le performance

### Overfitting = high variance
![[Pasted image 20250621104057.png]]

Forte gap tra errore in training e validation
-> errore in validation sta diminuendo, aumentando esempi di training performance in validation potrebbe migliorare e scendere sotto soglia rossa

## Possibili soluzioni
- + esempi di training -> risolve high variance 
- - feature -> risolve high variance
- + feature -> risolve high bias
- diminuire $\lambda$ -> risolve high bias (viene dato più "potere" ai parametri delle feature)
- aumentare $\lambda$ -> risolve high variance (parametri troppo influenti fanno specializzare eccessivamente modello su alcune feature -> aumentare $\lambda$ diminuisce influenza parametri)

