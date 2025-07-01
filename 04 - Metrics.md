Metriche seguenti collegate al mondo della classificazione (Logistic Regression)

## Confusion Matrix
![[Pasted image 20250618162424.png|300]]
- Valori reali: Vero/Falso
- Valori predetti: Positivo/Negativo

> **N.B:** spesso nel ML dato un dataset, 70% usato per training e 30% per test

Sistema molto buono quando la diagonale principale contiene la maggior parte degli elementi (ci sono più veri positivi e veri negativi)

Matrice sopra si adatta a problema binario, possono essere adattate a problemi multiclasse
![[04 - Metrics 2025-06-18 17.01.07.excalidraw|250]]
-> questo classificatore a 3 classi tende a confondersi tra classe 1 e 3; classe 2 identificata perfettamente 
(es: andare in bicicletta - correre - camminare -> più difficile distinguere correre e camminare)

***

## Accuracy
$$\text{Accuracy}=\frac{TP+TN}{FP+FN+TP+TN}$$

-> se tutti gli elementi stanno in TP e TN (diagonale completa) abbiamo $\frac{TP+TN}{TP+TN}=1$

**Problema:** accuratezza non descrittiva se problema non bilanciato -> dove le due classi sono molto sbilanciate

### Esempio
![[04 - Metrics 2025-06-18 16.38.53.excalidraw|250]]
Accuratezza: $\frac{990+0}{10+0+0+990}=\frac{990}{1000}=0.99$
Ma il sistema non funziona bene: predice sempre positivo, accuratezza esce alta solo perché ci sono effettivamente più elementi della classe positiva che negativa -> risultato falsato
(Se ci fossero più elementi della classe negativa, l'accuratezza diminuirebbe)

***
![[Pasted image 20250618164708.png|400]]
## Precision
Quante volte il sistema predice Positivo in modo corretto (quanti sono TP tra tutti gli esempi classificati Positivi)
*"Quante volte il sistema alza la mano correttamente"*
$$P=\frac{TP}{TP+FP}$$
Nell'esempio: $P=\frac{5}{8}$

***

## Recall
Quante volte il sistema predice Positivo tra tutti gli elementi effettivamente Positivi (quanti sono TP su tutti gli elementi che dovrebbero essere classificati Positivi)
$$R=\frac{TP}{FN+TP}$$
Nell'esempio: $R=\frac{5}{12}$

> **N.B:** spesso Precision e Recall vanno in contraddizione (al salire di una scende l'altra) -> bisogna trovare un compromesso

***

## F-Score (F1)
Unisce Precision e Recall 
$$\text{F-score}=2\cdot \frac{P\cdot R}{P+R}$$
Unica metrica che dà informazioni globali sulla bontà del sistema -> F-score alta se sia Precision che Recall sono alte