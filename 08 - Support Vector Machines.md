Classificazione binaria: cercare retta che separi i dati delle 2 classi -> problema: ci possono essere tante possibili rette che separano i dati, alcune troppo vicine ad alcuni esempi di training (esempi di training leggermente scostati possono finire nella classe sbagliata)

SVM -> trovare retta (scegliere parametri) che *massimizza il margine* (distanza fra i punti più vicini delle due classi)
>Pregio SVM: usate anche per trovare *decision boundary non lineari* (erano fondamentali prima delle reti neurali)

![[Pasted image 20250621213607.png|300]]

> [!Reminder]
> Con Logistic Regression, previsione y=1 se argomento di g(z) ≥ 0 -> $\theta_{0}+\theta_{1}x_{1}+\theta_{2}x_{2}≥0$

### Esempio
 $h_{\theta}=g(\theta_{0}+\theta_{1}x_{1}+\theta_{2}x_{2})$
 $\theta=[\theta_{0},\theta_{1},\theta_{2}]^T,x=[1,x_{1},x_{2}]^T$
-> con Logistic Regression otteniamo per es. $\theta=[-3,1,2]$
$h_{\theta}(x)=g(\theta^Tx)>0.5\to \theta^Tx>0$

Quindi il decision boundary (retta) è: $h_{\theta}(x)=0.5\to \theta^Tx=0\to-3+x_{1}+2x_{2}=0$, rappresentabile in forma matriciale come:
$\begin{bmatrix} 1 &2\end{bmatrix} \begin{bmatrix} x_{1} \\ x_{2}\end{bmatrix} -3=0$
  $w^T$    $x$     +  $b$    -> equazione generica *iperpiano*

Margine: due rette parallele al decision boundary, una sopra e una sotto
- $w^Tx+b=0$ decision boundary
- $w^Tx+b=\pm 1$ margine -> se *dati linearmente separabilli*, si può fare scaling per fare in modo che rette del margine abbiano termine noto $\pm 1$ per agevolare calcoli
![[Pasted image 20250621215358.png|400]]

> **Support vectors:** punti più vicini, sui margini, delle due classi

Scegliere $w$ e $b$ t.c. 
- $w^Tx_{i}+b≥1 \text{ for } y_{i}=1$ -> esempi classe positiva sopra al margine superiore
- $w^Tx_{i}+b≤-1 \text{ for } y_{i}=-1$ -> esempi classe negativa sotto margine inferiore
-> combinabili così: 
$$y_{i}(w^Tx_{i}+b)≥1 \space\forall x_{i}$$

## Calcolo distanza tra i due margini:
- scegliamo due punti $x_{1},x_{2}$ sul decision boundary -> equazioni $w^Tx_{1}+b=0$, $w^Tx_{2}+b=0$
- sottrazione equazioni: $w^T(x_{1}-x_{2})=0$ -> prodotto scalare tra vettore $w^T$ e vettore $x_{1}-x_{2}$ che sta sul decision boundary -> $w^T$ è *ortogonale al decision boundary* perché prodotto scalare dà 0
![[Pasted image 20250621221414.png|400]]

I due boundary sono paralleli:
- prendo $x_{1}$ sul boundary inferiore $w^Tx+b=-1$
- punto più vicino sul boundary superiore $w^Tx+b=1$ è $x_{2}=x_{1}+\lambda w$ con $w$ perpendicolare ai boundary -> vogliamo trovare $\lambda$

### Trovare $\lambda$
![[08 - Support Vector Machines 2025-06-22 11.23.53.excalidraw|300]]

Quindi la distanza tra $x_{1}$ e $x_{2}$ (ovvero l'ampiezza del margine) è: 
$$\lambda||w||=\frac{2}{||w||^2}||w||=\frac{2}{||w||}=\frac{2}{\sqrt{ w^Tw}}$$

> **Obiettivo:** massimizzare il margine $\frac{2}{\sqrt{ w^Tw }}$ -> equivale a *minimizzare* $\frac{\sqrt{ w^Tw }}{2}$ -> equivale a trovare il minimo di $\frac{w^Tw}{2}$

## Funzione di costo SVM
- Per dati separabili linearmente:
$$min_{w,b}\frac{w^Tw}{2}$$
$$\text{con questo vincolo su w: } y_{i}(w^Tx_{i}+b)≥1 \space \forall x_{i}$$

- Per dati non linearmente separabili:
  -> introdotte variabili di slack ("penalità"): un $\epsilon_{i}>0 \space \forall x_{i}$
$$min_{w,b}\frac{w^Tw}{2}+C\sum_{i}\epsilon_{i}$$$$\text{con vincolo: } y_{i}(w^Tx_{i}+b)≥1-\epsilon_{i} \text{ e } \epsilon_{i}>0 \space \forall x_{i}$$
	-> **parametro C > 0:** controlla penalità per elementi di training classificati erroneamente (scegliere in modo simile a parametro $\lambda$ per la regolarizzazione)
	- C grande: cerca di classificare tutti gli elementi correttamente, ma boundary molto stretto
	- C piccolo: boundary più largo, non tutti gli elementi potrebbero essere classificati correttamente

***

## SVM con kernel
![[Pasted image 20250622123713.png|250]]
Decision boundary non lineare -> spesso difficile da calcolare (problema decisionale per le funzioni polinomiali, decidere grado del polinomio)

Reti neurali -> risolvono il problema, ma può essere complesso trovare l'architettura adatta per la rete (grande flessibilità)

SVM -> meno gradi di libertà, immediatamente testabili

**Kernel gaussiano:** modo per calcolare similarità tra punti (modo diverso di definire distanza)
![[Pasted image 20250622122003.png|300]]
> $l^{(i)}$ landmarks -> punti sul piano 

$f_{i}=similarity(x,l^{(i)})=\exp\left(-\frac{||x-l^{(i)}||^2}{2\sigma^2}\right)$
se $x\approx l^{(1)}\to f_{1}\approx 1;f_{2}\approx 0;f_{3}\approx 0$
![[Pasted image 20250622122737.png|300]]

**Idea:** *descrivere nuovi punti* sul piano invece che con le loro coordinate, *tramite similarità coi landmark* -> 2 coordinate: 2 dimensioni; 3+ landmark: 3+ dimensioni -> mi sposto in **nuovo feature space**

### Raggio del kernel
![[Pasted image 20250622122914.png|400]]

### Feature space: esempio
![[Pasted image 20250622124852.png|250]]
Decision boundary basato sulle nuove feature $f_{1},f_{2},f_{3}: \theta_{0}+\theta_{1}f_{1}+\theta_{2}f_{2}+\theta_{3}f_{3}= 0$ -> equazione *iperpiano*
SVM con kernel ottiene: $\theta_{0}=-0.5;\quad \theta_{1}=1;\quad \theta_{2}=1;\quad \theta_{3}=0$

Abbiamo 3 punti:
- $x_{1} \space \to \space f_{1}\approx1;f_{2}\approx 0;f_{3}\approx 0 \space \to \space -0.5+1*1+1*0+0*0=0.5≥0 \space \to \space y=1$
- $x_{2} \space \to \space f_{1}\approx 0;f_{2}\approx 1;f_{3}\approx 0 \space \to \space -0.5+1*0+1*1+0*0=0.5≥0 \space \to \space y=1$
- $x_{3} \space \to \space f_{1}\approx 0;f_{2}\approx 0;f_{3}\approx 0 \space \to \space -0.5+1*0+1*0+0*0=-0.5<0 \space \to \space y=0$

-> aumentando $\sigma$ il decision boundary diventa più "uniforme", se $\sigma$ è basso il decision boundary è più focalizzato sui landmark

### Scegliere i landmark
Posti *in sovrapposizione sugli esempi di training* -> ogni punto rappresentato con similarità con sé stesso e tutti gli altri esempi
Avremo tante feature $f_{i}$ quanti esempi di training; equazione decision boundary (iperpiano) -> $\theta_{0}+\theta_{1}f_{1}+\dots+\theta_{n}f_{n}$ con n esempi di training. 

Decision boundary diventa come tante "palline" in corrispondenza di ogni landmark; in base a $\sigma$ cambia la "profondità" e uniformità del boundary (da scegliere)
-  $\sigma$ piccolo -> decision boundary molto frammentato -> si va verso overfitting (allontanandosi leggermente da un landmark, elemento passa all'altra classe - va "sotto" al boundary)
- $\sigma$ grande -> problema inverso (underfitting)


![[1403824.webp|400]]

> Nello spazio originale (2D) abbiamo decision boundary non lineare
![[Effect-of-SVM-kernels-on-the-decision-boundary-a-Polynomial-SVM-kernel-with-polynomial.ppm.png|300]]

