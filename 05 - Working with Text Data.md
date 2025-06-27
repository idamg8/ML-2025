Dato deve essere rappresentato in forma numerica -> testo problematico: trovare rappresentazione che tenga conto della semantica più che dei caratteri (es. catturare differenza cat e cut) -> ASCII non considera semantica (due lettere distanziate di un numero ma potrebbero avere distanza semantica molto maggiore; problema non avviene con le immagini)

*Sentiment Analysis:* capire se il feedback (es. recensioni) è positivo o negativo tramite un classificatore

## Preprocessing
Testo viene processato prima di essere dato in input a un modello di IA (talvolta fatto anche su immagini, es: applicare filtri per migliorare visibilità nell'immagine)

- *Lower-casing*: uniforma il testo, più semplice da utilizzare (non sempre: es. se bisogna trovare i nomi propri)
- *Word stemming*: ridurre parole alla loro forma base (es. tuned, tune, tuning -> tun)
  -> token: testo suddiviso in parole che il sistema riconosce (sa dare contenuto semantico), numero di token dipende da grandezza del sistema -> più token = maggiore varietà semantica
- *Lemmatization*: (parallelo a word stemming, più complicato ma spesso non porta migliori performance) raggruppa parole a un singolo termine (lemma), es: better -> good

**Stop words:** parole di uso molto comune per rendere più comprensibile il testo (es: and, is, at, a, has...) ma hanno basso valore semantico (usate in tutti i contesti, non danno indicazioni semantiche, come invece per es: parlamento / fuorigioco -> forti indicazioni semantiche)
-> possono essere tolte

**Punteggiatura:** inizialmente ignorata, ultimamente di interesse perché possono avere importanza semantica

**Regular expressions:** cercare nel testo elementi per capire se estrarli o rimuoverli (es: per sostituire tag HTML con singolo carattere per semplificare, oppure rimuoverli)

***

## Bag of Words
- Si crea un *dizionario* basato su dati usati nell'addestramento -> guardare testo, controllare se parole sono già nel dizionario, se no vengono aggiunte (può essere anche multilingua, es. ChatGPT)
  -> posso decidere di considerare solo parole che appaiono più di tot volte 
 - Si mappa ogni parola del testo pre-processato all'indice del termine corrispondente nel dizionario -> *rappresentazione numerica*, tuttavia variabile in base alla dimensione del testo: avremo vettori di lunghezza diversa e quindi punti su spazi a dimensionalità diversa
   -> in che spazio metto decision boundary?
   -> vogliamo rappresentazione fissa
   - creiamo vettore di dimensione del vocabolario -> spazio a dimensionalità fissa
   - analizziamo testo in rappresentazione numerica: ogni occorrenza di un numero inserita nel vettore -> testo rappresentato dal conteggio delle parole 

***

## TF-IDF
Pesa in modo diverso contenuto informativo delle parole (versione avanzata di BoW)
Term-Frequency -> BoW: considera solo la frequenza delle parole
IDF (Inverse Document Frequency) -> parole molto frequenti devono avere impatto più basso (informazioni semantiche potrebbero essere meno spiccate, es: "is")

$$tfidf(w,d)=tf\times\left(\ln\left(\frac{N+1}{N_{w}+1}\right)+1\right)$$
**Notazione:**
- $N$ = numero documenti nel training set
- $N_{w}$ = numero documenti in cui appare parola $w$ (non ha forte valore discriminativo)
- $tf$ = numero di volte che $w$ appare nel documento $d$

### Esempio
$N=100$
$N_{w}=100$ se la parola è per es. "is"
$\ln\left(\frac{100+1}{100+1}\right)=\ln(1)=0$
-> valore $tfidf$ molto basso rispetto a parola che appare una sola volta

### Normalizzazione L-2
Nel pacchetto Scikit-learn TF-IDF include normalizzazione L-2:
$$v_{\text{norm}}=\frac{v}{||v||_{2}}=\frac{v}{\sqrt{ v_{1}^2+v_{2}^2+\dots+v_{n}^2 }}$$
> **N.B:** dividere ogni componente di v per la norma

## Limitazioni
Frasi simili con termini in posizioni diverse (semantica molto diversa!) hanno stessa rappresentazione BoW.
es: "Mary is smarter than John" vs "John is smarter than Mary"

Considerare parole composte se appaiono spesso insieme con significato preciso
es: "not bad" "New York" "not worth"

***

## Word Representations
Invece di concentrarsi su frasi o testi interi -> concentrarsi sulla semantica di singole parole (rappresentazione numerica per singole parole)

Strategia semplice: associare a ogni parola del vocabolario un numero via via crescente -> tuttavia i numeri non danno contributo semantico

**One-hot Vectors:** vettore di dimensione del vocabolario con tutti 0 e 1 solo in corrispondenza della parola da rappresentare
$+$ semplice
$+$ non ha problemi di ordine
$-$ vettori enormi
$-$ distanza tra due vettori si ottiene sempre cambiando uno 0 in 1 e un 1 in 0 -> tutte le parole hanno la stessa distanza (tuttavia non è vero semanticamente)

![[Pasted image 20250620171124.png|400]]

**Word Embeddings:** far sì che parole vicine numericamente abbiano significato semantico simile e viceversa
-> si definisce asse semantico con estremi (es. negativo/positivo), si posizionano le parole di conseguenza -> parole rappresentate da valore numerico in base a coordinata sull'asse
-> si possono aumentare gli assi (es. concreto/astratto)

![[Pasted image 20250620170152.png|400]]
$+$ bassa dimensionalità
$+$ incorpora significato (*distanza semantica* tra parole)

-> inizialmente tutte parole rappresentate da numeri casuali
-> in base a quanto due o più parole occorrono insieme nei documenti analizzati (o in base a quanto sono simili i contesti di utilizzo), si avvicinano numericamente (es. re/regina, mamma/papà/famiglia/casa)
-> aumentando dimensionalità feature space si aumenta espressività (es. ChatGPT ca. 500 dimensioni con parole di tante lingue tutte insieme)

