> **N.B:** utenti, oggetti rappresentati in forma numerica

## Classificazione
Esperienza -> classificatori usati per suddividere dati tramite *decision boundary*
Apprendimento basato su:
- esperienza
- dati -> possono essere buoni o cattivi -> focalizzare eccessivamente i classificatori su caratteristiche specifiche (predominanti) -> *bias sui dataset* (es: cigni tutti bianchi, solo in acqua)

## Regressione 
Trovare modello sui dati per descriverli -> interrogarlo su esempi non visti con l'esperienza per generare *previsioni*: dato un input, dà risultato numerico

***

## Dataset
-> **Training Set**: dati per allenare il modello
-> **Test Set**: dati usati come dati reali per testare il modello, capire se funziona

***

## Supervised Learning
Dati correlati da annotazioni fornite dall'umano -> etichettati come positivi/negativi (downside: time consuming)

## Unsupervised Learning
Dati non etichettati (non vengono fornite informazioni per l'apprendimento) -> raggruppabili in sottoinsiemi basandosi su varie caratteristiche -> obiettivo: *dare struttura* a dati non strutturati, aiuta a semplificare la ricerca

### Esempio
- Foto etichettate "bosco" vs "spiaggia"
- Foto mescolate, da classificare senza input iniziali in base a caratteristiche comuni
  -> Google foto: cluster con i volti simili senza label fornite dall'utente
  -> Segmentazione mercato per prodorre prodotti diversi ai vari clienti (es. consigliati Amazon)

## Self Supervised Learning
-> per ovviare ai tempi lunghi per annotazione nel Supervised Learning
Utilizzare natura dei dati per aiutare all'apprendimento, senza intervento umano (nei casi di grandissimi dataset)
-> feedback positivo/negativo proviene dal web o altra fonte dati

### Esempio
Trarre frase da set dati, mascherare una parola, chiedere alla macchina la parola mancante
-> funzionamento di ChatGPT: parole in input, fingo di aver mascherato una parola alla fine e la genero (in base ad apprendimento fatto su dati presenti sul web); passo la frase più lunga di nuovo in input e continuo, genero una risposta testuale
-> mascheramento funziona anche per generazione immagini







