# Progetto Finale per il corso di Data Analyst 📊

Questo progetto rappresenta il lavoro conclusivo del corso di Data Analyst e si concentra sull'analisi e l'esplorazione di un ampio dataset relativo agli anime. L'obiettivo principale è stato quello di estrarre informazioni significative dai dati, individuare trend e pattern interessanti, e offrire strumenti interattivi per la consultazione e l'approfondimento dei risultati ottenuti.

### Dataset 📁

Il [dataset](https://www.kaggle.com/datasets/tanishksharma9905/top-popular-anime) utilizzato è stato scaricato da Kaggle e contiene una lista dettagliata degli anime, con informazioni provenienti dal sito MyAnimeList. Tra le variabili disponibili figurano titolo, anno di uscita, genere, punteggio (score), numero di episodi, popolarità e altri attributi rilevanti per l'analisi.

### Analisi Esplorativa 🔍

Nel notebook `progetto_finale_anime.ipynb` è stata condotta un'analisi esplorativa approfondita del dataset. Sono stati esaminati la distribuzione dei punteggi nel tempo, la popolarità dei diversi generi, la relazione tra numero di episodi e valutazione, e altri aspetti salienti. Sono stati utilizzati grafici e statistiche descrittive per visualizzare i principali trend e facilitare la comprensione dei dati.

Sono state inoltre aggiunte due analisi avanzate:
- **Regressione lineare multipla:** è stato implementato un modello di regressione lineare multipla per prevedere il punteggio degli anime sulla base di variabili come il numero di episodi, i generi trattati e l'anno di uscita, al fine di individuare le relazioni tra queste caratteristiche e il punteggio assegnato.
- **KNNImputer:** per gestire i valori mancanti nel dataset, è stato utilizzato il metodo KNNImputer, che consente di stimare i dati mancanti sfruttando la similarità tra le osservazioni, migliorando così la qualità e l'affidabilità delle analisi successive.

### Chatbot 🤖

Il file `progetto_finale_anime_chatbot.py` implementa un chatbot basato sulle API di OpenAI, progettato per interagire direttamente con il dataset. Il chatbot permette agli utenti di porre domande sugli anime presenti, ricevere suggerimenti personalizzati, ottenere statistiche e informazioni dettagliate, e navigare tra i dati in modo intuitivo. Sono state definite funzioni e strumenti specifici per migliorare l'esperienza di consultazione.

**Nota:** Per utilizzare il chatbot è necessario disporre di una chiave API OpenAI valida, da salvare in un file `.env` nella directory del progetto.

### Struttura del progetto 🗂️

- `progetto_finale_anime.ipynb`: Notebook per l'analisi esplorativa dei dati.
- `progetto_finale_anime_chatbot.py`: Script Python per l'interazione tramite chatbot.
- `popular_anime.csv`: File CSV contenente il dataset utilizzato per l'analisi.
- `README.md`: Documentazione e istruzioni per l'uso del progetto.
- `requirements.txt`: Elenco delle librerie Python necessarie per eseguire il progetto.

### Requisiti ⚙️

- Python 3.x
- Librerie: pandas, matplotlib, openai, dotenv, ecc.
- Chiave API OpenAI per l'utilizzo del chatbot

### Istruzioni 📝

1. Clonare il repository e installare le dipendenze necessarie eseguendo:

    ```bash
    pip install -r requirements.txt
    ```
2. Inserire la propria chiave API OpenAI nel file `.env`.
3. Eseguire il notebook per esplorare i dati o avviare il chatbot per interagire con il dataset.

Questo progetto offre un esempio pratico di come applicare tecniche di data analysis e strumenti di intelligenza artificiale per l'esplorazione e la valorizzazione di dati reali.