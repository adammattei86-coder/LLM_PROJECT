README — Fine-Tuning T5
# Fine-Tuning T5

Questo progetto permette di fare il **fine-tuning di un modello T5** su un dataset personalizzato e poi usarlo per generare output tramite inferenza.

---

##  Cosa fa il progetto

- Allena un modello T5 su un dataset personalizzato
- Salva il modello addestrato
- Permette di usare il modello per generare risposte (inferenza)
- Include una pipeline semplice per training e test

---

##  Struttura del progetto


fine_tuning/
|--- train.py → script per il training
|--- dataset.py → caricamento dataset
|---tokenizer.py → gestione tokenizzazione
|--- config.py → parametri del modello

ollama_pipeline/
|--- inference.py → esecuzione del modello

data/
|--- train.json → dataset training
|--- eval.json → dataset valutazione

outputs/
|--- model/ → modello salvato dopo training


---

##  Installazione

Installa le dipendenze:

```bash
pip install -r requirements.txt

- Training del modello

Per avviare il fine-tuning:

python fine_tuning/train.py

- Usare il modello (inferenza)

Dopo il training, puoi generare output con:

python ollama_pipeline/inference.py

- Cosa devi fare prima di partire
Assicurati che il dataset sia dentro la cartella data/
Controlla i parametri nel file config.py
Assicurati di avere installato PyTorch e Transformers

- Output del progetto

Dopo il training troverai il modello salvato in:

outputs/model/

- Requisiti
Python 3.10+
PyTorch
Transformers (Hugging Face)
Datasets

- Flusso del progetto
Prepari dataset
Avvii training (train.py)
Il modello viene salvato in outputs/model
Usi inference.py per testarlo