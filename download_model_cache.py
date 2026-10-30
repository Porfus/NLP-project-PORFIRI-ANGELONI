"""
Script di pre-caching del modello Transformer e del Tokenizer da eseguire sul Login Node del Cluster
(o in locale) dove è presente la connessione a Internet.
Salva pesi e vocabolario in una directory locale './models/bert-base-italian-cased'
in modo che i nodi di calcolo GPU offline possano caricarlo senza errori di connessione.
"""

import os
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "dbmdz/bert-base-italian-xxl-cased"
SAVE_DIR = "models/bert-base-italian-xxl-cased"

def download_and_cache():
    print(f"Download e caching locale di '{MODEL_NAME}' in '{SAVE_DIR}'...")
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(SAVE_DIR)
    print("-> Tokenizer salvato con successo.")

    model = AutoModel.from_pretrained(MODEL_NAME)
    model.save_pretrained(SAVE_DIR)
    print("-> Pesi del Modello salvati con successo.")
    print("\nCompletato! Ora il modello può essere caricato in modalità 100% offline dai nodi di calcolo GPU.")

if __name__ == '__main__':
    download_and_cache()
