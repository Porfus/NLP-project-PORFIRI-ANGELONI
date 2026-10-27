"""
Modulo per l'estrazione delle feature ausiliarie simboliche e morfologiche:
1. Tokenizer Fertility Ratio: len(subwords) / len(words)
   Cattura quanto il tokenizer dell'italiano standard fatica/frammenta le parole del dialetto.
2. Character n-grams TF-IDF:
   Cattura le desinenze, suffissi e peculiarità fonetiche subword (es. terminazioni in -u, -ei, troncamenti).
3. Matrice tensoriale combinata:
   Normalizzazione e combinazione per il ramo ausiliario della rete neurale.
"""

import re
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

class DialectAuxiliaryFeatureExtractor:
    def __init__(self, tokenizer, max_char_features=500, char_ngram_range=(2, 4)):
        self.tokenizer = tokenizer
        self.max_char_features = max_char_features
        self.char_ngram_range = char_ngram_range
        
        # Vectorizer per n-grammi di caratteri
        self.char_tfidf = TfidfVectorizer(
            analyzer='char_wb', # n-grammi all'interno dei confini di parola (prefissi, radici, desinenze)
            ngram_range=self.char_ngram_range,
            max_features=self.max_char_features,
            sublinear_tf=True,
            min_df=5
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

    def _compute_fertility(self, texts):
        """
        Calcola il Tokenizer Fertility Ratio per ciascuna frase:
        fertility = numero di token generati da BERT / numero di parole
        """
        fertilities = []
        for text in texts:
            words = [w for w in re.findall(r'\b\w+\b', text) if w]
            num_words = max(len(words), 1)
            # Token subword generati dal tokenizer di BERT (escludendo [CLS] e [SEP])
            tokens = self.tokenizer.tokenize(text)
            num_tokens = max(len(tokens), 1)
            fertilities.append(num_tokens / num_words)
        return np.array(fertilities, dtype=np.float32).reshape(-1, 1)

    def fit(self, texts):
        """
        Addestra il vocabolario TF-IDF di caratteri e lo scaler sui testi di training.
        """
        print(f"Fit Feature Extractor su {len(texts):,} frasi...")
        char_matrix = self.char_tfidf.fit_transform(texts).toarray()
        fertility_feats = self._compute_fertility(texts)
        
        # Concateniamo char tfidf + fertility (totale: 501 feature)
        combined_feats = np.hstack([char_matrix, fertility_feats])
        self.scaler.fit(combined_feats)
        self.is_fitted = True
        print(f"-> Extractor pronto con {combined_feats.shape[1]} feature ausiliarie.")
        return self

    def transform(self, texts):
        """
        Estrae il tensore PyTorch di feature ausiliarie normalizzate per un batch di testi.
        """
        if not self.is_fitted:
            raise RuntimeError("L'estrattore deve essere addestrato (fit) prima di invocare transform!")
        
        char_matrix = self.char_tfidf.transform(texts).toarray()
        fertility_feats = self._compute_fertility(texts)
        combined_feats = np.hstack([char_matrix, fertility_feats])
        scaled_feats = self.scaler.transform(combined_feats)
        return torch.tensor(scaled_feats, dtype=torch.float32)

    def fit_transform(self, texts):
        return self.fit(texts).transform(texts)
