"""
Modulo per l'Explainability e l'Analisi del Code-Switching (SHAP + Saliency):
Fornisce funzioni per:
1. Wrap predittivo per SHAP (accetta liste di stringhe e restituisce probabilità softmax)
   sia per BaselineBERT che per DialectFeatureFusionBERT.
2. Token Attribution con Integrated Gradients / Embedding Saliency.
3. Ispezione dei pesi del ramo ausiliario (938 feature simboliche).
4. Render grafico e HTML per notebook e report scientifico.
"""

import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from dataset import DIALECT_CLASSES, LABEL2ID, ID2LABEL
from models import BaselineBERT, DialectFeatureFusionBERT

class ModelPredictWrapper:
    """
    Wrapper callable compatibile con SHAP (shap.Explainer / shap.maskers.Text):
    Prende in input un array/lista di stringhe di testo e restituisce un array numpy
    di shape (N, num_classes) contenente le probabilità softmax.
    """
    def __init__(self, model, tokenizer, device, is_fusion=False, extractor=None, max_len=70, batch_size=32):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.is_fusion = is_fusion
        self.extractor = extractor
        self.max_len = max_len
        self.batch_size = batch_size
        self.model.eval()

    def __call__(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        elif isinstance(texts, np.ndarray):
            texts = texts.tolist()

        all_probs = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            encoding = self.tokenizer(
                batch_texts,
                truncation=True,
                padding='max_length',
                max_length=self.max_len,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)

            kwargs = {}
            if self.is_fusion:
                if self.extractor is None:
                    raise RuntimeError("Extractor non fornito per il modello fusion!")
                aux_tensor = self.extractor.transform(batch_texts).to(self.device)
                kwargs['aux_features'] = aux_tensor

            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
                    logits = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(probs)

        return np.vstack(all_probs)

def inspect_auxiliary_features(model_fusion, extractor, text, top_k=15):
    """
    Analizza il contributo delle feature simboliche del ramo ausiliario per una specifica frase.
    Calcola l'attivazione prima della proiezione densa e identifica le feature più salienti.
    """
    model_fusion.eval()
    feat_names = extractor.get_feature_names_out()
    aux_tensor = extractor.transform([text]) # shape (1, 938)
    
    # Valore delle feature grezze normalizzate
    vals = aux_tensor[0].cpu().numpy()
    
    # Pesi della prima proiezione lineare del ramo ausiliario
    # aux_projector[0] è nn.Linear(aux_dim, aux_proj_dim)
    proj_weight = model_fusion.aux_projector[0].weight.detach().cpu().numpy() # (128, 938)
    
    # Importanza aggregata per feature = valore * media dell'intensità di proiezione
    mean_impact = vals * np.abs(proj_weight).mean(axis=0)
    
    # Top K indici per impatto assoluto
    top_indices = np.argsort(np.abs(mean_impact))[::-1][:top_k]
    
    records = []
    for idx in top_indices:
        records.append({
            'Feature': feat_names[idx],
            'Valore Normalizzato': float(vals[idx]),
            'Impatto Proiettato': float(mean_impact[idx])
        })
    return pd.DataFrame(records)

def highlight_tokens_html(tokens, scores, title="Token Attribution"):
    """
    Genera una stringa HTML per visualizzare le parole colorate in base all'attribuzione:
    Verde = attribuzione positiva (spinge verso il dialetto target),
    Rosso = attribuzione negativa (spinge lontano dal dialetto target).
    """
    max_score = max(max(abs(s) for s in scores), 1e-5)
    html_spans = []
    for token, score in zip(tokens, scores):
        norm_score = score / max_score
        if norm_score > 0:
            color = f"rgba(46, 204, 113, {min(abs(norm_score), 0.85):.2f})" # verde
        else:
            color = f"rgba(231, 76, 60, {min(abs(norm_score), 0.85):.2f})" # rosso
        
        # Gestione subword token di BERT (##)
        clean_token = token.replace("##", "")
        html_spans.append(
            f"<span style='background-color: {color}; padding: 2px 4px; margin: 1px; "
            f"border-radius: 3px; display: inline-block;' title='Score: {score:.4f}'>{clean_token}</span>"
        )
    
    html_output = f"""
    <div style='font-family: Arial, sans-serif; line-height: 1.8; padding: 12px; border: 1px solid #ddd; border-radius: 8px;'>
        <h4 style='margin-top: 0; color: #333;'>{title}</h4>
        <div>{' '.join(html_spans)}</div>
        <div style='margin-top: 10px; font-size: 11px; color: #777;'>
            <span style='color: #27ae60; font-weight: bold;'>■ Verde:</span> Contributo positivo verso la classe predetta | 
            <span style='color: #c0392b; font-weight: bold;'>■ Rosso:</span> Contributo negativo
        </div>
    </div>
    """
    return html_output
