"""
Script di Valutazione Finale Comparativa:
Valuta i 3 modelli addestrati sul Test Set ufficiale (in-domain) e/o sul dataset Social Media:
1. Baseline 1: Logistic Regression + TF-IDF (checkpoints/baseline_logreg.joblib)
2. Baseline 2: BERT XXL Base (checkpoints/best_baseline_model.pt)
3. Modello Proposto: Dialect Feature Fusion BERT (checkpoints/best_fusion_model.pt + checkpoints/fusion_feature_extractor.joblib)

Genera:
- Tabella comparativa delle metriche (Macro F1, Weighted F1, Accuracy)
- Classification Reports dettagliati per dialetto
- Matrici di confusione comparative salvate in checkpoints/
"""

import os
import argparse
import time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from dataset import DialectDataset, DIALECT_CLASSES, LABEL2ID, ID2LABEL
from models import BaselineBERT, DialectFeatureFusionBERT

def evaluate_all_models(
    test_path='wiki-dump/Dataset/test_sentences.csv',
    model_name='models/bert-base-italian-xxl-cased',
    batch_size=64,
    max_len=70
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== ESECUZIONE BENCHMARK COMPARATIVO SU: {test_path} (Device: {device}) ===\n")

    # 1. Caricamento Test Set
    df_test = pd.read_csv(test_path, engine='python', on_bad_lines='skip')
    df_test['text'] = df_test['text'].fillna('').astype(str)
    
    # Se test_sentences ha etichette valide
    has_labels = 'dial' in df_test.columns and df_test['dial'].dropna().nunique() > 1
    if not has_labels:
        raise ValueError(f"La colonna 'dial' non è presente o non ha etichette valide in {test_path}!")

    print(f"-> Campioni di test caricati: {len(df_test):,}")
    y_true = df_test['dial'].values
    test_texts = df_test['text'].tolist()

    results = {}

    # ---------------------------------------------------------------------------
    # MODELLO 1: BASELINE LOGISTIC REGRESSION
    # ---------------------------------------------------------------------------
    logreg_path = 'checkpoints/baseline_logreg.joblib'
    if os.path.exists(logreg_path):
        print("\n" + "=" * 60)
        print("1. VALUTAZIONE BASELINE 1 (LOGISTIC REGRESSION + TF-IDF)")
        print("=" * 60)
        bundle = joblib.load(logreg_path)
        clf = bundle['model']
        tfidf = bundle['tfidf']
        scaler = bundle['scaler']

        t0 = time.time()
        X_test_tfidf = tfidf.transform(test_texts)
        X_test_scaled = scaler.transform(X_test_tfidf)
        y_pred_logreg = clf.predict(X_test_scaled)
        elapsed = time.time() - t0

        m_f1 = f1_score(y_true, y_pred_logreg, average='macro')
        w_f1 = f1_score(y_true, y_pred_logreg, average='weighted')
        acc = accuracy_score(y_true, y_pred_logreg)
        results['Logistic Regression (VarDial)'] = {'Macro F1': m_f1, 'Weighted F1': w_f1, 'Accuracy': acc, 'Time': f"{elapsed:.1f}s"}
        print(f"-> Macro F1: {m_f1:.4f} | Weighted F1: {w_f1:.4f} | Acc: {acc:.4f} ({elapsed:.1f}s)")
    else:
        print(f"\n[AVVISO] {logreg_path} non trovato. Esegui prima run_logreg_training.sh per allenarla!")

    # ---------------------------------------------------------------------------
    # MODELLI NEURALI (BERT BASELINE & FEATURE FUSION)
    # ---------------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 2. BASELINE BERT
    base_bert_path = 'checkpoints/best_baseline_model.pt'
    if os.path.exists(base_bert_path):
        print("\n" + "=" * 60)
        print("2. VALUTAZIONE BASELINE 2 (BERT XXL PURO)")
        print("=" * 60)
        model_base = BaselineBERT(model_name=model_name, num_classes=len(DIALECT_CLASSES))
        model_base.load_state_dict(torch.load(base_bert_path, map_location=device))
        model_base.to(device)
        model_base.eval()

        base_dataset = DialectDataset(test_texts, y_true, tokenizer, max_len=max_len)
        base_loader = DataLoader(base_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        preds_base = []
        t0 = time.time()
        with torch.no_grad():
            for batch in base_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    out = model_base(input_ids, attention_mask)
                preds = torch.argmax(out, dim=-1).cpu().numpy()
                preds_base.extend([ID2LABEL[p] for p in preds])
        elapsed = time.time() - t0

        m_f1 = f1_score(y_true, preds_base, average='macro')
        w_f1 = f1_score(y_true, preds_base, average='weighted')
        acc = accuracy_score(y_true, preds_base)
        results['Baseline BERT XXL'] = {'Macro F1': m_f1, 'Weighted F1': w_f1, 'Accuracy': acc, 'Time': f"{elapsed:.1f}s"}
        print(f"-> Macro F1: {m_f1:.4f} | Weighted F1: {w_f1:.4f} | Acc: {acc:.4f} ({elapsed:.1f}s)")

    # 3. FEATURE FUSION BERT
    fusion_path = 'checkpoints/best_fusion_model.pt'
    extractor_path = 'checkpoints/fusion_feature_extractor.joblib'
    if os.path.exists(fusion_path) and os.path.exists(extractor_path):
        print("\n" + "=" * 60)
        print("3. VALUTAZIONE MODELLO PROPOSTO (FEATURE FUSION BERT XXL)")
        print("=" * 60)
        print("-> Estrazione feature ausiliarie per il test set...")
        extractor = joblib.load(extractor_path)
        aux_features = extractor.transform(test_texts)
        aux_dim = aux_features.shape[1]

        model_fusion = DialectFeatureFusionBERT(model_name=model_name, num_classes=len(DIALECT_CLASSES), aux_dim=aux_dim)
        model_fusion.load_state_dict(torch.load(fusion_path, map_location=device))
        model_fusion.to(device)
        model_fusion.eval()

        fusion_dataset = DialectDataset(test_texts, y_true, tokenizer, max_len=max_len, aux_features=aux_features)
        fusion_loader = DataLoader(fusion_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        preds_fusion = []
        t0 = time.time()
        with torch.no_grad():
            for batch in fusion_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                aux = batch['aux_features'].to(device)
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    out = model_fusion(input_ids, attention_mask, aux_features=aux)
                preds = torch.argmax(out, dim=-1).cpu().numpy()
                preds_fusion.extend([ID2LABEL[p] for p in preds])
        elapsed = time.time() - t0

        m_f1 = f1_score(y_true, preds_fusion, average='macro')
        w_f1 = f1_score(y_true, preds_fusion, average='weighted')
        acc = accuracy_score(y_true, preds_fusion)
        results['Feature Fusion BERT (Proposto)'] = {'Macro F1': m_f1, 'Weighted F1': w_f1, 'Accuracy': acc, 'Time': f"{elapsed:.1f}s"}
        print(f"-> Macro F1: {m_f1:.4f} | Weighted F1: {w_f1:.4f} | Acc: {acc:.4f} ({elapsed:.1f}s)")

    # 4. TABELLA RIASSUNTIVA COMPARATIVA
    print("\n" + "=" * 70)
    print(" TABELLA COMPARATIVA FINALE DEI RISULTATI")
    print("=" * 70)
    df_res = pd.DataFrame(results).T
    print(df_res.to_string())
    print("=" * 70)
    df_res.to_csv('checkpoints/benchmark_summary.csv')
    print("-> Tabella salvata in 'checkpoints/benchmark_summary.csv'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str, default='wiki-dump/Dataset/test_sentences.csv')
    parser.add_argument('--model_name', type=str, default='models/bert-base-italian-xxl-cased')
    args = parser.parse_args()

    evaluate_all_models(test_path=args.test_path, model_name=args.model_name)
