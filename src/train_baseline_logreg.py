"""
Addestramento e Salvataggio della Baseline 1: Logistic Regression + TF-IDF (VarDial 2022).
Addestrata sullo stesso dataset neurale bilanciato 'wiki-dump/Dataset/train_neural_balanced.csv' (200.000 campioni)
per un confronto equo e ad armi pari con Baseline 2 (BERT) e Modello Proposto (Feature Fusion).

Salva l'intero bundle (modello, vectorizer, scaler, label encoder) in:
    checkpoints/baseline_logreg.joblib
"""

import os
import time
import argparse
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

def train_and_save_baseline(
    train_path='wiki-dump/Dataset/train_neural_balanced.csv',
    val_path='wiki-dump/Dataset/val_sentences.csv',
    save_path='checkpoints/baseline_logreg.joblib',
    cm_plot_path='checkpoints/cm_baseline_logreg.png'
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print("=== AVVIO ADDESTRAMENTO BASELINE 1: LOGISTIC REGRESSION + TF-IDF ===")
    
    # 1. Caricamento Dati
    print(f"1. Caricamento dati da '{train_path}' e '{val_path}'...")
    train_df = pd.read_csv(train_path, engine='python', on_bad_lines='skip')
    val_df = pd.read_csv(val_path, engine='python', on_bad_lines='skip')
    
    # Pulizia valori nulli o non stringa
    train_df['text'] = train_df['text'].fillna('').astype(str)
    val_df['text'] = val_df['text'].fillna('').astype(str)
    
    print(f"-> Train samples: {len(train_df):,}")
    print(f"-> Val samples:   {len(val_df):,}")

    y_train = train_df['dial'].values
    y_val = val_df['dial'].values

    # 2. Vettorizzazione TF-IDF (n-grammi 1-2, sublinear_tf=True, min_df=3)
    t0_tfidf = time.time()
    print("\n2. Estrazione feature TF-IDF (1-2 grammi, max 50,000)...")
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=50000,
        sublinear_tf=True,
        min_df=3
    )
    X_train_tfidf = tfidf.fit_transform(train_df['text'])
    X_val_tfidf = tfidf.transform(val_df['text'])
    print(f"-> TF-IDF completato in {time.time() - t0_tfidf:.1f}s")
    print(f"   Shape Train: {X_train_tfidf.shape} | Non-zero: {X_train_tfidf.nnz:,}")

    # 3. Standard Scaling Sparse-Safe (with_mean=False preserva la RAM)
    t0_scale = time.time()
    print("\n3. Scaling varianza con StandardScaler(with_mean=False)...")
    scaler = StandardScaler(with_mean=False)
    X_train_scaled = scaler.fit_transform(X_train_tfidf)
    X_val_scaled = scaler.transform(X_val_tfidf)
    print(f"-> Scaling completato in {time.time() - t0_scale:.1f}s")

    # 4. Addestramento Logistic Regression (solver='sag', class_weight='balanced')
    t0_train = time.time()
    print("\n4. Addestramento Logistic Regression (solver='sag', class_weight='balanced')...")
    clf = LogisticRegression(
        solver='sag',
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    clf.fit(X_train_scaled, y_train)
    print(f"-> Addestramento completato in {time.time() - t0_train:.1f}s")

    # 5. Salvataggio del bundle completo per inferenza / test
    bundle = {
        'model': clf,
        'tfidf': tfidf,
        'scaler': scaler,
        'classes': clf.classes_
    }
    joblib.dump(bundle, save_path)
    print(f"\n-> Modello salvato con successo in '{save_path}' ({os.path.getsize(save_path) / (1024*1024):.2f} MB)")

    # 6. Valutazione sul Validation Set
    print("\n5. Valutazione sul Validation Set...")
    y_pred = clf.predict(X_val_scaled)
    macro_f1 = f1_score(y_val, y_pred, average='macro')
    weighted_f1 = f1_score(y_val, y_pred, average='weighted')
    acc = accuracy_score(y_val, y_pred)

    print("=" * 65)
    print(f" RISULTATI BASELINE 1 (LOGISTIC REGRESSION + TF-IDF) [200k train]")
    print("=" * 65)
    print(f"-> MACRO F1-SCORE:    {macro_f1:.4f}  (metrica ufficiale di riferimento)")
    print(f"-> WEIGHTED F1-SCORE: {weighted_f1:.4f}")
    print(f"-> ACCURACY:          {acc:.4f}")
    print("=" * 65)
    print("\n--- CLASSIFICATION REPORT DETTAGLIATO ---")
    print(classification_report(y_val, y_pred, digits=4))

    # 7. Salvataggio Confusion Matrix
    labels = sorted(list(set(y_val)))
    cm = confusion_matrix(y_val, y_pred, labels=labels, normalize='true')
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues', ax=ax, values_format='.2f', xticks_rotation=45)
    plt.title("Confusion Matrix Normalizzata - Baseline Logistic Regression (200k)")
    plt.tight_layout()
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"-> Matrice di confusione salvata in '{cm_plot_path}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='wiki-dump/Dataset/train_neural_balanced.csv')
    parser.add_argument('--val_path', type=str, default='wiki-dump/Dataset/val_sentences.csv')
    parser.add_argument('--save_path', type=str, default='checkpoints/baseline_logreg.joblib')
    args = parser.parse_args()

    train_and_save_baseline(
        train_path=args.train_path,
        val_path=args.val_path,
        save_path=args.save_path
    )
