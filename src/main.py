"""
Pipeline di addestramento e validazione per i modelli Transformer:
- Baseline Transformer (BERT puro)
- Modello con Feature Fusion (BERT + Feature Morfologiche/Simboliche)

Ottimizzato per GPU con Mixed Precision (AMP), gradient clipping, AdamW e scheduler lineare.
Calcola Macro F1, Weighted F1 e salva il best model checkpoint.
"""

import os
import argparse
import joblib
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from feature_extraction import DialectAuxiliaryFeatureExtractor
from models import BaselineBERT, DialectFeatureFusionBERT
from dataset import DialectDataset, DIALECT_CLASSES
from training_functions import fit, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, choices=['baseline', 'fusion'], required=True,
                        help="Tipo di modello: 'baseline' (BERT puro) o 'fusion' (BERT + Aux Features)")
    parser.add_argument('--model_name', type=str, default='dbmdz/bert-base-italian-cased')
    parser.add_argument('--train_path', type=str, default='wiki-dump/Dataset/train_neural_balanced.csv')
    parser.add_argument('--val_path', type=str, default='wiki-dump/Dataset/val_sentences.csv')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_len', type=int, default=70)
    parser.add_argument('--lr_bert', type=float, default=2e-5)
    parser.add_argument('--lr_aux', type=float, default=1e-3)
    parser.add_argument('--val_subset', type=int, default=50000,
                        help="Numero max di campioni di validation per velocizzare i controlli ad ogni epoca")
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== AVVIO TRAINING [{args.model_type.upper()}] SU DEVICE: {device} ===")

    # 1. Caricamento tokenizer e dati
    print(f"1. Inizializzazione tokenizer '{args.model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print(f"2. Caricamento dati da '{args.train_path}' e '{args.val_path}'...")
    train_df = pd.read_csv(args.train_path)
    val_df = pd.read_csv(args.val_path)

    # Subsample controllato per la validazione durante le epoche (stratificato per dialetto)
    if args.val_subset and len(val_df) > args.val_subset:
        n_per_class = args.val_subset // len(DIALECT_CLASSES)
        val_df = pd.concat(
            [g.sample(n=min(len(g), n_per_class), random_state=42) for _, g in val_df.groupby('dial')],
            ignore_index=True
        )
    print(f"-> Train: {len(train_df):,} campioni | Val (subsample): {len(val_df):,} campioni")

    train_texts = train_df['text'].tolist()
    train_labels = train_df['dial'].tolist()
    val_texts = val_df['text'].tolist()
    val_labels = val_df['dial'].tolist()

    # 2. Estrazione Feature Ausiliarie (se fusion)
    train_aux = None
    val_aux = None
    aux_dim = 0
    is_fusion = (args.model_type == 'fusion')

    if is_fusion:
        print("\n--- ESTRAZIONE FEATURE AUSILIARIE (CHAR N-GRAMS + FERTILITY) ---")
        extractor = DialectAuxiliaryFeatureExtractor(tokenizer=tokenizer, max_char_features=500)
        train_aux = extractor.fit_transform(train_texts)
        val_aux = extractor.transform(val_texts)
        aux_dim = train_aux.shape[1]
        print(f"-> Feature ausiliarie estratte: dimensione = {aux_dim}")
        # Salva l'estrattore (vocabolari TF-IDF + scaler): serve per l'inferenza
        extractor_path = os.path.join(args.save_dir, "fusion_feature_extractor.joblib")
        joblib.dump(extractor, extractor_path)
        print(f"-> Estrattore salvato in '{extractor_path}'")

    train_dataset = DialectDataset(train_texts, train_labels, tokenizer, max_len=args.max_len, aux_features=train_aux)
    val_dataset = DialectDataset(val_texts, val_labels, tokenizer, max_len=args.max_len, aux_features=val_aux)

    # drop_last=True: BatchNorm1d (ramo fusion) fallisce con un batch da 1 campione
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=2, pin_memory=True)

    # 3. Creazione Modello
    if is_fusion:
        model = DialectFeatureFusionBERT(model_name=args.model_name, num_classes=len(DIALECT_CLASSES), aux_dim=aux_dim)
        optimizer_params = [
            {'params': model.encoder.parameters(), 'lr': args.lr_bert},
            {'params': model.aux_projector.parameters(), 'lr': args.lr_aux},
            {'params': model.classifier.parameters(), 'lr': args.lr_aux}
        ]
    else:
        model = BaselineBERT(model_name=args.model_name, num_classes=len(DIALECT_CLASSES))
        optimizer_params = [
            {'params': model.encoder.parameters(), 'lr': args.lr_bert},
            {'params': model.classifier.parameters(), 'lr': args.lr_aux}
        ]

    # 4. Model training setup
    model.to(device)
    optimizer = torch.optim.AdamW(optimizer_params, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    # 5. Training-validation loop
    best_macro_f1 = 0.0
    best_checkpoint = os.path.join(args.save_dir, f"best_{args.model_type}_model.pt")

    print("\n=== INIZIO ADDESTRAMENTO ===")

    for epoch in range(1, args.epochs + 1):
        # train step (un'intera epoca)
        train_loss, elapsed = fit(model, device, optimizer, scheduler, criterion, scaler, train_loader, is_fusion)
        print(f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} | Tempo: {elapsed:.1f}s")

        # val step
        val_loss, macro_f1, weighted_f1, acc, _, _ = evaluate(model, val_loader, device, criterion, is_fusion)
        print(f"\n---> FINE EPOCA {epoch} | Val Loss: {val_loss:.4f} | MACRO F1: {macro_f1:.4f} "
              f"| Weighted F1: {weighted_f1:.4f} | Acc: {acc:.4f}")

        # checkpoint
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            torch.save(model.state_dict(), best_checkpoint)
            print(f"*** Nuovo miglior modello salvato con Macro F1: {best_macro_f1:.4f} in '{best_checkpoint}' ***\n")

    print(f"\n=== TRAINING COMPLETATO [{args.model_type.upper()}] | Best Macro F1: {best_macro_f1:.4f} ===")


if __name__ == '__main__':
    main()