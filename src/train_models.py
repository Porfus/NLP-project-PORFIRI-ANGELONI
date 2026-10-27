"""
Pipeline di addestramento e validazione per i modelli Transformer:
- Baseline Transformer (BERT puro)
- Modello con Feature Fusion (BERT + Feature Morfologiche/Simboliche)

Ottimizzato per GPU con Mixed Precision (AMP), gradient clipping, AdamW e scheduler lineare.
Calcola Macro F1, Weighted F1 e salva il best model checkpoint.
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score, classification_report

from feature_extraction import DialectAuxiliaryFeatureExtractor
from models import BaselineBERT, DialectFeatureFusionBERT

DIALECT_CLASSES = sorted([
    'eml', 'fur', 'it', 'lij', 'lld', 'lmo',
    'nap', 'pms', 'roa_tara', 'sc', 'scn', 'vec'
])
LABEL2ID = {d: i for i, d in enumerate(DIALECT_CLASSES)}
ID2LABEL = {i: d for i, d in enumerate(DIALECT_CLASSES)}

class DialectDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=70, aux_features=None):
        self.texts = texts
        self.labels = [LABEL2ID[l] for l in labels]
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.aux_features = aux_features

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }
        if self.aux_features is not None:
            item['aux_features'] = self.aux_features[idx]
        return item

def evaluate(model, val_loader, device, criterion, is_fusion=False):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            kwargs = {}
            if is_fusion:
                kwargs['aux_features'] = batch['aux_features'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    acc = accuracy_score(all_labels, all_preds)

    return avg_loss, macro_f1, weighted_f1, acc, all_preds, all_labels

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

    # Subsample controllato per la validazione durante le epoche
    if args.val_subset and len(val_df) > args.val_subset:
        val_df = val_df.groupby('dial', group_keys=False).apply(
            lambda x: x.sample(min(len(x), int(args.val_subset / 12)), random_state=42)
        ).reset_index(drop=True)
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

    train_dataset = DialectDataset(train_texts, train_labels, tokenizer, max_len=args.max_len, aux_features=train_aux)
    val_dataset = DialectDataset(val_texts, val_labels, tokenizer, max_len=args.max_len, aux_features=val_aux)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=2, pin_memory=True)

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

    model.to(device)
    optimizer = torch.optim.AdamW(optimizer_params, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    # 4. Loop di addestramento
    best_macro_f1 = 0.0
    best_checkpoint = os.path.join(args.save_dir, f"best_{args.model_type}_model.pt")

    print("\n=== INIZIO ADDESTRAMENTO ===")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0
        start_time = time.time()

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            kwargs = {}
            if is_fusion:
                kwargs['aux_features'] = batch['aux_features'].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_train_loss += loss.item()
            if step % 200 == 0 or step == len(train_loader):
                elapsed = time.time() - start_time
                print(f"Epoch {epoch}/{args.epochs} | Step {step}/{len(train_loader)} | Loss: {loss.item():.4f} | Tempo: {elapsed:.1f}s")

        # Validazione a fine epoca
        val_loss, macro_f1, weighted_f1, acc, _, _ = evaluate(model, val_loader, device, criterion, is_fusion=is_fusion)
        print(f"\n---> FINE EPOCA {epoch} | Val Loss: {val_loss:.4f} | MACRO F1: {macro_f1:.4f} | Weighted F1: {weighted_f1:.4f} | Acc: {acc:.4f}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            torch.save(model.state_dict(), best_checkpoint)
            print(f"*** Nuovo miglior modello salvato con Macro F1: {best_macro_f1:.4f} in '{best_checkpoint}' ***\n")

    print(f"\n=== TRAINING COMPLETATO [{args.model_type.upper()}] | Best Macro F1: {best_macro_f1:.4f} ===")

if __name__ == '__main__':
    main()
