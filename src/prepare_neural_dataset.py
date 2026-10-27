"""
Script per il campionamento bilanciato del dataset di addestramento per i modelli neurali (BERT).
Target: 10.000 frasi per ciascuna delle 12 varietà linguistiche (120.000 frasi totali).

Strategia:
- Per le classi con >= 10.000 campioni (11 classi su 12): Downsampling casuale a 10.000 frasi.
- Per la classe con < 10.000 campioni ('eml' con ~7.506 campioni): Random Oversampling (con reinserimento)
  per raggiungere 10.000 campioni senza introdurre rumore sintetico non dialettale.
- Generazione di grafici comparativi per documentare e giustificare la distribuzione pre e post bilanciamento.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def balance_dataset(
    train_path='wiki-dump/Dataset/train_sentences.csv',
    output_path='wiki-dump/Dataset/train_neural_balanced.csv',
    plot_path='balancing_distribution_comparison.png',
    target_per_class=10000,
    random_state=42
):
    print(f"1. Caricamento del dataset da '{train_path}'...")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"File {train_path} non trovato!")
    
    df_train = pd.read_csv(train_path)
    print(f"-> Totale frasi caricate: {len(df_train):,}")
    
    orig_counts = df_train['dial'].value_counts()
    print("\n--- DISTRIBUZIONE ORIGINALE ---")
    for dial, count in orig_counts.items():
        print(f"{dial:>10}: {count:>10,} ({count/len(df_train)*100:6.2f}%)")
    
    # 2. Applicazione di Downsampling ed Oversampling
    print(f"\n2. Bilanciamento con target {target_per_class:,} frasi per classe...")
    balanced_dfs = []
    
    for dial in orig_counts.index:
        df_dial = df_train[df_train['dial'] == dial]
        n_samples = len(df_dial)
        
        if n_samples >= target_per_class:
            # Downsampling
            sampled = df_dial.sample(n=target_per_class, random_state=random_state)
            print(f"[{dial:>8}] Downsampling da {n_samples:>10,} -> {target_per_class:,}")
        else:
            # Random Oversampling (sampling con replacement)
            # Motivazione SOTA: garantisce autenticità fonetica/sintattica dialettale evitando
            # artefatti spuri di traduzione automatica/sinonimi italiani
            sampled = df_dial.sample(n=target_per_class, replace=True, random_state=random_state)
            print(f"[{dial:>8}] Random Oversampling da {n_samples:>6,} -> {target_per_class:,}")
        
        balanced_dfs.append(sampled)
    
    df_balanced = pd.concat(balanced_dfs, ignore_index=True)
    # Shuffle dell'intero dataset bilanciato
    df_balanced = df_balanced.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    
    balanced_counts = df_balanced['dial'].value_counts()
    print(f"\n-> Dataset bilanciato creato: {len(df_balanced):,} frasi totali.")
    
    # 3. Salvataggio su disco
    print(f"3. Salvataggio su '{output_path}'...")
    df_balanced[['text', 'dial']].to_csv(output_path, index=False)
    print(f"-> File salvato con successo ({os.path.getsize(output_path) / (1024*1024):.2f} MB)")
    
    # 4. Generazione del grafico comparativo pre vs post bilanciamento
    print(f"4. Generazione del grafico comparativo in '{plot_path}'...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Ordiniamo le classi per frequenza originale
    dial_order = orig_counts.index.tolist()
    
    # Subplot 1: Distribuzione Originale (scala logaritmica)
    ax1 = axes[0]
    bars1 = ax1.bar(dial_order, [orig_counts[d] for d in dial_order], color='#34495e', edgecolor='black', alpha=0.85)
    bars1[0].set_color('#e74c3c') # Evidenzia l'italiano standard
    ax1.set_yscale('log')
    ax1.set_title("Distribuzione Originale Train Set (Scala Logaritmica)", fontsize=13, fontweight='bold', pad=12)
    ax1.set_ylabel("Numero di Frasi (log)", fontsize=11)
    ax1.set_xlabel("Varietà Linguistica / Dialetto", fontsize=11)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.tick_params(axis='x', rotation=45)
    
    # Annotazioni sui conteggi originali
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width()/2.0,
            yval * 1.15,
            f"{yval/1000:.1f}k" if yval >= 1000 else str(yval),
            ha='center', va='bottom', fontsize=9, rotation=30
        )
    
    # Subplot 2: Distribuzione Target Bilanciata (scala lineare)
    ax2 = axes[1]
    bars2 = ax2.bar(dial_order, [balanced_counts[d] for d in dial_order], color='#27ae60', edgecolor='black', alpha=0.85)
    ax2.set_ylim(0, 12000)
    ax2.set_title("Distribuzione Bilanciata per Modelli Neurali (10k per classe)", fontsize=13, fontweight='bold', pad=12)
    ax2.set_ylabel("Numero di Frasi", fontsize=11)
    ax2.set_xlabel("Varietà Linguistica / Dialetto", fontsize=11)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    ax2.tick_params(axis='x', rotation=45)
    
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width()/2.0,
            yval + 200,
            f"{yval:,}",
            ha='center', va='bottom', fontsize=9
        )
        
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print("-> Grafico generato e salvato con successo.")

if __name__ == '__main__':
    balance_dataset()
