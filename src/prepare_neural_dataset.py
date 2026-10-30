"""
Script per il campionamento matematico del dataset di addestramento neurale tramite
Two-Stage Sampling:
1. Cap iniziale sull'Italiano standard (it_cap = 150.000 frasi, commisurato al dialetto massimo lmo a 125k)
2. Temperature / Power-Law Sampling (alpha = 0.5, total_samples = 200.000 frasi):
   p_i = (N_i_capped)^alpha / sum_j (N_j_capped)^alpha

Vantaggi metodologici:
- L'italiano rimane la classe #1 (~33k frasi, ~16.5% del dataset), preservando la prior naturale.
- L'83.5% del dataset e dei gradienti della GPU è dedicato alla discriminazione fine-grained tra gli 11 dialetti.
- Zero frasi duplicate (tutti i 7.378 campioni dell'emiliano sono autentici e unici).
- Generazione di grafici comparativi a 3 pannelli.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def temperature_sample_dataset(
    train_path='NLP_project/wiki-dump/Dataset/train_sentences.csv',
    output_path='NLP_project/wiki-dump/Dataset/train_neural_balanced.csv',
    plot_path='temperature_sampling_analysis.png',
    it_cap=150000,
    alpha=0.5,
    total_samples=200000,
    random_state=42
):
    print(f"=== ESECUZIONE TWO-STAGE TEMPERATURE SAMPLING ===")
    print(f"-> Parametri: it_cap = {it_cap:,} | alpha = {alpha} | Total = {total_samples:,}")
    print(f"1. Caricamento del dataset da '{train_path}'...")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"File {train_path} non trovato!")

    df_train = pd.read_csv(train_path, engine='python', on_bad_lines='skip')
    print(f"-> Totale frasi caricate: {len(df_train):,}")

    orig_counts = df_train['dial'].value_counts()
    print("\n--- DISTRIBUZIONE ORIGINALE ---")
    for dial, count in orig_counts.items():
        print(f"{dial:>10}: {count:>10,} ({count/len(df_train)*100:6.2f}%)")

    # 1. Applicazione del Cap iniziale sull'Italiano nel pool
    capped_counts = orig_counts.copy()
    if capped_counts['it'] > it_cap:
        capped_counts['it'] = it_cap
        print(f"\n-> Cap preventivo su 'it': ridotto da {orig_counts['it']:,} a {it_cap:,} frasi nel pool iniziale.")

    # 2. Calcolo dei pesi matematici con Temperature Sampling (alpha = 0.5)
    weights = {k: capped_counts[k]**alpha for k in capped_counts.index}
    sum_w = sum(weights.values())
    probs = {k: weights[k] / sum_w for k in capped_counts.index}

    allocations = {}
    for k, p in probs.items():
        alloc = int(round(p * total_samples))
        # Non possiamo eccedere i campioni unici originali disponibili
        alloc = min(alloc, orig_counts[k])
        allocations[k] = alloc

    # Eventuale aggiustamento di arrotondamento sull'italiano
    diff = total_samples - sum(allocations.values())
    if diff != 0:
        allocations['it'] += diff

    print(f"\n--- ALLOCAZIONE FINALE CON TWO-STAGE SAMPLING (alpha = {alpha}) ---")
    for dial in orig_counts.index:
        count = allocations[dial]
        pct = count / total_samples * 100
        print(f"{dial:>10}: {count:>8,} ({pct:5.2f}%) [Pool cappato: {capped_counts[dial]:>8,}]")

    # 3. Estrazione effettiva dei campioni (senza rimpiazzo, 100% campioni unici)
    sampled_dfs = []
    for dial in orig_counts.index:
        n_samples = allocations[dial]
        df_dial = df_train[df_train['dial'] == dial]
        sampled = df_dial.sample(n=n_samples, replace=False, random_state=random_state)
        sampled_dfs.append(sampled)

    df_balanced = pd.concat(sampled_dfs, ignore_index=True)
    # Shuffle casuale
    df_balanced = df_balanced.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    print(f"\n-> Dataset finale campionato generato: {len(df_balanced):,} frasi totali.")
    print(f"3. Salvataggio in '{output_path}'...")
    df_balanced[['text', 'dial']].to_csv(output_path, index=False)
    print(f"-> File salvato con successo ({os.path.getsize(output_path) / (1024*1024):.2f} MB)")

    # 4. Generazione della visualizzazione a 3 pannelli
    print(f"4. Generazione dei grafici comparativi in '{plot_path}'...")
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    dial_order = orig_counts.index.tolist()

    # Pannello 1: Distribuzione Originale (Log-Scale)
    ax1 = axes[0]
    bars1 = ax1.bar(dial_order, [orig_counts[d] for d in dial_order], color='#2c3e50', edgecolor='black', alpha=0.85)
    bars1[0].set_color('#c0392b')
    ax1.set_yscale('log')
    ax1.set_title("1. Train Set Originale (Log-Scale)\n(Italiano al 89.7% = 3.66M)", fontsize=12, fontweight='bold', pad=10)
    ax1.set_ylabel("Numero di Frasi (log)", fontsize=10)
    ax1.set_xlabel("Varietà Linguistica", fontsize=10)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.tick_params(axis='x', rotation=45)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval * 1.15,
                 f"{yval/1000:.1f}k" if yval >= 1000 else str(yval),
                 ha='center', va='bottom', fontsize=8, rotation=25)

    # Pannello 2: Curve di Percentuale al Variare di Alpha (con Cap IT a 150k)
    ax2 = axes[1]
    alphas_to_plot = [0.2, 0.35, 0.5, 0.7, 1.0]
    for a in alphas_to_plot:
        w_a = {k: capped_counts[k]**a for k in dial_order}
        s_a = sum(w_a.values())
        pcts = [w_a[k]/s_a * 100 for k in dial_order]
        linestyle = '-' if a == alpha else '--'
        linewidth = 2.5 if a == alpha else 1.2
        label = f"alpha = {a} (Scelto)" if a == alpha else f"alpha = {a}"
        ax2.plot(dial_order, pcts, marker='o', markersize=4, linestyle=linestyle,
                 linewidth=linewidth, label=label)
    ax2.set_title(f"2. Curve di Smoothing (Cap IT = {it_cap/1000:.0f}k)\n(Quota percentuale per classe)", fontsize=12, fontweight='bold', pad=10)
    ax2.set_ylabel("Quota Campionaria (%)", fontsize=10)
    ax2.set_xlabel("Varietà Linguistica", fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(fontsize=9)

    # Pannello 3: Distribuzione Finale Campionata (200k frasi totali)
    ax3 = axes[2]
    sampled_counts = df_balanced['dial'].value_counts()
    bars3 = ax3.bar(dial_order, [sampled_counts[d] for d in dial_order], color='#27ae60', edgecolor='black', alpha=0.85)
    bars3[0].set_color('#2980b9') # Italiano evidenziato in blu
    ax3.set_title(f"3. Dataset Finale Neurale (Total = {total_samples/1000:.0f}k)\n(IT ~16.5% = 33k | Dialetti ~83.5% = 167k)", fontsize=12, fontweight='bold', pad=10)
    ax3.set_ylabel("Numero di Frasi Campionate", fontsize=10)
    ax3.set_xlabel("Varietà Linguistica", fontsize=10)
    ax3.grid(axis='y', linestyle='--', alpha=0.5)
    ax3.tick_params(axis='x', rotation=45)
    for bar in bars3:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 600,
                 f"{yval/1000:.1f}k\n({yval/total_samples*100:.1f}%)",
                 ha='center', va='bottom', fontsize=7.5)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"-> Grafici salvati con successo in '{plot_path}'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--it_cap', type=int, default=150000, help="Cap preventivo sull'Italiano nel pool iniziale")
    parser.add_argument('--alpha', type=float, default=0.5, help="Esponente di Temperature Sampling")
    parser.add_argument('--total_samples', type=int, default=200000, help="Numero totale di frasi nel dataset finale")
    args = parser.parse_args()

    temperature_sample_dataset(
        it_cap=args.it_cap,
        alpha=args.alpha,
        total_samples=args.total_samples
    )
