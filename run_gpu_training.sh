#!/bin/bash
#SBATCH --job-name=bert_dialect_fusion
#SBATCH --partition=gpu-1xA100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_gpu_%j.log

echo "=== ESECUZIONE JOB GPU SU CLUSTER ==="
echo "Nodo assegnato: $SLURMD_NODENAME"
echo "Data avvio: $(date)"

# 1. Attivazione ambiente virtuale del cluster
source ~/nlp_env/bin/activate

# 2. Verifica presenza modello pre-scaricato
MODEL_DIR="models/bert-base-italian-xxl-cased"
if [ ! -d "$MODEL_DIR" ]; then
    echo "Directory $MODEL_DIR non trovata. Esegui prima 'python download_model_cache.py' sul login node!"
    exit 1
fi

# 3. Addestramento Baseline 2: Transformer Puro (BERT XXL Italian)
echo "============================================================"
echo "FASE 1: ADDESTRAMENTO BASELINE 2 (BERT PURO)"
echo "============================================================"
python3 src/main.py \
    --model_type baseline \
    --model_name "$MODEL_DIR" \
    --train_path wiki-dump/Dataset/train_neural_balanced.csv \
    --val_path wiki-dump/Dataset/val_sentences.csv \
    --epochs 3 \
    --batch_size 32 \
    --max_len 70 \
    --lr_bert 2e-5 \
    --save_dir checkpoints

# 4. Addestramento Modello Proposto: Dialect Feature Fusion BERT
echo "============================================================"
echo "FASE 2: ADDESTRAMENTO MODELLO PROPOSTO (FEATURE FUSION BERT)"
echo "============================================================"
python3 src/main.py \
    --model_type fusion \
    --model_name "$MODEL_DIR" \
    --train_path wiki-dump/Dataset/train_neural_balanced.csv \
    --val_path wiki-dump/Dataset/val_sentences.csv \
    --epochs 3 \
    --batch_size 32 \
    --max_len 70 \
    --lr_bert 2e-5 \
    --lr_aux 1e-3 \
    --save_dir checkpoints

echo "=== TUTTI GLI ADDESTRAMENTI SONO STATI COMPLETATI CON SUCCESSO ==="
echo "Data fine: $(date)"
