#!/bin/bash
#SBATCH --job-name=nlp_gpu_train
#SBATCH --partition=gpu-1xA100,gpu-2xA100 # Cerca su entrambe (gpu-2xA100 ha nodi IDLE liberi ora!)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G                         # 28GB (rientra perfettamente nei 3.94GB/core di Elja)
#SBATCH --time=04:00:00
#SBATCH --output=logs/gpu_train_%j.log
#SBATCH --error=logs/gpu_train_%j.err


# 1. Crea la cartella per i log se non esiste
mkdir -p logs

# Interrompe lo script se un comando fallisce
set -e
# 1. Spostati subito nella cartella del progetto
cd ~/NLP_project
mkdir -p logs checkpoints

# 2. Configura i percorsi dei moduli di Elja e carica Python e CUDA
module use /hpcapps/lib-edda/modules/all/Core
module load Python/3.11.3
module load CUDA/12.1.1 || true

echo "=== ESECUZIONE JOB GPU SU CLUSTER ==="
echo "Nodo assegnato: $SLURMD_NODENAME"
echo "Data avvio: $(date)"

# 3. Attiva il tuo virtual environment
source ~/nlp_env/bin/activate

# 4. Controllo diagnostico della GPU (scrive nei log la conferma)
echo "=========================================================="
echo "Job avviato su: $(hostname)"
echo "Data e ora:     $(date)"
echo "Python in uso:  $(which python3)"
echo "=========================================================="
nvidia-smi
python3 -c "import torch; print('CUDA disponibile in PyTorch:', torch.cuda.is_available()); print('Nome GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NESSUNA GPU RILEVATA')"
echo "=========================================================="


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
