#!/bin/bash
#SBATCH --job-name=logreg_baseline
#SBATCH --partition=any_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/logreg_%j.log
#SBATCH --error=logs/logreg_%j.err

# 1. Spostati nella cartella del progetto e crea le cartelle necessarie
cd ~/NLP_project
mkdir -p logs checkpoints

# 2. Carica il modulo Python corretto
module use /hpcapps/lib-edda/modules/all/Core
module load Python/3.11.3

# 3. Esporta le librerie dinamiche di Python
export LD_LIBRARY_PATH=/hpcapps/lib-edda/easybuild/software/Python/3.11.3-GCCcore-12.3.0/lib:/hpcapps/lib-edda/easybuild/software/Python/3.11.3-GCCcore-12.3.0/lib64:$LD_LIBRARY_PATH

# 4. Attiva l'ambiente virtuale
source ~/nlp_env/bin/activate

echo "=== ESECUZIONE TRAINING BASELINE 1 (LOGISTIC REGRESSION) ==="
echo "Nodo assegnato: $SLURMD_NODENAME"
echo "Data avvio: $(date)"
echo "Cores allocati: $SLURM_CPUS_PER_TASK"
echo "Python in uso: $(which python3)"
echo "==========================================================="

# 5. Esegui il training della Logistic Regression
python3 src/train_baseline_logreg.py

echo "=== TRAINING COMPLETATO: $(date) ==="
