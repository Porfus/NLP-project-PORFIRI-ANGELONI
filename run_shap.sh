#!/bin/bash
#SBATCH --job-name=eval_shap
#SBATCH --partition=gpu-2xA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=logs/shap_%j.log
#SBATCH --error=logs/shap_%j.err

# 1. Spostati nella cartella del progetto e crea le cartelle per i log/risultati
cd ~/NLP_project
mkdir -p logs results checkpoints

# 2. Carica i moduli Elja per Python e CUDA
module use /hpcapps/lib-edda/modules/all/Core
module load Python/3.11.3
module load CUDA/12.1.1 || true

# 3. Librerie dinamiche necessarie su Elja
export LD_LIBRARY_PATH=/hpcapps/lib-edda/easybuild/software/Python/3.11.3-GCCcore-12.3.0/lib:/hpcapps/lib-edda/easybuild/software/Python/3.11.3-GCCcore-12.3.0/lib64:$LD_LIBRARY_PATH

# 4. Attiva l'ambiente virtuale
source ~/nlp_env/bin/activate

echo "=== AVVIO VALUTAZIONE SHAP SU GPU ==="
echo "Nodo assegnato: $SLURMD_NODENAME"
echo "Data avvio: $(date)"
echo "GPU rilevata:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "=========================================="

echo " Esecuzione Notebook Interpretability & SHAP..."
jupyter nbconvert --to notebook --execute interpretability_and_codeswitching.ipynb \
    --output executed_interpretability_and_codeswitching.ipynb \
    --ExecutePreprocessor.timeout=-1

echo "=== JOB SHAP COMPLETATO CON SUCCESSO: $(date) ==="
