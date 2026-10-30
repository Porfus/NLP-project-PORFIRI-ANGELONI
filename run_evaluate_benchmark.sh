#!/bin/bash
#SBATCH --job-name=eval_benchmark
#SBATCH --partition=gpu-2xA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=logs/eval_%j.log
#SBATCH --error=logs/eval_%j.err

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

echo "=== AVVIO VALUTAZIONE BENCHMARK SU GPU ==="
echo "Nodo assegnato: $SLURMD_NODENAME"
echo "Data avvio: $(date)"
echo "GPU rilevata:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "=========================================="

# 5. Esegui lo script di benchmark
# (Aggiungi eventuali argomenti se richiesti da evaluate_benchmark.py, es: --model_path ...)
python3 src/evaluate_benchmark.py

echo "=== VALUTAZIONE COMPLETATA: $(date) ==="
