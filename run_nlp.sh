#!/bin/bash
#SBATCH --job-name=nlp_dump
#SBATCH --time=04:00:00          # Tempo massimo (es. 6 ore)
#SBATCH --cpus-per-task=8        # 8 core CPU
#SBATCH --mem=32G                # 32 GB di RAM
# #SBATCH --partition=gpu-1xA100        # (Decommenta se serve la GPU)
# #SBATCH --partition=gpu-2xA100        # (Decommenta se serve la GPU)
# #SBATCH --gres=gpu-1xA100:1           # (Decommenta se serve la GPU)
#SBATCH --output=slurm_%j.log    # File di output con i log
#SBATCH --error=slurm_%j.err     # Eventuali errori o warning
#SBATCH --partition=any_cpu


# Stampa info di diagnostica nel log
echo "=== JOB AVVIATO IL $(date) SUL NODO $SLURMD_NODENAME ==="
echo "Allocati: $SLURM_CPUS_PER_TASK core e $SLURM_MEM_PER_NODE MB di RAM"

# Carica l'ambiente
source ~/nlp_env/bin/activate

# Esegue il notebook dall'inizio alla fine salvando i risultati in un nuovo notebook
jupyter nbconvert --to notebook --execute dialects_wikipedia_dump.ipynb --ExecutePreprocessor.timeout=-1 --output executed_dialects_dump.ipynb

# 2. Evita che i print rimangano nel buffer e imposta i thread di calcolo
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun -t 02:00:00 --cpus-per-task=4 --mem=16G --pty /bin/bash


echo "=== JOB COMPLETATO IL $(date) ==="
