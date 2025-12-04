#!/bin/bash
#
# ===============================================
# Configuration SLURM pour l'Entraînement Baseline
# ===============================================
#SBATCH --job-name=Train_Baseline 
#SBATCH --partition=gpu-best 
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4 
#SBATCH --mem=16G 
#SBATCH --time=01:00:00        # Temps maximal pour l'entraînement (ajuster selon besoin)
#SBATCH --output=/scratch/mkalla/project_DL_KALLA/results/logs/baseline_train_%j.out # Chemin à AJUSTER
#SBATCH --error=/scratch/mkalla/project_DL_KALLA/results/logs/baseline_train_%j.err   # Chemin à AJUSTER

# -------------------------------------------------------------
# Configuration ENVIRONNEMENT
# -------------------------------------------------------------
set -euo pipefail
module purge
module load cuda/12.5 # À AJUSTER selon la version disponible sur votre cluster
export MPLBACKEND=Agg 

# ATTENTION : Remplacez par le chemin de votre projet et l'activation de votre environnement
# Utilisation du chemin trouvé : /scratch/mkalla/miniconda3
source /scratch/mkalla/miniconda3/etc/profile.d/conda.sh
cd /scratch/mkalla/project_DL_KALLA 
echo "--- Démarrage de l'entraînement Baseline ---"

# -------------------------------------------------------------
# EXÉCUTION DU CODE PYTHON
# -------------------------------------------------------------

# Lancement de main.py en mode train avec le modèle baseline.
python main.py \
    --mode train \
    --model baseline \
    --batch_size 64 \
    --epochs 50 \
    --max_length 128 \
    --save_path results 

echo "--- Entraînement Baseline terminé ---"