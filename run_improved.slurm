#!/bin/bash
#
# ===============================================
# Configuration SLURM pour l'Entraînement Improved
# ===============================================
#SBATCH --job-name=Train_Improved 
#SBATCH --partition=gpu-best 
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4 
#SBATCH --mem=16G 
#SBATCH --time=01:00:00        # Temps maximal pour l'entraînement
#SBATCH --output=/scratch/mkalla/project_DL_KALLA/results/logs/improved_train_%j.out 
#SBATCH --error=/scratch/mkalla/project_DL_KALLA/results/logs/improved_train_%j.err   

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
echo "--- Démarrage de l'entraînement Improved ---"

# -------------------------------------------------------------
# EXÉCUTION DU CODE PYTHON
# -------------------------------------------------------------

# Lancement de main.py en mode train avec le modèle improved.
python main.py \
    --mode train \
    --model improved \
    --batch_size 64 \
    --epochs 50 \
    --max_length 128 \
    --save_path results 

echo "--- Entraînement Improved terminé ---"