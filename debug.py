import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa.display
import random

# --- Configuration des chemins ---
DATA_DIR = "dataset/data"
METADATA_DIR = "metadata"
MEL_PATH = os.path.join(DATA_DIR, "mel_specs.npy")
CQT_PATH = os.path.join(DATA_DIR, "cqt_specs.npy") 
LABELS_PATH = os.path.join(DATA_DIR, "labels.npy")
TRACKS_CSV = os.path.join(METADATA_DIR, "tracks.csv")

# Dossier de sortie
VIS_OUTPUT_DIR = "results/visualization"

# Paramètres audio
SR = 22050
HOP_LENGTH = 512
N_MELS = 128

def load_genre_mapping(csv_path):
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path, index_col=0, header=[0, 1])
        genres = df['track']['genre_top']
        unique_genres = sorted(genres.dropna().unique())
        return {i: g for i, g in enumerate(unique_genres)}
    except Exception:
        return None

def plot_distribution(labels, mapping, save_dir):
    plt.figure(figsize=(10, 5))
    counts = np.bincount(labels)
    indices = np.arange(len(counts))
    if mapping:
        names = [mapping.get(i, str(i)) for i in indices]
        plt.bar(indices, counts, tick_label=names)
        plt.xticks(rotation=45, ha='right')
    else:
        plt.bar(indices, counts)
    plt.title("Genres")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "distribution_genres.png"))
    plt.close()

def visualize_samples(mel_data, cqt_data, labels, mapping, save_dir, num_samples=5):
    """
    Affiche Mel-spec et CQT côte à côte.
    """
    total_samples = len(labels)
    indices = random.sample(range(total_samples), num_samples)
    
    # --- Configuration des axes pour la CQT ---
    # La CQT fait 84 bins de haut (7 octaves).
    # Elle est centrée dans une image de 128 pixels.
    # Padding haut/bas = (128 - 84) / 2 = 22 pixels.
    # Donc le CQT commence au pixel 22 et finit au pixel 106.
    
    pad_bottom = 22
    
    # On veut afficher les octaves (C1, C2, ... C7) sur l'axe Y
    # C1 est au bin 0 du CQT (donc pixel 22)
    # C2 est au bin 12 du CQT (donc pixel 22 + 12 = 34)
    # etc.
    octaves = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
    yticks_pos = [pad_bottom + (i * 12) for i in range(len(octaves))]

    for idx in indices:
        mel = mel_data[idx]       
        cqt = cqt_data[idx] 
        label_id = labels[idx]
        
        genre_raw = mapping.get(label_id, str(label_id)) if mapping else str(label_id)
        genre_clean = "".join([c if c.isalnum() else "_" for c in genre_raw])
        
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"Sample #{idx} - Genre : {genre_raw}", fontsize=16)
        
        # --- 1. Mel-Spectrogramme ---
        img_mel = librosa.display.specshow(
            mel, sr=SR, hop_length=HOP_LENGTH, 
            x_axis='time', y_axis='mel', ax=ax[0], cmap='magma'
        )
        ax[0].set_title("Mel-Spectrogramme (Timbre)")
        fig.colorbar(img_mel, ax=ax[0], format='%+2.0f dB')
        
        # --- 2. CQT (Pitch / Notes) ---
        # On utilise imshow pour voir les pixels bruts (avec le padding)
        img_cqt = ax[1].imshow(
            cqt, 
            aspect='auto', 
            origin='lower', 
            cmap='coolwarm',
            interpolation='nearest'
        )
        
        ax[1].set_title("CQT (84 bins)")
        ax[1].set_xlabel("Time (frames)")
        ax[1].set_ylabel("Musical Pitch (Octaves)")
        
        # Placement des ticks personnalisés pour les octaves
        ax[1].set_yticks(yticks_pos)
        ax[1].set_yticklabels(octaves)
        
        # Lignes rouges pour montrer où commence et finit la vraie donnée CQT (hors padding)
        ax[1].axhline(y=22, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax[1].axhline(y=106, color='red', linestyle='--', linewidth=1, alpha=0.5)
        
        plt.tight_layout()
        
        # Sauvegarde
        filename = f"sample_{idx}_{genre_clean}_CQT.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print(f"Échantillon sauvegardé : {save_path}")

def main():
    plt.switch_backend('Agg') # Pour cluster sans écran
    
    print("--- Chargement des données CQT ---")
    if not os.path.exists(CQT_PATH):
        print(f"ERREUR : {CQT_PATH} introuvable. Lancez 'python main.py --mode preprocess'")
        return

    os.makedirs(VIS_OUTPUT_DIR, exist_ok=True)

    mel_specs = np.load(MEL_PATH, allow_pickle=True)
    cqt_specs = np.load(CQT_PATH, allow_pickle=True)
    labels = np.load(LABELS_PATH, allow_pickle=True)
    
    print(f"Mel shape : {mel_specs.shape}")
    print(f"CQT shape : {cqt_specs.shape}")
    
    mapping = load_genre_mapping(TRACKS_CSV)
    
    plot_distribution(labels, mapping, VIS_OUTPUT_DIR)
    visualize_samples(mel_specs, cqt_specs, labels, mapping, VIS_OUTPUT_DIR)

if __name__ == "__main__":
    main()