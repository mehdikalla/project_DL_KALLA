import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm  

DATASET_DIR = "metadata/fma_small"
TRACKS_CSV = "metadata/tracks.csv"
# MISE À JOUR : Nouveaux chemins de sortie
OUTPUT_MEL = "dataset/data/mel_specs.npy"
OUTPUT_DELTA1 = "dataset/data/mel_delta1.npy"
OUTPUT_DELTA2 = "dataset/data/mel_delta2.npy"
OUTPUT_LABELS = "dataset/data/labels.npy"

SR = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
MAX_LEN = 128  # longueur temporelle du mel-spec pour CNN


def load_genre_labels(csv_path):
    df = pd.read_csv(csv_path, index_col=0, header=[0, 1])
    genres = df['track']['genre_top']
    mapping = {g: i for i, g in enumerate(sorted(genres.dropna().unique()))}
    return genres, mapping


def preprocess_track(path):
    y, sr = librosa.load(path, sr=SR)
    
    # 1. Calcul du Mel-spectrogramme (MEL)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    mel = librosa.power_to_db(mel, ref=np.max)

    # 2. Calcul des Deltas (DELTAS)
    mel_delta1 = librosa.feature.delta(mel, order=1)
    mel_delta2 = librosa.feature.delta(mel, order=2) 
    
    # Normalisation par morceau pour chaque feature
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    mel_delta1 = (mel_delta1 - mel_delta1.mean()) / (mel_delta1.std() + 1e-6)
    mel_delta2 = (mel_delta2 - mel_delta2.mean()) / (mel_delta2.std() + 1e-6)
    
    features = [mel, mel_delta1, mel_delta2]
    processed_features = []
    
    # Padding ou truncation pour tous les features
    for feature in features:
        if feature.shape[1] < MAX_LEN:
            pad_width = MAX_LEN - feature.shape[1]
            # Padding sur la dimension temporelle (axe 1)
            feature = np.pad(feature, ((0, 0), (0, pad_width)))
        else:
            feature = feature[:, :MAX_LEN]
        processed_features.append(feature)

    # Retourne les 3 features séparément
    return processed_features[0], processed_features[1], processed_features[2]


def preprocess_dataset():
    genres, mapping = load_genre_labels(TRACKS_CSV)

    mel_specs = []
    delta1_specs = [] # NOUVEAU
    delta2_specs = [] # NOUVEAU
    labels = []

    # liste tous les fichiers mp3 d’abord
    all_files = []
    for root, _, files in os.walk(DATASET_DIR):
        for file in files:
            if file.endswith(".mp3"):
                all_files.append(os.path.join(root, file))

    # barre de progression avec tqdm
    for track_path in tqdm(all_files, desc="Preprocessing audio", ncols=100):
        track_id = int(os.path.splitext(os.path.basename(track_path))[0])
        genre = genres.get(track_id)

        if pd.isna(genre):
            continue

        try:
            mel, delta1, delta2 = preprocess_track(track_path) # Réception des 3
            mel_specs.append(mel)
            delta1_specs.append(delta1) # Ajout
            delta2_specs.append(delta2) # Ajout
            labels.append(mapping[genre])
        except Exception as e:
            print(f"Erreur avec {track_path} : {e}")

    mel_specs = np.array(mel_specs, dtype=np.float32)
    delta1_specs = np.array(delta1_specs, dtype=np.float32) # Conversion
    delta2_specs = np.array(delta2_specs, dtype=np.float32) # Conversion
    labels = np.array(labels, dtype=np.int64)

    # SAUVEGARDE DES 3 FICHIERS DE FEATURES
    np.save(OUTPUT_MEL, mel_specs)
    np.save(OUTPUT_DELTA1, delta1_specs)
    np.save(OUTPUT_DELTA2, delta2_specs)
    np.save(OUTPUT_LABELS, labels)

    print("\nPréprocessing terminé.")
    print("Shape mel_specs :", mel_specs.shape)
    print("Shape delta1_specs :", delta1_specs.shape)
    print("Shape delta2_specs :", delta2_specs.shape)
    print("Shape labels :", labels.shape)


def relabel(labels_path):
    """
    Remappe les labels existants pour qu’ils soient consécutifs de 0 à n-1.
    Écrase le même fichier .npy.
    """
    labels = np.load(labels_path)
    unique_labels = np.unique(labels)
    label_map = {old: new for new, old in enumerate(sorted(unique_labels))}
    labels_mapped = np.array([label_map[l] for l in labels], dtype=np.int64)
    np.save(labels_path, labels_mapped)
    print("Labels remappés :", np.unique(labels_mapped))