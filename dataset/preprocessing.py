import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm  

# Configuration des chemins et des paramètres
DATASET_DIR = "metadata/fma_small"
TRACKS_CSV = "metadata/tracks.csv"
OUTPUT_MEL = "dataset/data/mel_specs.npy"
OUTPUT_CQT = "dataset/data/cqt_specs.npy" 
OUTPUT_LABELS = "dataset/data/labels.npy"

SR = 22050
N_MELS = 128
N_CQT_BINS = 84 
BINS_PER_OCTAVE = 12

N_FFT = 2048
HOP_LENGTH = 512
MAX_LEN = 128

# Fonction pour charger les étiquettes de genre
def load_genre_labels(csv_path):
    df = pd.read_csv(csv_path, index_col=0, header=[0, 1])
    genres = df['track']['genre_top']
    mapping = {g: i for i, g in enumerate(sorted(genres.dropna().unique()))}
    return genres, mapping

# Fonction de prétraitement pour une piste audio
def preprocess_track(path):
    y, sr = librosa.load(path, sr=SR)
    
    # 1. Mel-spectrogramme 
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel = librosa.power_to_db(mel, ref=np.max)

    # 2. Constant-Q Transform
    cqt = librosa.cqt(
        y=y, sr=sr, 
        hop_length=HOP_LENGTH, 
        n_bins=N_CQT_BINS, 
        bins_per_octave=BINS_PER_OCTAVE
    )
    cqt = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    
    # Normalisation
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    cqt = (cqt - cqt.mean()) / (cqt.std() + 1e-6)
    
    features = [mel, cqt]
    processed_features = []
    
    for i, feature in enumerate(features):
        
        # --- PADDING VERTICAL POUR CQT ---
        if i == 1: 
            pad_height = N_MELS - feature.shape[0]
            pad_top = pad_height // 2
            pad_bottom = pad_height - pad_top
            feature = np.pad(feature, ((pad_bottom, pad_top), (0, 0)), mode='constant')
            
        if feature.shape[1] < MAX_LEN:
            pad_width = MAX_LEN - feature.shape[1]
            feature = np.pad(feature, ((0, 0), (0, pad_width)))
        else:
            feature = feature[:, :MAX_LEN]
            
        processed_features.append(feature)

    return processed_features[0], processed_features[1]

# Fonction principale de prétraitement du dataset
def preprocess_dataset():
    genres, mapping = load_genre_labels(TRACKS_CSV)

    mel_specs = []
    cqt_specs = [] 
    labels = []

    all_files = []
    for root, _, files in os.walk(DATASET_DIR):
        for file in files:
            if file.endswith(".mp3"):
                all_files.append(os.path.join(root, file))

    for track_path in tqdm(all_files, desc="Preprocessing CQT+Mel", ncols=100):
        track_id = int(os.path.splitext(os.path.basename(track_path))[0])
        genre = genres.get(track_id)

        if pd.isna(genre):
            continue

        try:
            mel, cqt = preprocess_track(track_path)
            mel_specs.append(mel)
            cqt_specs.append(cqt)
            labels.append(mapping[genre])
        except Exception as e:
            print(f"Erreur avec {track_path} : {e}")
            continue

    mel_specs = np.array(mel_specs, dtype=np.float32)
    cqt_specs = np.array(cqt_specs, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    np.save(OUTPUT_MEL, mel_specs)
    np.save(OUTPUT_CQT, cqt_specs)
    np.save(OUTPUT_LABELS, labels)

    print("\nPréprocessing terminé.")
    print("Shape mel_specs :", mel_specs.shape)
    print("Shape cqt_specs :", cqt_specs.shape) # (N, 128, 128)
    print("Shape labels :", labels.shape)

# Fonction pour remapper les étiquettes de genre
def relabel(labels_path):
    labels = np.load(labels_path)
    unique_labels = np.unique(labels)
    label_map = {old: new for new, old in enumerate(sorted(unique_labels))}
    labels_mapped = np.array([label_map[l] for l in labels], dtype=np.int64)
    np.save(labels_path, labels_mapped)
    print("Labels remappés :", np.unique(labels_mapped))