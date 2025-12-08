import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm  

DATASET_DIR = "metadata/fma_small"
TRACKS_CSV = "metadata/tracks.csv"
OUTPUT_MEL = "dataset/data/mel_specs.npy"
OUTPUT_CHROMA = "dataset/data/chroma_stft.npy"
OUTPUT_LABELS = "dataset/data/labels.npy"

SR = 22050
N_MELS = 128
N_CHROMA = 12 
N_FFT = 2048
HOP_LENGTH = 512
MAX_LEN = 128  # longueur temporelle


def load_genre_labels(csv_path):
    df = pd.read_csv(csv_path, index_col=0, header=[0, 1])
    genres = df['track']['genre_top']
    mapping = {g: i for i, g in enumerate(sorted(genres.dropna().unique()))}
    return genres, mapping


def preprocess_track(path):
    y, sr = librosa.load(path, sr=SR)
    
    # 1. Mel-spectrogramme (Timbre)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel = librosa.power_to_db(mel, ref=np.max)

    # 2. Chroma Features (Harmonie)
    chroma = librosa.feature.chroma_stft(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_chroma=N_CHROMA
    )
    
    # Normalisation
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    chroma = (chroma - chroma.mean()) / (chroma.std() + 1e-6)
    
    features = [mel, chroma]
    processed_features = []
    
    for i, feature in enumerate(features):
        
        # --- CORRECTION MAJEURE : ÉTIREMENT DU CHROMA ---
        if i == 1: # Si c'est le Chroma (Shape [12, T])
            # On répète chaque note 10 fois verticalement
            # 12 notes * 10 = 120 pixels de haut
            feature = np.repeat(feature, 10, axis=0)
            
            # Il manque 8 pixels pour atteindre 128 (128 - 120 = 8)
            # On ajoute 4 pixels de padding en haut et 4 en bas pour centrer
            feature = np.pad(feature, ((4, 4), (0, 0)), mode='constant')
            
            # Maintenant feature est [128, T], rempli d'infos utiles !
            
        # Padding Temporel (Axe 1 - Temps)
        if feature.shape[1] < MAX_LEN:
            pad_width = MAX_LEN - feature.shape[1]
            feature = np.pad(feature, ((0, 0), (0, pad_width)))
        else:
            feature = feature[:, :MAX_LEN]
            
        processed_features.append(feature)

    return processed_features[0], processed_features[1]


def preprocess_dataset():
    genres, mapping = load_genre_labels(TRACKS_CSV)

    mel_specs = []
    chroma_specs = []
    labels = []

    all_files = []
    for root, _, files in os.walk(DATASET_DIR):
        for file in files:
            if file.endswith(".mp3"):
                all_files.append(os.path.join(root, file))

    for track_path in tqdm(all_files, desc="Preprocessing audio", ncols=100):
        track_id = int(os.path.splitext(os.path.basename(track_path))[0])
        genre = genres.get(track_id)

        if pd.isna(genre):
            continue

        try:
            mel, chroma = preprocess_track(track_path)
            mel_specs.append(mel)
            chroma_specs.append(chroma)
            labels.append(mapping[genre])
        except Exception as e:
            print(f"Erreur avec {track_path} : {e}")
            continue

    mel_specs = np.array(mel_specs, dtype=np.float32)
    chroma_specs = np.array(chroma_specs, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    np.save(OUTPUT_MEL, mel_specs)
    np.save(OUTPUT_CHROMA, chroma_specs)
    np.save(OUTPUT_LABELS, labels)

    print("\nPréprocessing terminé.")
    print("Shape mel_specs :", mel_specs.shape)
    print("Shape chroma_specs :", chroma_specs.shape)
    print("Shape labels :", labels.shape)


def relabel(labels_path):
    labels = np.load(labels_path)
    unique_labels = np.unique(labels)
    label_map = {old: new for new, old in enumerate(sorted(unique_labels))}
    labels_mapped = np.array([label_map[l] for l in labels], dtype=np.int64)
    np.save(labels_path, labels_mapped)
    print("Labels remappés :", np.unique(labels_mapped))