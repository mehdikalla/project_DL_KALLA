import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm  
DATASET_DIR = "dataset/metadata/fma_small"
TRACKS_CSV = "dataset/metadata/tracks.csv"
OUTPUT_FEATURES = "dataset/data/mel_specs.npy"
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
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    mel = librosa.power_to_db(mel, ref=np.max)

    # normalisation par morceau
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)

    # padding ou truncation
    if mel.shape[1] < MAX_LEN:
        pad_width = MAX_LEN - mel.shape[1]
        mel = np.pad(mel, ((0, 0), (0, pad_width)))
    else:
        mel = mel[:, :MAX_LEN]

    return mel


def preprocess_dataset():
    genres, mapping = load_genre_labels(TRACKS_CSV)

    mel_specs = []
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
            mel = preprocess_track(track_path)
            mel_specs.append(mel)
            labels.append(mapping[genre])
        except Exception as e:
            print(f"Erreur avec {track_path} : {e}")

    mel_specs = np.array(mel_specs, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    np.save(OUTPUT_FEATURES, mel_specs)
    np.save(OUTPUT_LABELS, labels)

    print("\nPréprocessing terminé.")
    print("Shape mel_specs :", mel_specs.shape)
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