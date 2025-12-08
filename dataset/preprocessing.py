import os
import numpy as np
# torch et torch.nn.functional ne sont plus nécessaires ici
import pandas as pd
import librosa
from tqdm import tqdm  

# MISE À JOUR : Nouveaux chemins de sortie
DATASET_DIR = "metadata/fma_small"
TRACKS_CSV = "metadata/tracks.csv"
# MISE À JOUR : Nouveaux chemins de sortie (Mel et Chroma)
OUTPUT_MEL = "dataset/data/mel_specs.npy"
OUTPUT_CHROMA = "dataset/data/chroma_stft.npy" # Remplacement de Delta1/Delta2
OUTPUT_LABELS = "dataset/data/labels.npy"

SR = 22050
N_MELS = 128
N_CHROMA = 12 # Nouveau : Nombre de bins Chroma (12 notes)
N_FFT = 2048
HOP_LENGTH = 512
MAX_LEN = 128  # longueur temporelle du mel-spec pour CNN


def load_genre_labels(csv_path):
    # Reste inchangé
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

    # 2. Calcul des Chroma Features (HARMONIQUE/TONAL)
    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_chroma=N_CHROMA
    )
    
    # Normalisation par morceau pour chaque feature
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    chroma = (chroma - chroma.mean()) / (chroma.std() + 1e-6)
    
    features = [mel, chroma]
    processed_features = []
    
    # Padding ou truncation (temporel) pour tous les features (MAX_LEN)
    for i, feature in enumerate(features):
        
        # HACK ARCHITECTURAL: Padding vertical pour Chroma
        # Ceci force Chroma (12 bins) à avoir 128 bins de hauteur pour le CNN
        if i == 1 and feature.shape[0] < N_MELS:
             pad_height = N_MELS - feature.shape[0] # 128 - 12 = 116
             # Pad sur l'axe des fréquences (axe 0) pour que la dimension H soit 128
             feature = np.pad(feature, ((0, pad_height), (0, 0)))
        
        # Padding ou truncation sur l'axe du temps (MAX_LEN)
        if feature.shape[1] < MAX_LEN:
            pad_width = MAX_LEN - feature.shape[1]
            feature = np.pad(feature, ((0, 0), (0, pad_width)))
        else:
            feature = feature[:, :MAX_LEN]
            
        processed_features.append(feature)

    # Retourne les 2 features séparément (Mel [128, 128], Chroma [128, 128])
    return processed_features[0], processed_features[1]


def preprocess_dataset():
    genres, mapping = load_genre_labels(TRACKS_CSV)

    mel_specs = []
    chroma_specs = [] # CHANGEMENT de delta1_specs
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
            mel, chroma = preprocess_track(track_path) # Réception des 2 features
            
            # Si le calcul a réussi, on procède à l'ajout ATOMIQUE des 3 éléments
            mel_specs.append(mel)
            chroma_specs.append(chroma) # Ajout
            labels.append(mapping[genre])
            
        except Exception as e:
            # Si une erreur (librosa.load, etc.) survient, l'échantillon est ignoré
            print(f"Erreur avec {track_path} : {e}")
            continue # Passe à la piste suivante

    mel_specs = np.array(mel_specs, dtype=np.float32)
    chroma_specs = np.array(chroma_specs, dtype=np.float32) # Conversion
    labels = np.array(labels, dtype=np.int64)

    # SAUVEGARDE DES 2 FICHIERS DE FEATURES
    np.save(OUTPUT_MEL, mel_specs)
    np.save(OUTPUT_CHROMA, chroma_specs) # CHANGEMENT
    np.save(OUTPUT_LABELS, labels)

    print("\nPréprocessing terminé.")
    print("Shape mel_specs :", mel_specs.shape)
    print("Shape chroma_specs :", chroma_specs.shape) # CHANGEMENT
    print("Shape labels :", labels.shape)


def relabel(labels_path):
    # Reste inchangé
    labels = np.load(labels_path)
    unique_labels = np.unique(labels)
    label_map = {old: new for new, old in enumerate(sorted(unique_labels))}
    labels_mapped = np.array([label_map[l] for l in labels], dtype=np.int64)
    np.save(labels_path, labels_mapped)
    print("Labels remappés :", np.unique(labels_mapped))