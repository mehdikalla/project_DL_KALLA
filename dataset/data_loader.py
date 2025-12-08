import torch as tc
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

class FMADataset(Dataset):
    def __init__(self, features_paths, labels_path, max_length=128, transform=None):
        """
        features_paths: Liste des chemins vers les .npy features (1 pour baseline, 2 pour improved).
        labels_path   : chemin vers labels.npy
        max_length    : longueur temporelle des spectrograms (padding/truncation)
        transform     : transformations optionnelles
        """
        # S'assurer que features_paths est une liste
        if isinstance(features_paths, str):
            features_paths = [features_paths]
            
        self.X_features = [np.load(p, allow_pickle=True) for p in features_paths]
        self.y = np.load(labels_path, allow_pickle=True)
        
        self.max_length = max_length
        self.transform = transform
        self.num_channels = len(self.X_features)

        # VÉRIFICATION DE LA CONSISTANCE DES TAILLES (Sécurité contre l'erreur Index out of bounds)
        label_size = len(self.y)
        for i, feature_set in enumerate(self.X_features):
            if len(feature_set) != label_size:
                raise ValueError(
                    f"Inconsistance de taille: Labels ont {label_size} échantillons, "
                    f"mais la feature {i} ({features_paths[i]}) a {len(feature_set)} échantillons. "
                    "Veuillez vérifier votre phase de préprocessing."
                )
        # FIN DE LA VÉRIFICATION


    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # Récupère l'échantillon pour chaque feature (Mel, Chroma...)
        feature_list = [feature_set[idx] for feature_set in self.X_features]
        label = self.y[idx]
        
        if self.num_channels == 1:
            # Modèle Baseline (1 canal) : Ajoute la dimension du canal [128, 128] -> [1, 128, 128]
            spec = np.expand_dims(feature_list[0], axis=0) 
        else:
            # Modèle Improved (2 canaux) : Empile les features [Mel, Chroma] -> [2, 128, 128]
            spec = np.stack(feature_list, axis=0) 

        if self.transform:
            spec = self.transform(spec)

        return tc.tensor(spec, dtype=tc.float32), tc.tensor(label, dtype=tc.long)


def get_dataloaders(features_paths, labels_path, batch_size=32, val_split=0.1, test_split=0.1, max_length=128):
    """
    Retourne des DataLoaders PyTorch pour train, val, test.
    features_paths peut être un chemin (str) ou une liste de chemins (list).
    """
    
    dataset = FMADataset(features_paths, labels_path, max_length=max_length) 

    # Calculer les tailles des splits
    total_size = len(dataset)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    
    # Calculer train_size en dernier pour prendre le reste
    train_size = total_size - val_size - test_size

    # Utilise random_split pour diviser le Dataset
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader