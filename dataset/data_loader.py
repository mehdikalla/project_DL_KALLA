# Fichier : dataset/data_loader.py (à modifier)

import torch as tc
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

class FMADataset(Dataset):
    # Accepte une liste de chemins de features (ou un seul chemin)
    def __init__(self, features_paths, labels_path, max_length=128, transform=None):
        """
        features_paths: Liste des chemins vers les .npy features.
        labels_path   : chemin vers labels.npy
        """
        if isinstance(features_paths, str):
            features_paths = [features_paths]
            
        self.X_features = [np.load(p, allow_pickle=True) for p in features_paths]
        self.y = np.load(labels_path, allow_pickle=True)
        
        self.max_length = max_length
        self.transform = transform
        self.num_channels = len(self.X_features)

        # AJOUT : VÉRIFICATION DE LA CONSISTANCE DES TAILLES
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
        # L'erreur se produit ici si les tailles ne sont pas cohérentes
        feature_list = [feature_set[idx] for feature_set in self.X_features]
        label = self.y[idx]
        
        if self.num_channels == 1:
            # Baseline: [128, max_length] -> [1, 128, max_length]
            # Assurez-vous que le padding/truncation est RETIRÉ d'ici s'il est déjà fait en preprocessing
            spec = np.expand_dims(feature_list[0], axis=0) 
        else:
            # Improved: Empile les 3 features -> [3, 128, max_length]
            spec = np.stack(feature_list, axis=0) 

        if self.transform:
            spec = self.transform(spec)

        return tc.tensor(spec, dtype=tc.float32), tc.tensor(label, dtype=tc.long)

# Mise à jour de get_dataloaders
def get_dataloaders(features_paths, labels_path, batch_size=32, val_split=0.1, test_split=0.1, max_length=128):
    """
    Retourne des DataLoaders PyTorch pour train, val, test
    """
    
    # La conversion de string en list est faite dans FMADataset.__init__
    dataset = FMADataset(features_paths, labels_path, max_length=max_length) 
    
    # Calculer les tailles des splits
    total_size = len(dataset)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    
    # Calculer train_size en dernier pour prendre le reste
    train_size = total_size - val_size - test_size

    # random_split fonctionne correctement si total_size est exact
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader