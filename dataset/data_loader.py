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
        self.X_features = [np.load(p, allow_pickle=True) for p in features_paths]
        self.y = np.load(labels_path, allow_pickle=True)
        self.max_length = max_length
        self.transform = transform
        self.num_channels = len(self.X_features)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        feature_list = [feature_set[idx] for feature_set in self.X_features]
        label = self.y[idx]
        
        if self.num_channels == 1:
            # Baseline: [128, max_length] -> [1, 128, max_length]
            spec = np.expand_dims(feature_list[0], axis=0) 
        else:
            # Improved: Empile les 3 features -> [3, 128, max_length]
            spec = np.stack(feature_list, axis=0) 

        if self.transform:
            spec = self.transform(spec)

        return tc.tensor(spec, dtype=tc.float32), tc.tensor(label, dtype=tc.long)

# Mise à jour de get_dataloaders pour accepter une liste
def get_dataloaders(features_paths, labels_path, batch_size=32, val_split=0.1, test_split=0.1, max_length=128):
    """
    features_paths peut être un chemin (str) ou une liste de chemins (list)
    """
    if isinstance(features_paths, str):
        features_paths = [features_paths]
        
    dataset = FMADataset(features_paths, labels_path, max_length=max_length)
    
    # ... (reste inchangé: calcul des splits et création des DataLoaders)
    # ... (Assurez-vous que le reste du corps de la fonction est conservé)

    total_size = len(dataset)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size - test_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader