import torch as tc
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

class FMADataset(Dataset):
    def __init__(self, features_path, labels_path, max_length=128, transform=None):
        """
        features_path : chemin vers mel_specs.npy
        labels_path   : chemin vers labels.npy
        max_length    : longueur temporelle des spectrograms (padding/truncation)
        transform     : transformations optionnelles
        """
        self.X = np.load(features_path, allow_pickle=True)
        self.y = np.load(labels_path, allow_pickle=True)
        self.max_length = max_length
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        spec = self.X[idx]  # shape [128, T]
        label = self.y[idx]

        # Padding ou truncation pour uniformiser la longueur
        if spec.shape[1] < self.max_length:
            pad_width = self.max_length - spec.shape[1]
            spec = np.pad(spec, ((0,0),(0,pad_width)), mode='constant')
        elif spec.shape[1] > self.max_length:
            spec = spec[:, :self.max_length]

        # Ajouter canal pour CNN 2D
        spec = np.expand_dims(spec, axis=0)  # [1, 128, max_length]

        if self.transform:
            spec = self.transform(spec)

        return tc.tensor(spec, dtype=tc.float32), tc.tensor(label, dtype=tc.long)

def get_dataloaders(features_path, labels_path, batch_size=32, val_split=0.1, test_split=0.1, max_length=128):
    """
    Retourne des DataLoaders PyTorch pour train, val, test
    """
    dataset = FMADataset(features_path, labels_path, max_length=max_length)

    # Calculer les tailles des splits
    total_size = len(dataset)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size - test_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
