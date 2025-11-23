import os

def _unique_path(path):
    base, ext = os.path.splitext(path)
    k = 1
    new_path = path
    while os.path.exists(new_path):
        new_path = f"{base}_{k}{ext}"
        k += 1
    return new_path

def accuracy(y_true, y_pred):
    """
    Calcule la précision entre les étiquettes vraies et prédites.
    
    y_true : tenseur des étiquettes vraies
    y_pred : tenseur des étiquettes prédites
    """
    correct = (y_true == y_pred).sum().item()
    total = y_true.size(0)
    return correct / total if total > 0 else 0



def save_logs(logs, file_path='training_logs.txt'):
    """
    Sauvegarde les logs d'entraînement sans écraser les anciens.
    Chaque session est placée dans un nouveau fichier, numéroté automatiquement.
    """

    file_path = _unique_path(file_path)

    train_losses = logs.get("train_loss", [])
    val_losses = logs.get("val_loss", [])

    with open(file_path, 'w') as f:
        f.write("=== TRAINING LOGS ===\n\n")
        f.write(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10}\n")
        f.write("-" * 32 + "\n")

        for i, (tr, val) in enumerate(zip(train_losses, val_losses), 1):
            f.write(f"{i:>5} | {tr:10.4f} | {val:10.4f}\n")

        f.write("\n")