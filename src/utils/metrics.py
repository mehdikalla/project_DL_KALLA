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


def save_logs(logs, file_path):
    """
    Sauvegarde les métriques d'entraînement dans un fichier texte.
    
    Args:
        logs (dict): Dictionnaire contenant les listes de 'train_loss', 
                     'val_loss', 'train_accuracy' et 'val_accuracy'.
        file_path (str): Chemin du fichier de sortie.
    """
    train_losses = logs.get("train_loss", [])
    val_losses = logs.get("val_loss", [])
    train_accuracies = logs.get("train_accuracy", [])
    val_accuracies = logs.get("val_accuracy", [])

    # Assurez-vous que toutes les listes ont la même longueur pour l'itération
    max_len = max(len(train_losses), len(val_losses), len(train_accuracies), len(val_accuracies))

    with open(file_path, 'w') as f:
        f.write("=== TRAINING LOGS ===\n\n")
        # Mise à jour de l'en-tête pour inclure les Accuracy
        f.write(f"{'Epoch':>5} | {'Train Loss':>12} | {'Val Loss':>12} | {'Train Acc':>10} | {'Val Acc':>10}\n")
        f.write("-" * 55 + "\n")

        for i in range(max_len):
            # Utilisation de .get(i, None) pour gérer les cas où une liste est plus courte
            tr_loss = train_losses[i] if i < len(train_losses) else float('nan')
            val_loss = val_losses[i] if i < len(val_losses) else float('nan')
            tr_acc = train_accuracies[i] if i < len(train_accuracies) else float('nan')
            val_acc = val_accuracies[i] if i < len(val_accuracies) else float('nan')
            
            # Formatage des valeurs (Utilisation de .4f pour les pertes et les accuracies)
            f.write(
                f"{i + 1:>5} | "
                f"{tr_loss:12.4f} | "
                f"{val_loss:12.4f} | "
                f"{tr_acc:10.4f} | "
                f"{val_acc:10.4f}\n"
            )

        f.write("\n")