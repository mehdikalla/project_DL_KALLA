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
    Sauvegarde les logs d'entraînement dans un fichier texte.
    
    logs      : dictionnaire contenant les informations de log
    file_path : chemin du fichier de sauvegarde
    """
    with open(file_path, 'w') as f:
        for key, values in logs.items():
            f.write(f"{key}:\n")
            for value in values:
                f.write(f"{value}\n")
            f.write("\n")



