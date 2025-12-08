import os
import torch
import argparse

from src.network import main_network 
from src.utils.visualization import plot_loss_curve, plot_metrics_curve
from src.utils.metrics import save_logs
from dataset.preprocessing import preprocess_dataset, relabel


def main():
    parser = argparse.ArgumentParser(description="Pipeline FMA-small")

    parser.add_argument("--mode", type=str, choices=["preprocess", "train", "test"], required=True, help="Mode")
    parser.add_argument("--model", type=str, choices=["baseline", "improved"], default="baseline", help="Choix du modèle")
    parser.add_argument("--features", type=str, default="dataset/data/mel_specs.npy")
    parser.add_argument("--labels", type=str, default="dataset/data/labels.npy")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--save_path", type=str, default="results")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device utilisé : {device}")

    # --- Configuration des chemins de base ---
    MODEL_NAME = args.model
    BASE_PATH = os.path.join(args.save_path, MODEL_NAME)
    PLOTS_PATH = os.path.join(BASE_PATH, "plots")
    LOGS_PATH = os.path.join(BASE_PATH, "logs")
    
    # Correction: Le dossier 'weights' est inclus dans le chemin du fichier pour plus de clarté.
    WEIGHTS_DIR = os.path.join(BASE_PATH, "weights") # Ajout d'une variable pour le dossier


    
    # --- VERSIONING SIMPLE ---
    # On cherche le premier numéro disponible : 1, 2, 3...
    version = 1
    # On vérifie l'existence dans le dossier WEIGHTS_DIR
    while os.path.exists(os.path.join(WEIGHTS_DIR, f"weights_{version}.pth")): 
        version += 1
    
    # Si on est en mode TEST, on prend la dernière version existante
    if args.mode == "test" and version > 1:
        version -= 1
    
    print(f"--- Exécution version : {version} ---")
    
    # Construction des chemins des fichiers
    WEIGHTS_FILE_PATH = os.path.join(WEIGHTS_DIR, f"weights_{version}.pth")
    PLOT_LOSS_PATH = os.path.join(PLOTS_PATH, f"loss_curve_{version}.png")
    PLOT_ACC_PATH = os.path.join(PLOTS_PATH, f"accuracy_curve_{version}.png")
    LOGS_FILE_PATH = os.path.join(LOGS_PATH, f"training_logs_{version}.txt")
    # ----------------------------------

    if args.mode == "preprocess":
        preprocess_dataset()
        relabel(args.labels)
        return

    if not os.path.exists(args.features) or not os.path.exists(args.labels):
        raise FileNotFoundError("Données non trouvées.")

    # --- NOUVEAU: Logique de sélection des chemins de features ---
    feature_paths = [args.features] 

    if args.model == "improved":
        mel_path = args.features
        # On cherche le fichier CQT maintenant
        cqt_path = mel_path.replace("mel_specs", "cqt_specs") 

        if not os.path.exists(cqt_path):
            raise FileNotFoundError(f"Fichier CQT introuvable : {cqt_path}. Relancez le preprocess.")
            
        feature_paths = [mel_path, cqt_path] # Liste de 2 chemins (Mel + CQT)
    # -----------------------------------------------------------


    # La fonction main_network doit retourner l'instance de votre classe CNNet/ResNet
    net = main_network(MODEL_NAME, device) 
    # create_loaders accepte maintenant une liste de chemins
    net.create_loaders(feature_paths, args.labels, args.batch_size, args.max_length)

    if args.mode == "train":
        # Crée les dossiers nécessaires
        os.makedirs(PLOTS_PATH, exist_ok=True)
        os.makedirs(LOGS_PATH, exist_ok=True)
        os.makedirs(WEIGHTS_DIR, exist_ok=True) 
        
        train_losses, val_losses, train_accuracy, val_accuracy = net.train(num_epochs=args.epochs)
        
        # Sauvegarde des courbes et logs
        plot_loss_curve(train_losses, val_losses, save_path=PLOT_LOSS_PATH)
        plot_metrics_curve(
            metrics={"train": train_accuracy, "val": val_accuracy},
            metric_name='Accuracy',
            save_path=PLOT_ACC_PATH
        )   
        logs = {"train_loss": train_losses, "val_loss": val_losses, "train_accuracy": train_accuracy, "val_accuracy": val_accuracy}
        save_logs(logs, file_path=LOGS_FILE_PATH)
        
        # Sauvegarde du modèle
        torch.save(net.model.state_dict(), WEIGHTS_FILE_PATH)
        print(f"Modèle sauvegardé : {WEIGHTS_FILE_PATH}")

    elif args.mode == "test":
        if not os.path.exists(WEIGHTS_FILE_PATH):
            raise FileNotFoundError(f"Poids introuvables : {WEIGHTS_FILE_PATH}")
        
        print(f"Chargement : {WEIGHTS_FILE_PATH}")
        net.model.load_state_dict(torch.load(WEIGHTS_FILE_PATH, map_location=device))
            
        acc = net.test()
        print(f"Accuracy test : {acc:.4f}")

if __name__ == "__main__":
    main()