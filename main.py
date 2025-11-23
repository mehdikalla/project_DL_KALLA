import argparse
import torch
import os

from networks.baseline import CNNet
from utils.visualization import plot_loss_curve


def main():
    # ---------------------------------------------------------
    # Argument parsing
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser(description="Train CNN baseline on FMA-small")

    parser.add_argument("--features", type=str,
                        default="dataset/data/mel_specs.npy",
                        help="Chemin vers mel_specs.npy")

    parser.add_argument("--labels", type=str,
                        default="dataset/data/labels.npy",
                        help="Chemin vers labels.npy")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max_length", type=int, default=128)

    parser.add_argument("--save_plot", type=str, default="loss_curve.png",
                        help="Chemin pour sauvegarder la courbe de perte")

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device utilisé : {device}")

    # ---------------------------------------------------------
    # Vérification des fichiers
    # ---------------------------------------------------------
    if not os.path.exists(args.features):
        raise FileNotFoundError(f"Fichier introuvable : {args.features}")
    if not os.path.exists(args.labels):
        raise FileNotFoundError(f"Fichier introuvable : {args.labels}")

    # ---------------------------------------------------------
    # Initialisation du réseau baseline
    # ---------------------------------------------------------
    net = CNNet(device=device)

    # ---------------------------------------------------------
    # Dataloaders
    # ---------------------------------------------------------
    net.create_loaders(
        features_path=args.features,
        labels_path=args.labels,
        batch_size=args.batch_size,
        max_length=args.max_length
    )

    # ---------------------------------------------------------
    # Entraînement
    # ---------------------------------------------------------
    train_losses, val_losses = net.train(num_epochs=args.epochs)

    # ---------------------------------------------------------
    # Plot des courbes
    # ---------------------------------------------------------
    plot_loss_curve(train_losses, val_losses, save_path=args.save_plot)
    print(f"Courbe de perte sauvegardée dans {args.save_plot}")

    # ---------------------------------------------------------
    # Test final
    # ---------------------------------------------------------
    acc = net.test()
    print(f"Accuracy en test : {acc:.2f}%")


if __name__ == "__main__":
    main()
