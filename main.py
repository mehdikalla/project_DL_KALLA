import os
import torch
import argparse

from networks.baseline import CNNet
from networks.improved import ResNet  
from utils.visualization import plot_loss_curve
from utils.metrics import save_logs
from dataset.preprocessing import preprocess_dataset


def get_model(model_name, device):
    if model_name == "baseline":
        return CNNet(device=device)
    elif model_name == "improved":
        return ResNet(device=device)
    else:
        raise ValueError(f"Modèle inconnu : {model_name}")


def main():
    parser = argparse.ArgumentParser(description="Pipeline FMA-small")

    parser.add_argument(
        "--mode", type=str, choices=["preprocess", "train", "test"], required=True,
        help="Mode à exécuter : preprocessing, entrainement ou test"
    )
    parser.add_argument(
        "--model", type=str, choices=["baseline", "improved"], default="baseline",
        help="Choix du modèle"
    )
    parser.add_argument("--features", type=str, default="dataset/data/mel_specs.npy")
    parser.add_argument("--labels", type=str, default="dataset/data/labels.npy")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--save_plot", type=str, default="results/plots/loss_curve.png")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device utilisé : {device}")

    if args.mode == "preprocess":
        preprocess_dataset()
        return

    # Vérification fichiers pour train et test
    if not os.path.exists(args.features):
        raise FileNotFoundError(args.features)
    if not os.path.exists(args.labels):
        raise FileNotFoundError(args.labels)

    net = get_model(args.model, device)
    net.create_loaders(
        features_path=args.features,
        labels_path=args.labels,
        batch_size=args.batch_size,
        max_length=args.max_length
    )

    if args.mode == "train":
        train_losses, val_losses = net.train(num_epochs=args.epochs)
        plot_loss_curve(train_losses, val_losses, save_path=args.save_plot)
        logs = {"train_loss": train_losses, "val_loss": val_losses}
        save_logs(logs, file_path="results/logs/training_logs.txt")
        print("Entraînement terminé.")

    elif args.mode == "test":
        acc = net.test()
        print(f"Accuracy test : {acc:.2f}%")

if __name__ == "__main__":
    main()
