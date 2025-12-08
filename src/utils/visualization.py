import os
import torch.nn as nn
import matplotlib.pyplot as plt

# Fonctions de visualisation des courbes et des échantillons

def _unique_path(path):
    base, ext = os.path.splitext(path)
    k = 1
    new_path = path
    while os.path.exists(new_path):
        new_path = f"{base}_{k}{ext}"
        k += 1
    return new_path

def plot_loss_curve(train_losses, val_losses, save_path='loss_curve.png'):
    save_path = _unique_path(save_path)

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_metrics_curve(metrics, metric_name='Accuracy', save_path='metrics_curve.png'):
    save_path = _unique_path(save_path)

    plt.figure(figsize=(10, 5))
    plt.plot(metrics['train'], label=f'Train {metric_name}')
    plt.plot(metrics['val'], label=f'Validation {metric_name}')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.ylim(0,1)
    plt.title(f'{metric_name} Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    
def visualize_sample(sample, title='Sample Visualization', save_path='sample.png'):
    """
    Affiche un échantillon de données (ex: spectrogramme).
    
    sample      : tenseur ou tableau numpy représentant l'échantillon à visualiser
    title       : titre du graphique
    save_path   : chemin pour sauvegarder l'image
    """
    plt.figure(figsize=(8, 6))
    if len(sample.shape) == 2:
        plt.imshow(sample, aspect='auto', origin='lower')
    elif len(sample.shape) == 3 and sample.shape[0] == 1:
        plt.imshow(sample[0], aspect='auto', origin='lower')
    else:
        raise ValueError("Sample must be 2D or 3D with single channel.")
    
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.savefig(save_path)
    plt.close()