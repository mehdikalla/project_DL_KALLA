# Deep Learning Project: Music Genre Classification
### KALLA Mehdi - 3A CS ICE - 2025-2026
This project implements a Convolutional Neural Network (CNN) pipeline to classify music genres using the FMA (Free Music Archive) dataset. It features two distinct architectures: a **Baseline** model (CNN) and an **Improved** model (ResNet).

## Project Structure

The project is organized as follows:

```plaintext
project_DL_KALLA/
├── dataset/
│   ├── data_loader.py      # PyTorch Dataset class and DataLoader generation
│   ├── preprocessing.py    # Logic to extract Mel-specs and CQT from MP3s
│   └── data/               # Folder where processed .npy files will be saved
├── metadata/               
│   ├── fma_small/          # Directory containing the audio files (.mp3)
│   └── tracks.csv          # Metadata file containing labels and genre info
├── results/                
│   ├── baseline/           # Logs, plots, and weights for the baseline model
│   ├── improved/           # Logs, plots, and weights for the improved model
│   └── visualization/      # Debug visualizations
├── src/
│   ├── models/             # Neural network architectures (CNN, ResNet, Blocks)
│   │   ├── blocks.py       # Contains the fundamental blocks of each model
│   │   ├── cnn_model.py    # CNN model
│   │   └── resnn_model.py  # ResNN model
│   ├── utils/              # Metrics calculation and visualization tools
│   └── network.py          # Training and evaluation loops
├── main.py                 # Main entry point for the pipeline
└── debug.py                # Script to visualize input data (Sanity Check)
```

## Setup and Installation

**1. Dependencies** \
Ensure that the required Python libraries are installed (PyTorch, Librosa, Pandas, Numpy, Matplotlib, Tqdm).

**2. Data Setup** \
Before running the code, download the FMA dataset and place it in the metadata folder.

* Download fma_small.zip and unzip it.
* Download fma_metadata.zip and extract tracks.csv.
* Place them so the path are set up as :

  * metadata/fma_small/
  * metadata/tracks.csv

## Usage (main.py)

The main.py script is the central command center. It operates in three distinct modes: preprocess, train, and test.

**1. Preprocessing** \
This command converts raw MP3 files into numerical tensors (Labels, Mel-Spectrograms and CQT) and saves them as .npy files for faster loading during training.

```python
Bash

python main.py --mode preprocess

```
* Input: metadata/fma_small/ and tracks.csv.

* Output: Saves mel_specs.npy, cqt_specs.npy, and labels.npy in dataset/data/.

**2. Training**


* Train the Baseline Model (1 Channel: Mel-Spec):

```python
Bash

python main.py --mode train --model baseline --epochs 50 --batch_size 64
```

* Train the Improved Model (2 Channels: Mel-Spec + CQT):
```python
Bash

python main.py --mode train --model improved --epochs 50 --batch_size 64
```

* Output:

Weights saved in results/[model]/weights/
Loss/Accuracy curves saved in results/[model]/plots/
Training logs saved in results/[model]/logs/

**3. Testing** \
Loads the last saved version of the model weights and evaluates performance on the Test set.

```python
Bash

python main.py --mode test --model baseline
# OR
python main.py --mode test --model improved
```

## Visualization (Debug) 

To verify that the data has been processed correctly (checking alignment between Mel-spectrograms and CQT), run:

```python
Bash

python debug.py

```
This will generate sample images in results/visualization/.
