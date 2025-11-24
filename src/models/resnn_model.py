import torch.nn as nn
from .layers import FC_Block, Res_Block 
S = nn.Softmax(dim=1)

class ResNN_model(nn.Module):
    def __init__(self, in_channels=1, num_classes=8):
        super().__init__()

        # Définition des blocs résiduels
        self.res_layers = nn.Sequential(
            Res_Block(in_channels, 32, kernel_sizes=(1,3,5)),
            Res_Block(32, 64),
            Res_Block(64, 128))

        # Définition des blocs fully connected
        self.fc = FC_Block([128 * 16 * 16, 256, num_classes])

    def forward(self, x):
        x = self.res_layers(x)
        x = x.view(x.size(0), -1)  # Aplatir les tenseurs
        x = self.fc(x)
        x = S(x)
        return x
