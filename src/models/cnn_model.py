import torch.nn as nn
from .layers import Conv_Block, FC_Block
S = nn.Softmax(dim=1)

class CNN_model(nn.Module):
    def __init__(self, in_channels=1, num_classes=8):
        """
        in_channels : nombre de canaux d'entrée
        num_classes : nombre de classes de sortie
        """
        super().__init__()
        
        # Définition des blocs convolutifs
        self.conv_layers = nn.Sequential(
            Conv_Block(in_channels, 32),
            Conv_Block(32, 64),
            Conv_Block(64, 128)
        )

        # Définition des blocs fully connected
        self.fc = FC_Block([128 * 16 * 16, 256, num_classes])  
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Aplatir les tenseurs
        x = self.fc(x)
        x = S(x)
        return x
