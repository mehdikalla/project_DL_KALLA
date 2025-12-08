import torch.nn as nn
from .blocks import Conv_Block, FC_Block
S = nn.Softmax(dim=1)

# Classe principale du modèle CNN
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
            Conv_Block(32, 64, pool_size=4),
            Conv_Block(64, 128, pool_size=4))  

        # Définition du FC
        self.fc = FC_Block([128*4*4 , 256, num_classes])  
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
