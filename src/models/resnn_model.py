import torch.nn as nn
from .blocks import FC_Block, Res_Block 

class ResNN_model(nn.Module):
    def __init__(self, in_channels=2, num_classes=8):
        super().__init__()

        # Définition des blocs résiduels
        self.res_layers = nn.Sequential(
            Res_Block(in_channels, 32, kernel_sizes=(1,3,5)),
            Res_Block(32, 64),
            Res_Block(64, 128))

        # AJOUT MAJEUR : Pooling pour réduire 16x16 à 1x1 
        # (Réduit 32768 paramètres à 128)
        self.final_pool = nn.MaxPool2d(kernel_size=4) 

        # Le FC_Block utilise désormais 128 entrées (au lieu de 32768)
        self.fc = FC_Block([128*4*4, 128*4, 256, num_classes]) 
    
    def forward(self, x):
        x = self.res_layers(x)
        x = self.final_pool(x)       # Appliquer le pooling final
        x = x.view(x.size(0), -1)    # Aplatir (la dimension est maintenant 128)
        x = self.fc(x)
        return x
