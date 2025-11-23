import torch.nn as nn
import torch.nn.functional as F

class Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, pool_size=2, dropout_prob=0.25):
        """ 
        in_channels : nombre de canaux d'entrée
        out_channels : nombre de canaux de sortie
        kernel_size : taille du noyau de convolution
        padding : padding pour la convolution
        pool_size : taille du noyau de pooling
        dropout_prob : probabilité de dropout
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.pool = nn.MaxPool2d(pool_size, pool_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x

class FC_Block(nn.Module):
    def __init__(self, layer_sizes):
        """
        layer_sizes : liste d'entiers [in_dim, hidden1, hidden2, ..., out_dim]
        """
        super().__init__()
        layers = []
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        # Propagation à travers toutes les couches sauf la dernière avec ReLU
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        return x
    
