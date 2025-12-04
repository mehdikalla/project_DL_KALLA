import torch as tc
import torch.nn as nn
from src.models.cnn_model import CNN_model
from src.models.resnn_model import ResNN_model
from dataset.data_loader import get_dataloaders
from src.utils.metrics import accuracy

class main_network():
    def __init__(self, model_name, device):
        self.device = device
        if model_name == 'baseline':
            self.model = CNN_model().to(self.device)
        elif model_name == 'improved' :
            self.model = ResNN_model().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = tc.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = tc.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.9)

        # stockage des prédictions
        self.train_preds = None
        self.train_true = None
        self.val_preds = None
        self.val_true = None
        self.test_preds = None
        self.test_true = None

    def create_loaders(self, features_paths, labels_path, batch_size=64, max_length=128): 
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            features_paths=features_paths, # Utilise la liste ou le chemin unique
            labels_path=labels_path,
            batch_size=batch_size,
            max_length=max_length)

    def train(self, num_epochs):
        train_losses, val_losses = [], []
        train_accuracy, val_accuracy = [], [] # Contient l'accuracy de chaque époque

        for epoch in range(num_epochs):
            print(("--------------------------------"))
            print(f"--- Epoch {epoch+1}/{num_epochs} ---")
            
            # ------- TRAIN -------
            self.model.train()
            batch_losses = []
            # CORRECTION : Accumuler les résultats de TOUS les batches de l'époque
            Y_true_epoch, Y_pred_epoch = [], [] 

            for batch_idx, batch in enumerate(self.train_loader,1):
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                batch_losses.append(loss.item())
                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}/{len(self.train_loader)} – Loss: {loss.item():.4e}")
                
                # CORRECTION : Stockage
                Y_true_epoch.extend(labels.cpu().numpy())
                Y_pred_epoch.extend(tc.argmax(outputs, dim=1).cpu().numpy())
                
            train_losses.append(sum(batch_losses)/len(batch_losses))
            # CORRECTION : Calculer l'accuracy sur TOUTES les prédictions accumulées
            acc_train = accuracy(tc.tensor(Y_true_epoch), tc.tensor(Y_pred_epoch))
            train_accuracy.append(acc_train)

            # scheduler
            self.scheduler.step()

            # ------- VALIDATION -------
            self.model.eval()
            val_batch_losses = []
            # CORRECTION : Accumuler les résultats de TOUS les batches de l'époque
            Y_true_val_epoch, Y_pred_val_epoch = [], [] 

            with tc.no_grad():
                for batch_idx, batch in enumerate(self.val_loader,1):
                    inputs, labels = batch
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    
                    val_batch_losses.append(loss.item())
                    
                    # CORRECTION : Stockage
                    Y_true_val_epoch.extend(labels.cpu().numpy())
                    Y_pred_val_epoch.extend(tc.argmax(outputs, dim=1).cpu().numpy())

            val_losses.append(sum(val_batch_losses)/len(val_batch_losses))
            # CORRECTION : Calculer l'accuracy sur TOUTES les prédictions accumulées
            acc_val = accuracy(tc.tensor(Y_true_val_epoch), tc.tensor(Y_pred_val_epoch))
            val_accuracy.append(acc_val)
            
            # Nettoyage des valeurs pour le print (si accuracy retourne un Tensor)
            if isinstance(acc_train, tc.Tensor): acc_train_print = acc_train.cpu().item()
            else: acc_train_print = acc_train
            if isinstance(acc_val, tc.Tensor): acc_val_print = acc_val.cpu().item()
            else: acc_val_print = acc_val

            print(f"Train Loss: {train_losses[-1]:.4e} | Val Loss: {val_losses[-1]:.4e}")
            print(f"Train Accuracy: {acc_train_print:.4f} | Val Accuracy: {acc_val_print:.4f}")

        return train_losses, val_losses, train_accuracy, val_accuracy

    def test(self):
        self.model.eval()
        losses = []
        # CORRECTION : Accumuler les résultats de TOUS les batches de test
        Y_true_epoch, Y_pred_epoch = [], []

        with tc.no_grad():
            for batch_idx, batch in enumerate(self.test_loader,1):
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                losses.append(loss.item())
                
                # CORRECTION : Stockage
                Y_true_epoch.extend(labels.cpu().numpy())
                Y_pred_epoch.extend(tc.argmax(outputs, dim=1).cpu().numpy())

        print(f"Test Loss: {sum(losses)/len(losses):.4e}")

        # CORRECTION : Calculer l'accuracy sur TOUTES les prédictions accumulées
        acc = accuracy(tc.tensor(Y_true_epoch), tc.tensor(Y_pred_epoch))
        
        # Nettoyage des valeurs pour le print
        if isinstance(acc, tc.Tensor): acc_print = acc.cpu().item()
        else: acc_print = acc

        print(f"Test Accuracy: {acc_print:.4f}")
        return acc_print