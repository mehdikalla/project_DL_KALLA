import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from models.cnn import CNN
from dataset.data_loader import get_dataloaders
from utils.metrics import accuracy

class CNNet():
    def __init__(self, device):
        self.device = device
        self.model = CNN().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = tc.optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = tc.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

    def create_loaders(self, features_path, labels_path, batch_size=64, max_length=128):
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            features_path=features_path,
            labels_path=labels_path,
            batch_size=batch_size,
            max_length=max_length
        )

    def train(self, num_epochs):
        train_losses, val_losses = [], []
        Y_true, Y_pred = [], []
        for epoch in range(num_epochs):

            # --------- Entraînement ---------
            self.model.train()
            batch_losses = []
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                batch_losses.append(loss.item())
                Y_true.extend(labels.cpu().numpy())
                Y_pred.extend(tc.argmax(outputs, dim=1).cpu().numpy())
            
            acc = accuracy(tc.tensor(Y_true), tc.tensor(Y_pred))
            epoch_loss = sum(batch_losses) / len(batch_losses)
            self.scheduler.step()
            train_losses.append(epoch_loss)
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}')
            print(f'Accuracy after epoch {epoch+1}: {acc:.4f}')

            # --------- Validation ---------
            self.model.eval()
            val_batch_losses = []
            Y_true, Y_pred = [], []
            with tc.no_grad():
                for inputs, labels in self.val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    val_batch_losses.append(loss.item())
                    Y_true.extend(labels.cpu().numpy())
                    Y_pred.extend(tc.argmax(outputs, dim=1).cpu().numpy())
            
            acc = accuracy(tc.tensor(Y_true), tc.tensor(Y_pred))
            print(f'Validation Accuracy after epoch {epoch+1}: {acc:.4f}')
            val_epoch_loss = sum(val_batch_losses) / len(val_batch_losses)
            val_losses.append(val_epoch_loss)
            print(f'Validation Loss: {val_epoch_loss:.4f}')

        return train_losses, val_losses
    
    def test(self):
        self.model.eval()
        test_losses = []
        Y_true, Y_pred = [], []
        with tc.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                test_losses.append(loss.item())
                Y_true.extend(labels.cpu().numpy())
                Y_pred.extend(tc.argmax(outputs, dim=1).cpu().numpy())
                
        acc = accuracy(tc.tensor(Y_true), tc.tensor(Y_pred))
        print(f'Test Accuracy: {acc:.4f}')
        return acc