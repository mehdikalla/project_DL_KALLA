import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from models.cnn import CNN
from dataset.data_loader import get_dataloaders
from utils.metrics import accuracy, confusion_matrix, save_logs
from utils.visualization import plot_loss_curve, plot_metrics_curve

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
            epoch_loss = sum(batch_losses) / len(batch_losses)
            self.scheduler.step()
            train_losses.append(epoch_loss)
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}')

            # --------- Validation ---------
            self.model.eval()
            val_batch_losses = []
            with tc.no_grad():
                for inputs, labels in self.val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    val_batch_losses.append(loss.item())
            val_epoch_loss = sum(val_batch_losses) / len(val_batch_losses)
            val_losses.append(val_epoch_loss)
            print(f'Validation Loss: {val_epoch_loss:.4f}')

        return train_losses, val_losses
    
    def test(self):
        self.model.eval()
        correct = 0
        total = 0
        test_losses = []
        with tc.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                test_losses.append(loss.item())
                _, predicted = tc.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        test_loss = sum(test_losses) / len(test_losses)
        accuracy = 100 * correct / total
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%')
        return accuracy