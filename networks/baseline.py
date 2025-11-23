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
            max_length=max_length)

    def train(self, num_epochs):
        train_losses, val_losses = [], []

        for epoch in range(num_epochs):
            print(f"\n{'='*30}\nEpoch {epoch+1}/{num_epochs}\n{'='*30}")
            # --------- Entraînement ---------
            self.model.train()
            batch_losses = []
            Y_true, Y_pred = [], []

            for i, (inputs, labels) in enumerate(self.train_loader, 1):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                batch_losses.append(loss.item())

                Y_true.extend(labels.cpu().numpy())
                Y_pred.extend(tc.argmax(outputs, dim=1).cpu().numpy())

                # print intermédiaire toutes les 10 batches
                if i % 10 == 0 or i == len(self.train_loader):
                    print(f"[Batch {i}/{len(self.train_loader)}] "
                        f"Batch Loss: {loss.item():.4f}")

            epoch_loss = sum(batch_losses) / len(batch_losses)
            acc = accuracy(tc.tensor(Y_true), tc.tensor(Y_pred))
            train_losses.append(epoch_loss)
            self.scheduler.step()

            print(f"\n-- Epoch {epoch+1} Completed --")
            print(f"Train Loss: {epoch_loss:.4f}")
            print(f"Train Accuracy: {acc:.4f}\n")

            # --------- Validation ---------
            self.model.eval()
            val_batch_losses = []
            Y_true, Y_pred = [], []

            with tc.no_grad():
                for i, (inputs, labels) in enumerate(self.val_loader, 1):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    val_batch_losses.append(loss.item())
                    Y_true.extend(labels.cpu().numpy())
                    Y_pred.extend(tc.argmax(outputs, dim=1).cpu().numpy())

                    if i % 10 == 0 or i == len(self.val_loader):
                        print(f"[Val Batch {i}/{len(self.val_loader)}] "
                            f"Batch Loss: {loss.item():.4f}")

            val_epoch_loss = sum(val_batch_losses) / len(val_batch_losses)
            val_acc = accuracy(tc.tensor(Y_true), tc.tensor(Y_pred))
            val_losses.append(val_epoch_loss)

            print(f"\n== Validation for Epoch {epoch+1} ==")
            print(f"Validation Loss: {val_epoch_loss:.4f}")
            print(f"Validation Accuracy: {val_acc:.4f}")
            print("="*30)

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