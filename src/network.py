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
        self.optimizer = tc.optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = tc.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

        # stockage des prédictions
        self.train_preds = None
        self.train_true = None
        self.val_preds = None
        self.val_true = None
        self.test_preds = None
        self.test_true = None

    def create_loaders(self, features_path, labels_path, batch_size=64, max_length=128):
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            features_path=features_path,
            labels_path=labels_path,
            batch_size=batch_size,
            max_length=max_length)

    def train(self, num_epochs):
        train_losses, val_losses = [], []

        # stocker toutes les prédictions de la dernière époque uniquement
        # (plus utile et cohérent)
        for epoch in range(num_epochs):
            print(("--------------------------------"))
            print(f"--- Epoch {epoch+1}/{num_epochs} ---")
            # ------- TRAIN -------
            self.model.train()
            Y_true_train, Y_pred_train = [], []
            batch_losses = []
            train_accuracy = []

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
                Y_true_train.extend(labels.cpu().numpy())
                Y_pred_train.extend(tc.argmax(outputs, dim=1).cpu().numpy())

            train_losses.append(sum(batch_losses)/len(batch_losses))
            train_accuracy.append(accuracy(tc.tensor(Y_true_train), tc.tensor(Y_pred_train)))

            # garder les prédictions de la dernière époque
            if epoch == num_epochs - 1:
                self.train_preds = Y_pred_train
                self.train_true  = Y_true_train

            # scheduler
            self.scheduler.step()

            # ------- VALIDATION -------
            self.model.eval()
            Y_true_val, Y_pred_val = [], []
            val_batch_losses = []
            val_accuracy = []

            with tc.no_grad():
                for batch_idx, batch in enumerate(self.val_loader,1):
                    inputs, labels = batch
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    val_batch_losses.append(loss.item())
                    Y_true_val.extend(labels.cpu().numpy())
                    Y_pred_val.extend(tc.argmax(outputs, dim=1).cpu().numpy())

            val_losses.append(sum(val_batch_losses)/len(val_batch_losses))
            val_accuracy.append(accuracy(tc.tensor(Y_true_val), tc.tensor(Y_pred_val)))

            print(f"Train Loss: {train_losses[-1]:.4e} | Val Loss: {val_losses[-1]:.4e}")
            print(f"Train Accuracy: {train_accuracy[-1]:.4f} | Val Accuracy: {val_accuracy[-1]:.4f}")
            if epoch == num_epochs - 1:
                self.val_preds = Y_pred_val
                self.val_true  = Y_true_val

        return train_losses, val_losses, train_accuracy, val_accuracy

    def test(self):
        self.model.eval()
        Y_true_test, Y_pred_test = [], []
        losses = []

        with tc.no_grad():
            for batch_idx, batch in enumerate(self.test_loader,1):
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                losses.append(loss.item())
                Y_true_test.extend(labels.cpu().numpy())
                Y_pred_test.extend(tc.argmax(outputs, dim=1).cpu().numpy())

        print(f"Test Loss: {sum(losses)/len(losses):.4e}")
        self.test_preds = Y_pred_test
        self.test_true  = Y_true_test

        acc = accuracy(tc.tensor(Y_true_test), tc.tensor(Y_pred_test))
        print(f"Test Accuracy: {acc:.4f}")
        return acc
