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

        # stockage des prédictions
        self.train_preds = None
        self.train_true = None
        self.val_preds = None
        self.val_true = None
        self.test_preds = None
        self.test_true = None

    def train(self, num_epochs):
        train_losses, val_losses = [], []

        # stocker toutes les prédictions de la dernière époque uniquement
        # (plus utile et cohérent)
        for epoch in range(num_epochs):

            # ------- TRAIN -------
            self.model.train()
            Y_true_train, Y_pred_train = [], []
            batch_losses = []

            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                batch_losses.append(loss.item())
                Y_true_train.extend(labels.cpu().numpy())
                Y_pred_train.extend(tc.argmax(outputs, dim=1).cpu().numpy())

            train_losses.append(sum(batch_losses)/len(batch_losses))
            acc_train = accuracy(tc.tensor(Y_true_train), tc.tensor(Y_pred_train))

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

            with tc.no_grad():
                for inputs, labels in self.val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                    val_batch_losses.append(loss.item())
                    Y_true_val.extend(labels.cpu().numpy())
                    Y_pred_val.extend(tc.argmax(outputs, dim=1).cpu().numpy())

            val_losses.append(sum(val_batch_losses)/len(val_batch_losses))
            acc_val = accuracy(tc.tensor(Y_true_val), tc.tensor(Y_pred_val))

            if epoch == num_epochs - 1:
                self.val_preds = Y_pred_val
                self.val_true  = Y_true_val

        return train_losses, val_losses, acc_train, acc_val

    def test(self):
        self.model.eval()
        Y_true_test, Y_pred_test = [], []
        losses = []

        with tc.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                losses.append(loss.item())
                Y_true_test.extend(labels.cpu().numpy())
                Y_pred_test.extend(tc.argmax(outputs, dim=1).cpu().numpy())

        self.test_preds = Y_pred_test
        self.test_true  = Y_true_test

        acc = accuracy(tc.tensor(Y_true_test), tc.tensor(Y_pred_test))
        print(f"Test Accuracy: {acc:.4f}")
        return acc
