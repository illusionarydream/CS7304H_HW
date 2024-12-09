import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import scipy.sparse as sp
import numpy as np
import pickle
from datetime import datetime
from tqdm import tqdm
import pandas as pd

# Dataset class for handling sparse data


class SparseDataset(Dataset):
    def __init__(self, features, labels=None):
        """
        Args:
            features: Sparse feature matrix (scipy sparse matrix).
            labels: Corresponding labels (None for evaluation dataset).
        """
        self.features = torch.tensor(features.todense(), dtype=torch.float32)
        self.labels = torch.tensor(
            labels, dtype=torch.long) if labels is not None else None

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]

# MLP model definition


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLP, self).__init__()
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            # layers.append(nn.Dropout(0.5))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

    def reset_parameters(self):
        for layer in self.network:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

# Function to train and evaluate the MLP model


def train_and_evaluate(model, train_loader, val_loader, eval_loader,
                       epochs,
                       lr,
                       if_save_model=True,
                       if_predict=True):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # *store path
    max_acc = 0
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    father_dir = "output/MLP"
    file_name = "mlp_model_{}".format(time_str)
    store_path = f"{father_dir}/{file_name}.pth"

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        # * Training phase
        model.train()
        train_loss = 0.0
        train_acc = 0
        train_total = 0
        train_bar = tqdm(train_loader)
        for features, labels in train_bar:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            train_acc += (predicted == labels).sum().item()
            train_total += labels.size(0)

            train_bar.set_postfix(train_loss=train_loss,
                                  train_acc=train_acc/train_total)

        # * Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        val_bar = tqdm(val_loader)
        with torch.no_grad():
            for features, labels in val_bar:
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                val_bar.set_postfix(val_loss=val_loss, val_acc=correct/total)

        # Print epoch statistics
        print(f"Epoch {epoch+1}/{epochs}, "
              f"Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, "
              f"Train Acc: {train_acc/train_total:.4f}, "
              f"Val Acc: {correct/total:.4f}")

        if max_acc < correct/total:
            max_acc = correct/total

            # * Save model
            if if_save_model:
                model.save_model(store_path)
                print(f"Model saved to {store_path}")

            # * Predict on evaluation dataset
            if if_predict:
                model.eval()
                pred = []
                with torch.no_grad():
                    for features in eval_loader:
                        outputs = model(features)
                        _, predicted = torch.max(outputs, 1)
                        pred.extend(predicted.numpy())

                # Save predictions
                predictions = pd.DataFrame({'ID': range(0, len(pred)),
                                            'Label': pred})
                predictions.to_csv(
                    f"result/result_{time_str}.csv", index=False)
                print(f"Predictions saved to result/result_{time_str}.csv")


# * MAIN FUNCTION
def processing(train_feature_path, train_label_path, eval_feature_path,
               if_predict=True,
               if_save_model=True):

    # * Load data
    train_features = pickle.load(open(train_feature_path, 'rb'))
    train_labels = np.load(train_label_path)
    eval_features = pickle.load(open(eval_feature_path, 'rb'))

    # * Create datasets
    train_dataset = SparseDataset(train_features, train_labels)

    # Split train dataset into training and validation sets
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(
        train_dataset, [train_size, val_size])

    # * Data loaders
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
    eval_loader = DataLoader(SparseDataset(
        eval_features), batch_size=64, shuffle=False)

    # * Define MLP model
    input_dim = train_features.shape[1]
    hidden_dims = [1024, 256, 64]
    output_dim = len(np.unique(train_labels))  # Number of classes

    model = MLP(input_dim, hidden_dims, output_dim)

    # * Train and evaluate and predict

    # find the best hyperparameters
    train_and_evaluate(model, train_loader, val_loader, eval_loader,
                       epochs=100, lr=0.001,
                       if_save_model=if_save_model,
                       if_predict=if_predict)


# Main function
if __name__ == "__main__":
    # data path
    train_feature_path = "datasets/train_feature.pkl"
    train_label_path = "datasets/train_labels.npy"
    eval_feature_path = "datasets/test_feature.pkl"

    processing(train_feature_path, train_label_path, eval_feature_path,
               if_predict=False,
               if_save_model=False)
