import numpy as np
from tqdm import tqdm
import cvxpy as cp
from dataloader import DataLoader

import matplotlib.pyplot as plt


class SVM:
    def __init__(self, features, labels, kernel_type: str = "linear"):
        self.features = features
        self.labels = labels
        self.vector_dim = features.shape[1]
        self.n_samples = features.shape[0]
        self.class_dim = max(labels) + 1
        self.kernel_type = kernel_type

        # one vs rest
        self.weights = np.zeros((self.class_dim, self.vector_dim))
        self.bias = np.zeros(self.class_dim)

    # * multi-class SVM
    def one_vs_rest(self):
        print("Training one vs rest SVM...")

        for i in tqdm(range(self.class_dim)):
            labels = self.labels.copy()

            # set the label of the i-th class to 1 and the others to -1
            for j in range(self.n_samples):
                if labels[j] == i:
                    labels[j] = 1
                else:
                    labels[j] = -1

            # apply one vs one
            self.weights[i], self.bias[i] = self.one_vs_one(
                self.features, labels)

    # * two-class SVM
    def kernel(self, X):

        if self.kernel_type == "linear":
            return X @ X.T

        elif self.kernel_type == "rbf":
            gamma = 1 / self.vector_dim
            return np.exp(-gamma * np.linalg.norm(X[:, None] - X, axis=2) ** 2)

        else:
            raise ValueError("Invalid kernel type")

    def one_vs_one(self, features, labels, C=1.0):
        # Initialize variables
        alphas = cp.Variable(self.n_samples)  # Lagrange multipliers

        # Kernel matrix (linear kernel assumed here; can be replaced with others)
        K = self.kernel(features)
        Q = np.outer(labels, labels) * K
        Q += 1E-5 * np.eye(self.n_samples)

        # Objective function: maximize dual problem with slack penalties
        objective = cp.Maximize(
            cp.sum(alphas) - 0.5 * cp.quad_form(alphas, Q))

        # Constraints
        constraints = [
            alphas >= 0,                  # Alphas should be non-negative
            alphas <= C,                  # Alphas should be bounded by C
            labels @ alphas == 0,         # Equality constraint for alphas
        ]

        # Solve the problem
        problem = cp.Problem(objective, constraints)
        problem.solve()

        # Retrieve results
        alpha_values = alphas.value
        support_vector_indices = np.where(alpha_values > 1e-5)[0]

        # Compute weights and bias
        weights = np.sum(alpha_values * labels * features.T, axis=1)
        bias = np.mean(labels[support_vector_indices]
                       - np.dot(features[support_vector_indices], weights))

        return weights, bias

    # * evaluate
    def predict(self, X):
        return np.argmax([np.dot(X, self.weights[i]) + self.bias[i] for i in range(self.class_dim)], axis=0)

    def evaluate(self, features, labels):
        pred = self.predict(features)
        return np.mean(pred == labels)

    # * print info
    def print_info(self):
        print('features shape:', self.features.shape)
        print('labels shape:', self.labels.shape)
        print('vector dim:', self.vector_dim)
        print('class dim:', self.class_dim)
        print('kernel type:', self.kernel_type)


if __name__ == '__main__':

    # get data
    train_file_path = 'datasets/train_feature.pkl'
    train_label_file_path = 'datasets/train_labels.npy'
    eval_file_path = 'datasets/test_feature.pkl'

    data_loader = DataLoader(
        train_file_path,
        train_label_file_path,
        eval_file_path)

    # build SVM model
    svm = SVM(data_loader.test_features, data_loader.test_labels)

    # train
    svm.one_vs_rest()

    # evaluate
    print("Train accuracy:", svm.evaluate(
        data_loader.test_features, data_loader.test_labels))

    svm.print_info()
