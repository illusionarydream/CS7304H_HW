import numpy as np
from tqdm import tqdm
from cvxopt import matrix, solvers
import multiprocessing
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
    def one_vs_rest(self, C=1.0):
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

            # store the weights and bias
            np.save(f"output/weights/weights_{i}.npy", self.weights[i])
            np.save(f"output/bias/bias_{i}.npy", self.bias[i])

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

        # get the kernel matrix
        K = self.kernel(features)
        P = matrix(np.outer(labels, labels) * K)
        P = matrix(P + 1e-5 * np.eye(self.n_samples))

        q = matrix(-np.ones(self.n_samples))

        # inequality constraints: Gx <= h => 0 <= alpha <= C
        G = matrix(np.vstack((-np.eye(self.n_samples), np.eye(self.n_samples))))
        h = matrix(
            np.hstack((np.zeros(self.n_samples), np.ones(self.n_samples) * C)))

        # equality constriants: Ax = b
        A = matrix(labels, (1, self.n_samples), 'd')
        b = matrix(0.0)

        solution = solvers.qp(
            P, q, G, h, A, b,  kktsolver='ldl', options={'kktreg': 1e-9})
        alphas = np.ravel(solution['x'])

        # get the support vectors
        sv = alphas > 1e-5

        # get the weights
        weights = np.dot(features.T, alphas * labels)

        # get the bias
        bias = np.mean(labels[sv] - np.dot(features[sv], weights.T).squeeze())

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
    train_file_path = 'datasets/pca_train_feature.pkl'
    train_label_file_path = 'datasets/train_labels.npy'
    eval_file_path = 'datasets/test_feature.pkl'

    data_loader = DataLoader(
        train_file_path,
        train_label_file_path,
        eval_file_path)

    # build SVM model
    svm = SVM(data_loader.train_features,
              data_loader.train_labels, kernel_type='linear')

    # train
    svm.one_vs_rest()

    # evaluate
    print("Train accuracy:", svm.evaluate(
        data_loader.train_features, data_loader.train_labels))

    svm.print_info()
