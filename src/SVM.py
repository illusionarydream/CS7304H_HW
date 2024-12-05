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
        for i in range(self.class_dim):
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

    def one_vs_one(self, features, labels):
        # initialize variables
        alphas = cp.Variable(self.n_samples)

        # kernel matrix
        K = self.kernel(features)
        Q = np.outer(labels, labels) * K
        Q += 1E-5 * np.eye(self.n_samples)

        # objective function
        objective = cp.Maximize(cp.sum(alphas) - 0.5 * cp.quad_form(alphas, Q))
        # constraints
        constraints = [alphas >= 0, labels @ alphas == 0]

        # solve the problem
        problem = cp.Problem(objective, constraints)
        problem.solve()

        # result
        alpha_values = alphas.value
        support_vector_indices = np.where(alpha_values > 1e-5)[0]

        # calculate weights and bias
        weights = np.sum(alpha_values[support_vector_indices]
                         * labels[support_vector_indices, None]
                         * features[support_vector_indices], axis=0)
        bias = np.mean(labels[support_vector_indices]
                       - np.dot(features[support_vector_indices], weights))

        return weights, bias

    def print_info(self):
        print('features shape:', self.features.shape)
        print('labels shape:', self.labels.shape)
        print('vector dim:', self.vector_dim)
        print('class dim:', self.class_dim)
        print('kernel type:', self.kernel_type)


def plot_data(X, y, w, b):

    # plot data
    plt.scatter(X[:, 0], X[:, 1], c=y)
    # plot y = wx + b
    x = np.linspace(0, 4, 100)
    y = (-w[0] * x - b) / w[1]
    plt.plot(x, y, '-r')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('SVM Classification')
    plt.show()


if __name__ == '__main__':

    # # get data
    # train_file_path = 'datasets/train_feature.pkl'
    # train_label_file_path = 'datasets/train_labels.npy'
    # eval_file_path = 'datasets/test_feature.pkl'

    # data_loader = DataLoader(
    #     train_file_path,
    #     train_label_file_path,
    #     eval_file_path)

    # # build SVM model
    # svm = SVM(data_loader.train_features, data_loader.train_labels)

    # svm.print_info()

    X = np.array([[1.0, 3.0], [1.0, 1.0], [2.0, 3.0], [3.0, 4.0], [3.0, 2.0]])
    y = np.array([1, 1, 0, 0, 0])

    svm = SVM(X, y)
    svm.one_vs_rest()

    print(svm.weights)
    print(svm.bias)

    plot_data(X, y, svm.weights[0], svm.bias[0])
