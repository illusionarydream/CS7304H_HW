import numpy as np
import pickle
from sklearn.decomposition import PCA


class PCA:
    def __init__(self, n_components):
        """
        Initialize the PCA class
        :param n_components: Number of dimensions to reduce to
        """
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        """
        Compute the principal components
        :param X: Input data, shape (n_samples, n_features)
        """
        # Compute the mean and center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Compute the covariance matrix
        covariance_matrix = np.cov(X_centered, rowvar=False)

        # Compute the eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Sort the eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Select the top n_components eigenvectors
        self.components = eigenvectors[:, :self.n_components]

    def transform(self, X):
        """
        Project the data to the lower-dimensional space
        :param X: Input data, shape (n_samples, n_features)
        :return: Transformed data, shape (n_samples, n_components)
        """
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

    def fit_transform(self, X):
        """
        Compute the principal components and project the data
        :param X: Input data, shape (n_samples, n_features)
        :return: Transformed data, shape (n_samples, n_components)
        """
        self.fit(X)
        return self.transform(X)


def reduce_data_for_all(train_path, test_path):

    train_data = pickle.load(open(train_path, "rb"))
    test_data = pickle.load(open(test_path, "rb"))

    train_data = train_data.toarray()
    test_data = test_data.toarray()

    all_data = np.concatenate((train_data, test_data), axis=0)

    n_components_list = [100, 500, 1000, 5000]

    for n_components in n_components_list:
        # build pca
        pca = PCA(n_components=n_components)

        # fit and transform the data
        data_reduced = pca.fit_transform(all_data)

        # store the PCA reduced data
        pickle.dump(data_reduced[:train_data.shape[0]], open(
            "datasets/pca_reduced_all/pca_train_feature_{}.pkl".format(n_components), "wb"))
        pickle.dump(data_reduced[train_data.shape[0]:], open(
            "datasets/pca_reduced_all/pca_test_feature_{}.pkl".format(n_components), "wb"))

        print(f"Data reduced to {n_components} dimensions")


# Example usage
if __name__ == "__main__":
    train_path = "datasets/train_feature.pkl"
    test_path = "datasets/test_feature.pkl"

    reduce_data_for_all(train_path, test_path)
