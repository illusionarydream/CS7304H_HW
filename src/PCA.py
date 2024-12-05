import numpy as np
import pickle


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


# Example usage
if __name__ == "__main__":
    data = pickle.load(open("datasets/train_feature.pkl", "rb"))

    pca = PCA(n_components=1000)
    data_reduced = pca.fit_transform(data)

    print(data_reduced.shape)

    # store the PCA reduced data
    pickle.dump(data_reduced, open("datasets/pca_train_feature.pkl", "wb"))
