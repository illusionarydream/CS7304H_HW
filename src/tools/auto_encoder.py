import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras import layers, models
from scipy.sparse import vstack, csr_matrix


class AutoEncoder:
    def __init__(self, input_dim, latent_dim):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self._build_model()

    def _build_model(self):
        # Encoder
        input_layer = layers.Input(shape=(self.input_dim,))
        encoded = layers.Dense(128, activation='relu')(input_layer)
        encoded = layers.Dense(64, activation='relu')(encoded)
        latent = layers.Dense(self.latent_dim, activation='relu')(encoded)

        # Decoder
        decoded = layers.Dense(64, activation='relu')(latent)
        decoded = layers.Dense(128, activation='relu')(decoded)
        output_layer = layers.Dense(
            self.input_dim, activation='sigmoid')(decoded)

        # Autoencoder
        self.autoencoder = models.Model(input_layer, output_layer)
        self.encoder = models.Model(input_layer, latent)

        # Compile the model
        self.autoencoder.compile(optimizer='adam', loss='mse')

    def train(self, train_data, epochs=50, batch_size=128):

        print("Training AutoEncoder...")
        self.autoencoder.fit(train_data, train_data,
                             epochs=epochs, batch_size=batch_size, shuffle=True)

    def transform(self, data):
        return self.encoder.predict(data)

# Helper function to handle sparse matrices


def sparse_to_dense(matrix):
    return matrix.toarray()


def prepare_data(train_sparse, test_sparse):
    # Combine train and test sparse matrices
    combined_sparse = vstack([train_sparse, test_sparse])

    # Convert sparse matrix to dense for AutoEncoder
    combined_dense = sparse_to_dense(combined_sparse)

    # Split back into train and test
    train_dense = combined_dense[:train_sparse.shape[0], :]
    test_dense = combined_dense[train_sparse.shape[0]:, :]

    return train_dense, test_dense


if __name__ == "__main__":
    # Load your sparse train and test features
    train_sparse = pickle.load(open("datasets/train_feature.pkl", "rb"))
    test_sparse = pickle.load(open("datasets/test_feature.pkl", "rb"))

    # Prepare data
    train_dense, test_dense = prepare_data(train_sparse, test_sparse)

    # Initialize AutoEncoder
    input_dim = train_dense.shape[1]

    latent_dim_list = [100, 500, 5000]

    for latent_dim in latent_dim_list:

        autoencoder = AutoEncoder(input_dim=input_dim, latent_dim=latent_dim)

        # Train AutoEncoder
        all_dense = np.concatenate((train_dense, test_dense), axis=0)
        autoencoder.train(all_dense, epochs=20, batch_size=64)

        # Transform data
        train_latent = autoencoder.transform(train_dense)
        test_latent = autoencoder.transform(test_dense)

        # save the latent features as pickle files
        pickle.dump(train_latent, open(
            "datasets/AE_reduced/AE_train_feature_{0}.pkl".format(latent_dim), "wb"))
        pickle.dump(test_latent, open(
            "datasets/AE_reduced/AE_test_feature_{0}.pkl".format(latent_dim), "wb"))

        print(f"Data reduced to {latent_dim} dimensions")
