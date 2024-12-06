import numpy as np
import pickle
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_moons


def reduce_data(data_path):

    data = pickle.load(open(data_path, "rb"))

    N_list = [100, 500, 1000, 5000]

    for n_components in N_list:
        # build kpca
        kpca = KernelPCA(n_components=n_components,
                         kernel='rbf')

        # fit and transform the data
        data_reduced = kpca.fit_transform(data)

        # store the PCA reduced data
        pickle.dump(data_reduced, open(
            "datasets/kpca_test_feature_{}.pkl".format(n_components), "wb"))

        print(f"PCA reduced data shape (n_components={n_components}):",
              data_reduced.shape)


if __name__ == "__main__":
    data_path = "datasets/test_feature.pkl"
    reduce_data(data_path)
