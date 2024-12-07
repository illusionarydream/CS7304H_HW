import numpy as np
import pickle
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_moons


def reduce_data(data_path):

    data = pickle.load(open(data_path, "rb"))

    n_components_list = [100, 500, 1000, 5000]

    for n_components in n_components_list:
        # build kpca
        kpca = KernelPCA(n_components=n_components,
                         kernel='rbf')

        # fit and transform the data
        data_reduced = kpca.fit_transform(data)

        # store the PCA reduced data
        pickle.dump(data_reduced, open(
            "datasets/kpca_train_feature_{}.pkl".format(n_components), "wb"))

        print(f"Data reduced to {n_components} dimensions")


def reduce_data_for_all(train_path, test_path):

    train_data = pickle.load(open(train_path, "rb"))
    test_data = pickle.load(open(test_path, "rb"))

    train_data = train_data.toarray()
    test_data = test_data.toarray()

    all_data = np.concatenate((train_data, test_data), axis=0)

    n_components_list = [100, 500, 1000, 5000]

    for n_components in n_components_list:
        # build kpca
        kpca = KernelPCA(n_components=n_components,
                         kernel='rbf')

        # fit and transform the data
        data_reduced = kpca.fit_transform(all_data)

        # store the PCA reduced data
        pickle.dump(data_reduced[:train_data.shape[0]], open(
            "datasets/kpca_reduced_all/kpca_train_feature_{}.pkl".format(n_components), "wb"))
        pickle.dump(data_reduced[train_data.shape[0]:], open(
            "datasets/kpca_reduced_all/kpca_test_feature_{}.pkl".format(n_components), "wb"))

        print(f"Data reduced to {n_components} dimensions")


if __name__ == "__main__":
    train_path = "datasets/train_feature.pkl"
    test_path = "datasets/test_feature.pkl"

    reduce_data_for_all(train_path, test_path)
