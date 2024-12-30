import numpy as np
from tqdm import tqdm
from cvxopt import matrix, solvers
import multiprocessing
from dataloader import DataLoader
from datetime import datetime
from sklearn import svm
import pickle
import pandas as pd

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
            # np.save(f"output/weights/weights_{i}.npy", self.weights[i])
            # np.save(f"output/bias/bias_{i}.npy", self.bias[i])

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
        P = matrix(P + 1e-3 * np.eye(self.n_samples))

        q = matrix(-np.ones(self.n_samples))

        # inequality constraints: Gx <= h => 0 <= alpha <= C
        G = matrix(np.vstack((-np.eye(self.n_samples), np.eye(self.n_samples))))
        h = matrix(
            np.hstack((np.zeros(self.n_samples), np.ones(self.n_samples) * C)))

        # equality constriants: Ax = b
        A = matrix(labels, (1, self.n_samples), 'd')
        b = matrix(0.0)

        solution = solvers.qp(
            P, q, G, h, A, b,  kktsolver='ldl', options={'kktreg': 1e-3, 'maxiters': 15})
        alphas = np.ravel(solution['x'])

        # get the support vectors
        sv = alphas > 1e-5

        # get the weights
        weights = np.dot(features.T, alphas * labels)

        # get the bias
        # print(labels[sv] - np.dot(features[sv], weights.T).squeeze())
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


# use sklearn SVM
class SK_SVM:
    def __init__(self,
                 features,
                 labels,
                 kernel_type: str = "rbf",
                 if_array=False):
        self.if_array = if_array

        self.features = features
        self.labels = labels
        if not if_array:
            features = np.asarray(features)
            labels = np.asarray(labels)

        self.vector_dim = features.shape[1]
        self.n_samples = features.shape[0]
        self.class_dim = max(labels) + 1
        self.kernel_type = kernel_type

        # one vs rest
        self.svm = svm.SVC(kernel=kernel_type)

    # * multi-class SVM
    def one_vs_rest(self):
        print("Training one vs rest SVM...")
        self.svm.fit(self.features, self.labels)

    # * evaluate
    def evaluate(self, features, labels):
        if self.if_array:
            features = features.todense()

        pred = self.svm.predict(features)
        return np.mean(pred == labels)

    # * predict
    def predict(self, features):
        if self.if_array:
            features = features.todense()

        return self.svm.predict(features)

    # * print info
    def print_info(self):
        print('features shape:', self.features.shape)
        print('labels shape:', self.labels.shape)
        print('vector dim:', self.vector_dim)
        print('class dim:', self.class_dim)
        print('kernel type:', self.kernel_type)

    # * save model
    def save_model(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.svm, f)
        print(f"Model saved to {file_path}")

    # * load model
    def load_model(self, file_path):
        with open(file_path, 'rb') as f:
            self.svm = pickle.load(f)
        print(f"Model loaded from {file_path}")


def SK_process(train_file_path, train_label_file_path, eval_file_path,
               train_acc_list, test_acc_list, file_name_list,
               # more parameters
               itrs=5,
               if_save_pred=True,
               if_save_model=True):

    # data loader
    data_loader = DataLoader(
        train_file_path, train_label_file_path, eval_file_path)

    # iterate over the data
    itrs = 5
    train_acc = 0
    test_acc = 0

    # store the best model
    max_acc = 0
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    father_dir = "output"
    file_name = "svm_model_{}".format(time_str)
    store_path = f"{father_dir}/{file_name}.pkl"

    svm = SK_SVM(data_loader.train_features,
                 data_loader.train_labels, kernel_type="rbf")
    # svm = SVM(data_loader.train_features, data_loader.train_labels)

    for i in range(itrs):

        print(f"iteration {i + 1}")

        data_loader.random_split()

        # train the model
        svm.one_vs_rest()

        # evaluate the model: train
        temp_train_acc = svm.evaluate(
            data_loader.train_features, data_loader.train_labels)
        train_acc += temp_train_acc

        # evaluate the model: test
        temp_test_acc = svm.evaluate(
            data_loader.test_features, data_loader.test_labels)
        test_acc += temp_test_acc

        print(f"train acc: {temp_train_acc}")
        print(f"test acc: {temp_test_acc}")

        if temp_test_acc > max_acc:
            max_acc = temp_test_acc

            # save the model
            if if_save_model:
                print("Saving the model...")
                svm.save_model(store_path)

            # predict the evaluation data
            if if_save_pred:
                print("Predicting the evaluation data...")

                pred = svm.predict(data_loader.eval_features)

                # Save predictions to a CSV file
                predictions = pd.DataFrame({'ID': range(0, len(pred)),
                                            'Label': pred})
                predictions.to_csv(
                    f"result/result_{time_str}.csv", index=False)

    print(f"average train acc: {train_acc / itrs}")
    print(f"average test acc: {test_acc / itrs}")

    train_acc_list.append(train_acc / itrs)
    test_acc_list.append(test_acc / itrs)
    file_name_list.append(file_name)


def process(train_file_path, train_label_file_path, eval_file_path):

    # sample usage
    sample_len = 3000

    # data loader
    data_loader = DataLoader(
        train_file_path, train_label_file_path, eval_file_path)
    data_loader.random_split()

    train_features = data_loader.train_features[:sample_len]
    train_labels = data_loader.train_labels[:sample_len]

    svm = SVM(train_features,
              train_labels)

    # train the model
    svm.one_vs_rest()

    # evaluate the model: train
    train_acc = svm.evaluate(
        train_features, train_labels)

    # evaluate the model: test
    test_acc = svm.evaluate(
        data_loader.test_features, data_loader.test_labels)

    print(f"average train acc: {train_acc}")
    print(f"average test acc: {test_acc}")


if __name__ == '__main__':

    dim_list = [10000]

    train_acc_list = []
    test_acc_list = []
    file_name_list = []

    for itr in dim_list:

        # get data
        train_file_path = "datasets/pca_reduced_all/pca_train_feature_100.pkl"
        train_label_file_path = 'datasets/train_labels.npy'
        eval_file_path = "datasets/pca_reduced_all/pca_test_feature_100.pkl"
        # process
        # process(train_file_path, train_label_file_path, eval_file_path,
        #         train_acc_list, test_acc_list, file_name_list,
        #         itrs=5,
        #         if_save_pred=False,
        #         if_save_model=False)
        process(train_file_path, train_label_file_path, eval_file_path)
