from dataloader import DataLoader
import numpy as np
from tqdm import tqdm
from datetime import datetime
import pandas as pd


class WarmupCosineDecayScheduler:
    def __init__(self, max_iter, base_lr, warmup_iter, min_lr=0.0):

        self.max_iter = max_iter
        self.base_lr = base_lr
        self.warmup_iter = warmup_iter
        self.min_lr = min_lr

    def get_lr(self, current_iter):

        if current_iter < self.warmup_iter:
            # Warmup phase
            return self.base_lr * (current_iter / self.warmup_iter)
        else:
            # Cosine Decay phase
            progress = (current_iter - self.warmup_iter) / \
                (self.max_iter - self.warmup_iter)
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            return self.min_lr + (self.base_lr - self.min_lr) * cosine_decay


class CLRScheduler:
    def __init__(self, max_iter, base_lr, warmup_iter, min_lr=0.0, cycle_length=10):
        self.max_iter = max_iter
        self.base_lr = base_lr
        self.warmup_iter = warmup_iter
        self.min_lr = min_lr
        self.cycle_length = cycle_length

    def get_lr(self, current_iter):
        if current_iter < self.warmup_iter:
            # Warmup phase
            return self.base_lr * (current_iter / self.warmup_iter)
        else:
            # Cyclical Learning Rate phase
            cycle = np.floor(1 + current_iter / (2 * self.cycle_length))
            x = np.abs(current_iter / self.cycle_length - 2 * cycle + 1)
            lr = self.min_lr + (self.base_lr - self.min_lr) * max(0, (1 - x))
            return lr


class LogisticRegression:
    def __init__(self, feature_dim, class_dim):
        self.feature_dim = feature_dim
        self.class_dim = class_dim

        self.theta = np.zeros((self.class_dim, self.feature_dim))
        self.scheduler = None

    def sigmoid(self, theta, x):
        return 1 / (1 + np.exp(-np.dot(theta, x)))

    def loss(self, X, y, theta, m):
        Loss = 0
        for idx in range(m):
            sig = self.sigmoid(theta, X[idx])
            Loss += y[idx] * np.log(max(sig, 1e-10)) + \
                (1 - y[idx]) * np.log(max(1 - sig, 1e-10))

        return -Loss / m

    def optimizer(self, max_iter, learning_rate):
        learning_rates = []
        for i in range(max_iter):
            if i < max_iter/2:
                learning_rates.append(learning_rate * (i+1) / (max_iter/2))
            else:
                learning_rates.append(
                    learning_rate * (max_iter - i) / (max_iter/2))
        return learning_rates

    def calculate_gradient(self, X, y, theta, m, type='SGD'):

        if type == 'SGD':
            gradient = np.zeros(self.feature_dim)

            for idx in range(m):
                gradient += (self.sigmoid(theta, X[idx]) - y[idx]) * X[idx]

            return gradient / m

        elif type == 'Newton':
            Hessian = np.zeros((self.feature_dim, self.feature_dim))
            gradient = np.zeros(self.feature_dim)

            for idx in range(m):
                # Hessian
                Hessian += self.sigmoid(theta, X[idx]) * \
                    (1 - self.sigmoid(theta, X[idx])
                     ) * np.outer(X[idx], X[idx])
                # gradient
                gradient += (self.sigmoid(theta, X[idx]) - y[idx]) * X[idx]

            Hessian /= m
            gradient /= m

            return np.dot(np.linalg.inv(Hessian), gradient)

    def one_vs_rest(self, feature, label, i,
                    max_iter=1000, learning_rate=0.001):
        # discriminant function
        y = np.zeros(label.shape)
        y[label == i] = 1
        y[label != i] = 0
        X = feature
        m = feature.shape[0]

        # weight
        theta = np.zeros(self.feature_dim)

        # Loop
        bar = tqdm(range(max_iter))
        for _ in bar:

            loss = self.loss(X, y, theta, m)

            gradient = self.calculate_gradient(X, y, theta, m)

            theta -= learning_rate * gradient

            bar.set_description("Loss: {:.6f}".format(loss))

        return theta

    def train(self, train_features, train_labels, max_iter=50, learning_rate=5000):
        train_features = train_features.toarray()

        # * build scheduler
        self.scheduler = WarmupCosineDecayScheduler(
            max_iter, learning_rate, max_iter//10, learning_rate/10)

        # * Train
        for i in range(self.class_dim):
            print(f"Training class {i}")

            self.theta[i] = self.one_vs_rest(
                train_features, train_labels, i, max_iter, learning_rate)

    def predict(self, features):
        features = features.toarray()
        pred = np.zeros(features.shape[0])
        for idx in range(features.shape[0]):
            pred[idx] = np.argmax(
                [self.sigmoid(theta, features[idx]) for theta in self.theta])

        return pred

    def store_model(self, store_path):
        np.save(store_path, self.theta)

    def load_model(self, model_path):
        self.theta = np.load(model_path)


def processing(train_feature_path, train_label_path, eval_feature_path,
               if_save_model=True,
               store_path=None,
               if_predict=True):

    # * load data
    data_loader = DataLoader(
        train_feature_path, train_label_path, eval_feature_path)

    # * store path
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    father_dir = "output/LR"
    file_name = "LR_model_{}".format(time_str)
    store_path = f"{father_dir}/{file_name}.pth"

    # * train
    feature_shape = data_loader.train_features.shape[1]
    class_dim = data_loader.train_labels.max() + 1
    model = LogisticRegression(feature_shape, class_dim)

    data_loader.random_split()
    model.train(data_loader.train_features, data_loader.train_labels)

    # * predict
    train_pred = model.predict(data_loader.train_features)
    test_pred = model.predict(data_loader.test_features)

    train_acc = (train_pred == data_loader.train_labels).mean()
    print(f"Train Accuracy: {train_acc}")
    test_acc = (test_pred == data_loader.test_labels).mean()
    print(f"Test Accuracy: {test_acc}")

    # * store model
    if if_save_model:
        model.store_model(store_path)
        print(f"Model saved to {store_path}")

    # * predict on evaluation dataset
    if if_predict:

        pred = model.predict(data_loader.eval_features)

        predictions = pd.DataFrame({'ID': range(0, len(pred)),
                                    'Label': pred})
        predictions.to_csv(
            f"result/result_{time_str}.csv", index=False)
        print(f"Predictions saved to result/result_{time_str}.csv")


if __name__ == '__main__':
    train_file_path = 'datasets/train_feature.pkl'
    train_label_file_path = 'datasets/train_labels.npy'
    eval_file_path = 'datasets/test_feature.pkl'

    processing(train_file_path, train_label_file_path, eval_file_path)
