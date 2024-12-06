import pickle
import numpy as np


class DataLoader:
    def __init__(self, train_file_path, train_label_file_path, eval_file_path, random_seed=42):

        self.origin_features = pickle.load(open(train_file_path, 'rb'))
        self.origin_labels = np.load(train_label_file_path)
        self.eval_features = pickle.load(open(eval_file_path, 'rb'))

        self.train_features = None
        self.train_labels = None
        self.test_features = None
        self.test_labels = None

        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        self.random_shuffle()
        self.random_split()

    def random_shuffle(self):
        # shuffle the data
        indices = np.arange(self.origin_features.shape[0])
        np.random.shuffle(indices)
        self.origin_features = self.origin_features[indices]
        self.origin_labels = self.origin_labels[indices]

    def random_split(self, split_ratio=0.2):
        # random shuffle
        self.random_shuffle()
        # split the training data into training and evaluation sets
        split_index = int(self.origin_features.shape[0] * (1 - split_ratio))

        self.train_features = self.origin_features[:split_index]
        self.train_labels = self.origin_labels[:split_index]
        self.test_features = self.origin_features[split_index:]
        self.test_labels = self.origin_labels[split_index:]

    def print_info(self):
        print('Train features shape:', self.train_features.shape)
        print('Train labels shape:', self.train_labels.shape)
        print('Test features shape:', self.test_features.shape)
        print('Test labels shape:', self.test_labels.shape)
        print('Evaluate features shape:', self.eval_features.shape)


if __name__ == '__main__':

    train_file_path = 'datasets/train_feature.pkl'
    train_label_file_path = 'datasets/train_labels.npy'
    eval_file_path = 'datasets/test_feature.pkl'

    data_loader = DataLoader(
        train_file_path,
        train_label_file_path,
        eval_file_path)

    data_loader.print_info()
