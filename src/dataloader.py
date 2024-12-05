import pickle
import numpy as np


class DataLoader:
    def __init__(self, train_file_path, train_label_file_path, eval_file_path, split_ratio=0.2):

        self.train_features = pickle.load(open(train_file_path, 'rb'))
        self.train_labels = np.load(train_label_file_path)
        self.eval_features = pickle.load(open(eval_file_path, 'rb'))

        # split the training data into training and evaluation sets
        # split_index = int(len(original_train_features) * (1 - split_ratio))

    def print_info(self):
        print('Train features shape:', self.train_features.shape)
        print('Train labels shape:', self.train_labels.shape)
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
