from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pandas as pd

from dataloader import DataLoader


if __name__ == '__main__':

    # get data
    train_file_path = 'datasets/train_feature.pkl'
    train_label_file_path = 'datasets/train_labels.npy'
    eval_file_path = 'datasets/test_feature.pkl'

    data_loader = DataLoader(
        train_file_path,
        train_label_file_path,
        eval_file_path)

    data_loader.print_info()

    # build model
    clf = svm.SVC(kernel='linear')

    # classify on the training data
    clf.fit(data_loader.train_features, data_loader.train_labels)

    # evaluate on the training data
    accuracy = clf.score(data_loader.train_features, data_loader.train_labels)

    # predict on the eval data
    eval_predictions = clf.predict(data_loader.eval_features)

    # save predictions to a CSV file
    df = pd.DataFrame({
        'ID': range(len(eval_predictions)),
        'label': eval_predictions
    })
    df.to_csv('test_predictions.csv', index=False)
    print('Predictions saved to test_predictions.csv')
