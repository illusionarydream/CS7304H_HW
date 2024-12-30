import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap
import pickle
from sklearn.decomposition import PCA


def embeddingTo2D(data, method: str = 'umap'):
    if method == 'umap':
        reducer = umap.UMAP()
        embedding = reducer.fit_transform(data)
        return embedding

    elif method == 'pca':
        reducer = PCA(n_components=2)
        embedding = reducer.fit_transform(data)
        return embedding
    else:
        raise ValueError('Unknown method')


def plot2D(embedding, labels, save_path=None, figsize=(12, 9)):
    plt.figure(figsize=figsize)
    classes = set(labels)
    color_map = plt.cm.get_cmap('tab10', len(classes))

    for i, class_label in enumerate(classes):
        indices = [j for j, label in enumerate(labels) if label == class_label]
        x = embedding[indices, 0]
        y = embedding[indices, 1]
        plt.scatter(x, y, c=color_map(i), label=class_label)

    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    data = pickle.load(
        open("datasets/AE_reduced/AE_train_feature_100.pkl", "rb"))
    data = np.asarray(data)

    # Normalize the data
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    # Apply t-SNE
    embedding = embeddingTo2D(data, method='umap')

    # Plot the results
    labels = np.load("datasets/train_labels.npy")
    plot2D(embedding, labels, save_path='output/image/umap_AE.png')
