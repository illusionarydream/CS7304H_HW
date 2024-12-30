import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
import seaborn as sns
from tqdm import tqdm

# 设置Seaborn的样式
sns.set(style="whitegrid")


def feature_distributed(data):
    """
    统计并绘制特定特征的数值分布图

    参数:
    data (numpy.ndarray): 包含特征数据的二维数组
    """
    # 将二维特征数组压缩成一维数组
    feature_data = data[:1000, :].toarray().flatten()

    # 绘制特征的数值分布图
    plt.figure(figsize=(12, 8))
    sns.histplot(feature_data, color='skyblue', bins=30)
    plt.yscale('log')  # 设置纵轴为对数刻度
    plt.title('Distribution of Features', fontsize=16)
    plt.xlabel('Feature Values', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.savefig('image/Normalized_distribution_of_features.png')


def zero_nums(data):
    """
    统计数据集中特征值为0的数量

    参数:
    data (numpy.ndarray): 包含特征数据的二维数组
    """
    # 统计特征值为0的数量
    zero_ratio_list = []
    for i in tqdm(range(data.shape[1])):    # 遍历所有特征
        nonzero_nums = np.sum(data[:, i] != 0)
        zero_ratio = 1 - nonzero_nums / data.shape[0]

        zero_ratio_list.append(zero_ratio)

    # 绘制特征值为0的数量分布图
    plt.figure(figsize=(12, 8))
    sns.histplot(zero_ratio_list, color='skyblue', bins=30)
    plt.title('Distribution of Zero Ratios', fontsize=16)
    plt.yscale('log')
    plt.xlabel('Zero Ratios', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True, which="both", ls="--", linewidth=0.5)


def pearson_check(data):
    """
    计算特征之间的皮尔逊相关系数

    参数:
    data (numpy.ndarray): 包含特征数据的二维数组
    """
    # 计算特征之间的皮尔逊相关系数
    feature_index = np.random.choice(data.shape[1], 20, replace=False)
    sample_len = 2000
    data = data[:sample_len, feature_index].toarray()
    corr = np.corrcoef(data, rowvar=False)

    # 绘制特征之间的皮尔逊相关系数热力图
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title('Pearson Correlation Coefficients', fontsize=16)
    plt.savefig('image/Pearson_correlation_coefficients.png')


def mutual_info_check(data, label):
    """
    计算特征之间的互信息

    参数:
    data (numpy.ndarray): 包含特征数据的二维数组
    """
    feature_index = np.random.choice(data.shape[1], 200, replace=False)
    sample_len = 2000
    data = data[:sample_len, feature_index].toarray()

    # 计算特征之间的互信息
    mi = mutual_info_classif(data, label[:sample_len])
    mi = np.sort(mi)[::-1]

    # 绘制特征之间的互信息柱状图
    plt.figure(figsize=(12, 8))
    plt.bar(np.arange(len(mi)), mi, width=1.0, color='skyblue')
    plt.title('Mutual Information', fontsize=16)
    plt.xlabel('Feature Index', fontsize=14)
    plt.ylabel('Mutual Information', fontsize=14)
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.savefig('image/Mutual_information.png')
    # plt.show()


if __name__ == "__main__":
    data = pickle.load(open("datasets/normalized_train_feature.pkl", "rb"))
    label = np.load("datasets/train_labels.npy")

    # distribution of features
    # feature_distributed(data)

    # zero ratio statistics
    # zero_nums(data)

    # pearson correlation coefficients
    # pearson_check(data)

    # mutual information
    mutual_info_check(data, label)
