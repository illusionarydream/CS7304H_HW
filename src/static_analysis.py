import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

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
    plt.show()


if __name__ == "__main__":
    data = pickle.load(open("datasets/normalized_train_feature.pkl", "rb"))

    # 示例调用
    feature_distributed(data)
