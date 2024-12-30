import matplotlib.pyplot as plt
import numpy as np

# 定义数据
labels = ['PCA_linear_1000', 'PCA_poly_1000', 'PCA_rbf_100', 'PCA_rbf_500', 'PCA_rbf_1000', 'PCA_rbf_5000',
          'kPCA_rbf_100', 'kPCA_rbf_500', 'kPCA_rbf_1000', 'kPCA_rbf_5000',
          'kPCA_gamma_0.0001_100', 'kPCA_gamma_0.001_100', 'kPCA_gamma_0.01_100', 'kPCA_gamma_0.1_100',
          'kPCA_gamma_1_100', 'kPCA_gamma_10_100', 'AE_rbf_100', 'AE_rbf_500', 'AE_rbf_1000', 'AE_rbf_5000']
values = [0.933, 0.916, 0.884, 0.953, 0.967, 0.976, 0.881,
          0.954, 0.967, 0.976, 0.883, 0.882, 0.882, 0.881, 0.854, 0.787,
          0.080, 0.083, 0.078, 0.076]

# 创建柱状图
plt.figure(figsize=(14, 8))
bars = plt.bar(labels, values, color='skyblue', width=0.6)

# 设置x轴标记倾斜
plt.xticks(rotation=45, ha='right')

# 添加标签和标题
plt.xlabel('Methods', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.title('Bar Chart of Accuracy for Different Methods', fontsize=16)

# 添加数据标签
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005,
             f'{yval:.3f}', ha='center', va='bottom', fontsize=10)

# 美化图表
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# 显示图表
# plt.show()
plt.savefig('image/Bar_chart_of_accuracy_for_different_methods.png')
