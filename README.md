# CS7304H_HW

## To-Do List

### 数据预处理

- 确定数据样本点的分布是稀疏的还是稠密的.
  - 特征统计: 根据统计的结果来看, 几乎99%的数据都趋近于0, 因此该特征数据是稀疏的.
- 确定数据样本点是线性的还是非线性的.
  - 通过将数据进行降维处理, 发现该数据具有一定的非线性特征
  - linear PCA->100->linear PCA->2
  - kernel PCA->100->linear PCA->2
  - linear PCA->100->UMAP->2
  - kernel PCA->100->UMAP->2
- 确定是否要数据进行标准化处理
  - 中心标准化vs非中心标准化
  - 根据PCA和UMAP降维得到的图像可以看出, 标准化会丢失稀疏数据的信息, 让数据更难分类, 因此不采用稀疏化.
  - 同时用SVM分类的结果也可以看到中心化的质量更差.
- 数据分析结论
  - 具有**稀疏性**
  - 具有**非线性特征**
  - 不建议使用**标准化方法**

### 数据降维

- 采用不同降维方式进行比较.
  - linear: PCA
  - non-linear: KPCA, Auto Encoder
  - 确定kPCA的rbf核的gamma大小: 根据结果显示gamma=0.0001的质量更好
  - 比较PCA和kPCA的结果: 根据数据结果显示, kPCA的质量微微好于PCA的质量(gamma=0.0001)
  - 比较PCA和Auto Encoder的结果: 根据数据显示, auto encoder可以进行降维, 但是降维的结果不适合线性分类器进行分类, 至少在使用SVM上的质量要差很多. **(?)**

- 采用降维到不同维数来进行比较.
  - 10000(origin)
  - 5000
  - 1000
  - 500
  - 100
  - 使用kPCA/PCA+kSVM得到的结果显示: 随着数据维度的下降分类效果也逐渐下降.
- 数据降维结论:
  - 在训练集上, 降维会导致**质量下降**, 在5000~1000区间内, 质量下降的较少, 但是1000以下会出现快速下降
  - 在测试集上, 降维到**5000**时质量最佳
  - 采用**RBF kPCA**(gamma==0.0001/default, 5000)

### 分类模型

- 使用不同分类模型进行实验.
  - MLP
    - baseline: 过拟合
    - 对不同降维维度的样本进行分类:
      - 10000
      - 5000
      - 1000
    - 使用不同层数的神经网络进行实验: 根据数据, 隐藏层层数越少抗拟合性能越好, 因此采用单层隐藏层即可.
    - 使用不同宽度的神经网络进行实验: 根据数据, 基本表达能力类似(1024左右较佳)
    - 使用dropout: dropout能够一定程度上避免过拟合的发生(比较诡异, 差异不明显)
    - 使用self-supervised learning:
      - Naive实现: 质量没有显著提升
      - MixMatch: **(?)**
  - SVM
    - 使用不同的核进行分类: RBF核的效果最好.
  - Logistic Regression
    - 比较简单, 完全自己实现. One vs Rest LR多分类器
    - 不同的梯度下降法:
      - SGD method:
      - Newton method:
    - 不同学习率策略:
      - warmup+cosine decay
      - CLR

### 数据可视化

- 下面这些方法的相对关系保持的比较好, 但是需要先进行PCA降维再使用, 不然时间复杂度太高.
- t-SNE
- UMAP: 使用PCA/kPCA/AE降至100后, 在使用UMAP进行数据可视化.
- Local MDS
