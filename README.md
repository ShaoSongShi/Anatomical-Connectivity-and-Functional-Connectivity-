# Anatomical-Connectivity-and-Functional-Connectivity-
通过BIC评估不同维度的ERM模型，研究细胞类型聚类在功能空间的分布

## 几种结构连接矩阵与功能连接矩阵的简介
### 耦合协方差矩阵
#### 耦合协方差矩阵的定义
 $$Cov_{ij}= (JJ^T)_{ij}$$ 其中 $$Cov_{ij}$$ 为协方差矩阵ij位置的元素，J为结构连接矩阵。
#### 耦合协方差矩阵的生物学意义
相较于耦合相关矩阵，协方差矩阵保留了神经元之间在入度和出度上的异质性。考虑到在中等密度区域（神经数据所处的区域），神经元活动水平的异质性（由 E(σ^4) 刻画）是增强尺度不变性的关键因素之一，下面针对不同数据集，进行了协方差矩阵的特征谱图的比较，研究其尺度不变性。
#### 耦合协方差矩阵的特征值谱
斑马鱼全脑协方差谱在谱体部（bulk）具有幂律衰减性质，耦合协方差谱可能也具有类似的性质。
我们在loglog坐标下，拟合特征值谱的体部，得到特征值幂律衰减的相关性质。
在拟合时，不使用特征值谱头部的原因如下：
- 低秩结构（全局同步模态、任务相关信号、主导运动方向）会在谱的主体之外产生少数离群特征值，而特征谱几乎不受影响。（The spectrum of covariance matrices of randomly connected recurrent neuronal networks with linear dynamics. PLoS Computational Biology, 2022, 18(7): e1010327 ）
- 理论幂律表达式本身就是高密度极限下对大特征值谱体的渐近结果，对最顶端几个点本来就不一定适用。
- 头部几个点较为稀疏，统计上波动大。
不使用快速衰减的特征值尾部的原因如下：
- 总方差有限，幂律不可能延伸到任意小的特征值，特征值谱在 r/N→1  处必然出现截断和加速衰减，这段衰减不符合幂律。
- 尾部受实验噪声等噪声影响大。
耦合相关矩阵input alignment matrix (or structural alignment matrix)
耦合相关矩阵的定义
 $$C_{ij}= \frac{(JJ^T)_{ij}}{\sqrt{(JJ^T)_{ii}} \sqrt{(JJ^T)_{jj}}}$$ 其中 $$C_{ij}$$ 为耦合相关矩阵ij位置的元素，J为结构连接矩阵。C是J的余弦相似度矩阵。
本文采用余弦相似度而非 Pearson 相关构建耦合相关矩阵。一方面，电镜连接组数据为无符号计数，Pearson 中心化产生的负元素缺乏对应的连接学含义；另一方面，ERM 框架假设相关核为正值慢衰减核 $$f(x)=\varepsilon^\mu(\varepsilon^2+\|x\|^2)^{-\mu/2}\in(0,1]$$，元素分布拟合定义在 $$R>0$$ 上，余弦相似度的值域 $$[0,1]$$ 与之匹配。
但是A Structural Principle for Macroscopic Neural Dynamics Correlations文章中在做实验验证时，使用的是Pearson相关。

## 数据集简介
本研究使用的果蝇连接组数据来自 Janelia FlyEM Hemibrain 成年果蝇半脑电镜连接组数据集，并基于Turner等人公开的代码（https://github.com/mhturner/SC-FC）以及Whole-brain annotation and multi-connectome cell typing of Drosophila的标准方法实现部分结构连接矩阵的计算。
- hemibrain 完整连接组实际包含约 50,000 个神经元（约 25,000 个已分型，其余为未分型神经元或碎片）。本研究所使用的 traced-neurons.csv 只包含已经被手动追踪（proofread）或完整重建的神经元子集。
- Supplemental_file1_neuron_annotations.tsv文件包含cell_type、morphology_group、cell_sub_class、cell_class以及super_class等多个不同层级的分类，还包含hemibrain_type标签，可以实现从hemibrain_type到cell_type等的映射。
- Supplemental_file5_hemibrain_meta.csv包含hemibrain连接组的bodyId及其他信息

## 评估维度的pipeline
### pipeline
1. 准备距离矩阵
输入相关性矩阵C（这边使用对结构连接矩阵计算pearson相关得到的耦合相关矩阵C），通过拟合元素概率密度分布，得到不同维度假设下的ERM参数，并计算得到距离矩阵D（D = find_D(C, p)）。
2. 使用不同的MDS方法（Classic / Huber / IRLS Sammon MDS），获得嵌入不同维度的 位置坐标X
3. 对三种MDS方法和每个嵌入维度，进行欧几里得的距离矩阵重构（得到D2 = pairwise(Euclidean(), X, dims=1)），计算C2后进行BIC评估，BIC的残差计算应当依据MDS降维方法的不同而改变（参考之前的pdf文件），再使用NCC、MSE等参数比较重构前后相关性矩阵C和C2（C2=Corr(D2,p)）的相似性。
4. 绘制重构前后的耦合相关矩阵的散点图，可视化重构效果。通过绘制神经元在二维或三维功能空间内的分布，并按细胞类型层级染色，研究细胞类型聚类情况。

### 评估维度的评价指标简介
1. BIC贝叶斯信息准则
BIC的数学形式为 $$BIC = -\ln L(\theta) + k\ln n$$ ，其中惩罚项 $$k\ln n$$ 来自 Laplace 近似中 Fisher 信息矩阵的行列式项。
由于在理想情况下重构前后相关性矩阵的残差为高斯噪声， $$r_ij = C_{ij} - C2_{ij}服从N(0, sigma^2)$$ ，其中C为原始的相关性矩阵（本研究使用耦合相关矩阵），C2为重构的相关性矩阵，轮廓化  $$sigma^2$$并丢弃常数项后
 $$BIC(d) = n * ln(RSS/n) + k_d * ln(n)$$ ,
 $$RSS = \sum_{i<j} r_{ij}^2, n = \frac{N(N-1)}{2}$$ ,
 $$k_d = N*d - d(d+1)/2 + 1$$ , k_d 扣除 d 维欧氏嵌入的旋转/平移/反射不可识别自由度d(d+1)/2，+1 为噪声尺度参数。
由于三种 MDS 方法假设的噪声分布一致，共用同一似然族与 k/n 口径，因此BIC 曲线可横向比较。
2. NCC归一化互相关
用于比较重构前后的耦合相关矩阵的相似程度。
x = a - a.mean()          # a = Corr[iu]，原始矩阵上三角
y = b - b.mean()          # b = C2[iu]，重构矩阵上三角
ncc = sum(x*y) / sqrt(sum(x^2) * sum(y^2))
3. MSE均方误差
用于比较重构前后的耦合相关矩阵在数值上的接近程度。计算方法为，将重构前后矩阵相减，对每个元素求平方，再对每个元素求平均，得到重构前后矩阵的均方误差。
a = Corr[iu] # 原始矩阵上三角
b = C2[iu] # 重构矩阵上三角
mse = float(np.mean((a - b) ** 2))
