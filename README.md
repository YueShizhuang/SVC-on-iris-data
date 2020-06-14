机器学习——SVM Python实战
一、数据集准备
本文用的数据集为Iris.data可从UCI数据库中下载，http://archive.ics.uci.edu/ml/datasets/Iris
Iris.data的数据格式如下：共5列，前4列为样本特征，第5列为类别，分别有三种类别Iris-setosa/ Iris-versicolor/Iris-virginica。
NOTE1：第五列的字符串数据最好转化成数字量。
 
【函数说明】
@set_module('numpy')
def loadtxt(fname, dtype=float, comments='#', delimiter=None,
            converters=None, skiprows=0, usecols=None, unpack=False,
            ndmin=0, encoding='bytes', max_rows=None):
函数功能：从text文档中加载数据（数据的每一行必须维度相同）
参数解释：
delimiter : str, optional（设置分隔号）
    The string used to separate values. For backwards compatibility, byte strings will be decoded as 'latin1'. The default is whitespace.
converters : dict, optional转换器（字典型）通过字典将列值映射至一个函数。此函数将会列值解析为期望值。此处使用此函数对数据集中的字符值转化成数值。
    A dictionary mapping column number to a function that will parse the column string into the desired value.  E.g., if column 0 is a date string: ``converters = {0: datestr2num}``.  Converters can also be used to provide a default value for missing data (but see also `genfromtxt`): ``converters = {3: lambda s: float(s.strip() or 0)}`` Default: None.

二、导入SVM模块并划分数据与标签
Python中的sklearn库集成了SVM算法。
【函数说明】
train_test_split(train_data,train_label,test_size=数字, random_state=0)
函数功能：随机划分训练集与测试集。
参数解释：
train_data：所要划分的样本特征集
train_label：所要划分的样本类别
test_size：样本占比，如果是整数的话就是样本的数量。
test_size:测试样本占比。默认情况下，该值设置为0.25。只有train_size没有指定时，它将保持0.25，否则它将补充指定的train_size，例如train_size=0.6,则test_size默认为0.4。
train_size:训练样本占比。
random_state：是随机数的种子。随机数种子：其实就是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。比如你每次都填1，其他参数一样的情况下你得到的随机数组是一样的。但填0或不填，每次都会不一样。随机数的产生取决于种子，随机数和种子之间的关系遵从以下两个规则：种子不同，产生不同的随机数；种子相同，即使实例不同也产生相同的随机数。
三、训练分类器
【函数说明】
classifier=svm.SVC(C=2,kernel='rbf',gamma=10,decision_function_shape='ovr')
sytax:
def __init__(self, *, C=1.0, kernel='rbf', degree=3, gamma='scale',
             coef0=0.0, shrinking=True, probability=False,
             tol=1e-3, cache_size=200, class_weight=None,
             verbose=False, max_iter=-1, decision_function_shape='ovr',
             break_ties=False,
             random_state=None):
参数解释：
C : float, default=1.0（正则化参数，正则化强度与C成反比）
    Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive. The penalty is a squared l2 penalty.
kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}, default='rbf'（指定算法中的核函数类型，可以选择线性核函数，多项式核函数，高斯核函数， 核函数等。如果没有指定，则默认高斯核函数，对应的支持向量机是高斯径向基函数（radial basis function）分类器。）
    Specifies the kernel type to be used in the algorithm.  It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or a callable.
degree : int, default=3（设定多项式金额函数的阶数）
    Degree of the polynomial kernel function ('poly').
gamma : {'scale', 'auto'} or float, default='scale'（核函数的系数。gamma值越小，分类界面越连续；gamma值越大，分类界面越“散”，分类效果越好，但有可能会过拟合。）
    Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
    - if ``gamma='scale'`` (default) is passed then it uses 1 / (n_features * X.var()) as value of gamma.- if 'auto', uses 1 / n_features.
coef0 : float, default=0.0
    Independent term in kernel function.It is only significant in 'poly' and 'sigmoid'.
shrinking : bool, default=True  Whether to use the shrinking heuristic.
（优化算法检测到，在高概率（不是严格意义上的数学意义上的）情况下，可以通过在选项中旋转-h 0标志来加快训练速度。基本上，-h是收缩式启发式算法，在libsvm包中实现，对于某些数据，它显著减少了所需的计算次数，而在其他一些数据中，则会使计算速度变慢。）
probability : bool, default=False
是否启用概率估计。这必须在调用“fit”之前启用，这将减慢该方法的速度，因为它在内部使用5倍交叉验证，“predict-proba”可能与“predict”不一致。
class_weight : dict or 'balanced', default=None
    Set the parameter C of class i to class_weight[i]*C for SVC. If not given, all classes are supposed to have weight one.  The "balanced" mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as ``n_samples / (n_classes * np.bincount(y))``
decision_function_shape : {'ovo', 'ovr'}, default='ovr'（使用一对一策略还是一对多策略。其中一对一策略常用在在多分类策略上。）
    Whether to return a one-vs-rest ('ovr') decision function of shape  (n_samples, n_classes) as all other classifiers, or the original one-vs-one ('ovo') decision function of libsvm which has shape (n_samples, n_classes * (n_classes - 1) / 2). However, one-vs-one ('ovo') is always used as multi-class strategy. The parameter is ignored for binary classification.




