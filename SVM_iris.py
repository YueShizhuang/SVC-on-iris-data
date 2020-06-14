import numpy as np
import sklearn
from sklearn import svm
#define converts(字典)
def Iris_label(s):
    it={b'Iris-setosa':0, b'Iris-versicolor':1, b'Iris-virginica':2 }
    return it[s]
#1.读取数据集
path='E:/pyproject/Iris.data'
data=np.loadtxt(path, dtype=float, delimiter=',', converters={4:Iris_label} )
#converters={4:Iris_label}中“4”指的是第5列：将第5列的str转化为label(number)
#2.划分数据与标签
x,y=np.split(data,indices_or_sections=(4,),axis=1) #x为数据，y为标签
x=x[:,0:1 ] #为便于后边画图显示，只选取前两维度。若不用画图，可选取前四列x[:,0:4]
train_data,test_data,train_label,test_label =sklearn.model_selection.train_test_split(x,y, random_state=1, train_size=0.6,test_size=0.4)
#3.训练svm分类器
classifier=svm.SVC(C=2,kernel='rbf',gamma=10,decision_function_shape='ovr') # ovr:一对多策略
classifier.fit(train_data,train_label.ravel()) #ravel函数在降维时默认是行序优先
#4.计算svc分类器的准确率
print("训练集：",classifier.score(train_data,train_label))
print("测试集：",classifier.score(test_data,test_label))


