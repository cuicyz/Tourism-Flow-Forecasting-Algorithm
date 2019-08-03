# -*- coding:utf-8 -*-
from sklearn import svm  
#从sklearn机器学习模块中引入SVM算法，这样接下来可以直接调用SVM
import numpy as np                          #引入numpy矩阵处理模块，缩写为np
from sklearn.model_selection import train_test_split
#从sklearn中引入训练集、测试集分类工具train_test_split
from sklearn.model_selection import GridSearchCV        
#从sklearn中引入调参工具GridSearchCV
from sklearn.svm import SVR        
#从sklearn.svm中引入负责做回归分析的SVR（支持向量回归）
path = 'D:\\pwork\\bs\\JZG_VISITOR.txt'                     #定义数据文件路径
data = np.loadtxt(path, dtype=float, delimiter=',',skiprows=1,usecols=(1,2,3,4,5,6,7,8,9,10,11,12))
#导入txt格式数据文件，参数为：（文件路径，数据类型=浮点型，跳过第一行（名称行），选择第1-12列（跳过第0列序号列））
print data                                  #展示读取的数据，观察是否符合要求
x, y = np.split(data, (11,), axis=1)
#分割特征值及目标变量，0-10列为x（特征值），11列为y（目标变量）
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.8)
#分割训练集和测试集，采用随机分配，训练集=80%，测试集=20%，random_state是指数据分割的随机情况，random_state=1的话，每次的训练集和测试集划分出来的结果是不变的，random_state=0的话，每次划分出来的训练集和测试集都不一样
parameters={'kernel':('linear','rbf'),'C':[80000,100000,120000],'gamma':[0.0075,0.00755,0.0076,0.00765,0.0077]}#设定SVM参数，核函数：线性核或高斯核，惩罚系数C：80000、100000或120000，核函数系数gamma：0.0075、0.00755、0.0076、0.00765、0.0077。系数的调整的范围并不大，这是因为我从大范围的尝试中寻找了预测准确率相对高的系数，对系数已经做过初步筛选
svr = svm.SVR()                             #定义函数svr=对括号内使用SVR算法
clf = GridSearchCV(svr, parameters)     
#使用parameters里的参数，用GridSearchCV进行SVR调参
clf.fit(x_train,y_train)                 #训练SVR，将SVR的结果向训练集上拟合
print 'The parameters of the best model are: '  
print  clf.best_params_                                   #输出SVR的最佳参数
print 'train accuracy:'
print clf.score(x_train, y_train)                     #输出训练集拟合的准确率
y_hat = clf.predict(x_train)                #设定y_hat为SVR预测的训练集结果
print 'train predict:'
print y_hat                                             #输出预测的训练集结果
print 'train real:'
print y_train       #输出真实的训练集结果，方便我们与预测的训练集结果对比观察
print 'test accuracy:'
print clf.score(x_test, y_test)                       #输出测试集的预测准确率
y_hat = clf.predict(x_test)                 #设定y_hat为SVR预测的测试集结果
print 'test predict:'
print y_hat                                             #输出预测的测试集结果
print 'test real:'
print y_test        #输出真实的测试集结果，方便我们与预测的测试集结果对比观察
