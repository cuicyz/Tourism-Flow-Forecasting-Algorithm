# -*- coding:utf-8 -*- 
from sklearn import svm                                         #引入SVM模块
from sklearn import ensemble       #引入emsemble模块，GBDT的算法位于此模块下
import numpy as np                                            #引入numpy模块
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
path = 'D:\\pwork\\bs\\JZG_VISITOR.txt'                     #设定数据文件路径
data = np.loadtxt(path, dtype=float, delimiter=',',skiprows=1,usecols=(1,2,3,4,5,6,7,8,9,10,11,12))
print data
x, y = np.split(data, (11,), axis=1)                    #划分特征值与目标变量
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.8)                                           #划分训练集、测试集
parameters = {'n_estimators': [100,300,500,700,900], 'max_depth': [3,4,5,6], 'min_samples_split': [0.5,2,3],'learning_rate': [0.01,0.1,0.3,0.5]}                                         #设定参数
gbdt = ensemble.GradientBoostingRegressor()            
#定义函数gbdt=对括号内使用GBDT算法
clf = GridSearchCV(gbdt,parameters)  
#使用parameters里的参数，用GridSearchCV进行GBDT调参
clf.fit(x_train, y_train)              #训练GBDT，将GBDT的结果向训练集上拟合
print 'The parameters of the best model are: '  
print  clf.best_params_                                         #输出最佳参数
print 'train accuracy:'
print clf.score(x_train, y_train)                     #输出训练集拟合的准确率
y_hat = clf.predict(x_train)               #设定y_hat为GBDT预测的训练集结果
print 'train predict:'
print y_hat                                             #输出预测的训练集结果
print 'train real:'
print y_train       #输出真实的训练集结果，方便我们与预测的训练集结果对比观察
print 'test accuracy:'
print clf.score(x_test, y_test)                       #输出测试集的预测准确率
y_hat = clf.predict(x_test)                #设定y_hat为GBDT预测的测试集结果
print 'test predict:'
print y_hat                                             #输出预测的测试集结果
print 'test real:'
print y_test        #输出真实的测试集结果，方便我们与预测的测试集结果对比观察
