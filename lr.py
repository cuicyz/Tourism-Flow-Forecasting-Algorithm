# -*- coding:utf-8 -*-
from sklearn import svm                  #引入sklearn、numpy、matplotlib模块
import numpy as np
from sklearn.model_selection import train_test_split
#引入训练集、测试集划分工具
from sklearn.model_selection import GridSearchCV   #引入GridSearchCV调参工具
from sklearn.svm import SVR
from sklearn import ensemble
from sklearn import linear_model            #从sklearn模块中引入线性模型部分
from sklearn.linear_model import LinearRegression           #引入线性回归模块
path = 'D:\\pwork\\bs\\JZG_VISITOR.txt'                     #定义数据文件路径
data = np.loadtxt(path, dtype=float, delimiter=',',skiprows=1,usecols=(1,2,3,4,5,6,7,8,9,10,11,12))
print data
x, y = np.split(data, (11,), axis=1)                    #划分特征值与目标变量
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.8)                                           #划分训练集、测试集
clf = svm.SVR(kernel='rbf',C=110000,gamma=0.0075)  #设定clf为SVR的结果，这里我直接使用了之前在做SVR的时候得到的参数，这是我目前发现的准确率最高的参数
clf.fit(x_train,y_train)                 #训练SVR，将SVR的结果向训练集上拟合
print 'svr train accuracy:'
print clf.score(x_train, y_train)                 #输出SVR训练集拟合的准确率
y_hat = clf.predict(x_train)                #设定y_hat为SVR预测的训练集结果
y_hat = y_hat.reshape(-1,1)       #将y_hat做转置，方便对y_hat进行进一步处理
print 'svr test accuracy:'
print clf.score(x_test, y_test)                   #输出SVR测试集拟合的准确率
y_hat1 = clf.predict(x_test)               #设定y_hat1为SVR预测的测试集结果
y_hat1 = y_hat1.reshape(-1,1)   #将y_hat1做转置，方便对y_hat1进行进一步处理
clf = ensemble.GradientBoostingRegressor(n_estimators=500,max_depth=5,
min_samples_split=3,learning_rate=0.1)  
#设定clf1为GBDT的结果，我同样直接使用了之前在做GBDT的时候得到的参数
clf1.fit(x_train, y_train)             #训练GBDT，将GBDT的结果向训练集上拟合
print 'gbdt train accuracy:'
print clf1.score(x_train, y_train)               #输出GBDT训练集拟合的准确率
y_hat2 = clf1.predict(x_train)            #设定y_hat2为GBDT预测的训练集结果
y_hat2 = y_hat2.reshape(-1,1)   #将y_hat2做转置，方便对y_hat2进行进一步处理
print 'gbdt test accuracy:'
print clf1.score(x_test, y_test)                 #输出GBDT测试集拟合的准确率
y_hat3 = clf1.predict(x_test)             #设定y_hat3为GBDT测试的训练集结果
y_hat3 = y_hat3.reshape(-1,1)   #将y_hat3做转置，方便对y_hat3进行进一步处理
hz = np.vstack((y_hat,y_hat1))
#连接y_hat、y_hat1两个矩阵，形成SVR的预测总结果
hz1 = np.vstack((y_hat2,y_hat3))
 #连接y_hat2、y_hat3两个矩阵，形成GBDT的预测总结果
x = np.hstack((hz,hz1))
#连接hz、hz1，形成总的特征变量矩阵作为x，用来做LR预测
y = np.vstack((y_train,y_test)) 
#以与训练集对应的方式连接随机分布的训练集和测试集作为y，合成总的目标变量
x_train,x_test = np.split(x,(2358,),axis=0)
#划分特征变量的训练集和测试集，这次我选择用最后十天的数据做测试集，用前面所有天数的数据做训练集，观察小测试集的情况下的准确率情况
y_train,y_test = np.split(y,(2358,),axis=0)     #划分目标变量的训练集和测试集 
clf2 = LinearRegression()
#以已经定义好的x和y即特征变量和目标目标变量做线性回归
clf2.fit(x_train,y_train)                  #训练LR，将LR的结果向训练集上拟合
print 'lr train accuracy:'
print clf2.score(x_train, y_train)                 #输出LR训练集拟合的准确率
y_hat4 = clf2.predict(x_train)              #设定y_hat4为LR预测的训练集结果
y_hat4 = y_hat4.reshape(-1,1)   #将y_hat4做转置，方便对y_hat4进行进一步观察
print 'lr test accuracy:'
print clf2.score(x_test, y_test)                   #输出LR测试集拟合的准确率
y_hat5 = clf2.predict(x_test)               #设定y_hat5为LR预测的测试集结果
y_hat5 = y_hat5.reshape(-1,1)   #将y_hat5做转置，方便对y_hat5进行进一步观察
print 'x_test:'
print x_test                                      #输出SVR、GBDT的测试集结果
print 'y_test:'
print y_test                                            #输出真实的测试集结果
print 'y_predict:'
print y_hat5        #输出用LR对SVR和GBDT做二次学习后的测试集结果，进行对比
