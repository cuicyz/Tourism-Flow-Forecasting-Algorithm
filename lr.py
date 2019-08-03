# -*- coding:utf-8 -*-
from sklearn import svm                  #����sklearn��numpy��matplotlibģ��
import numpy as np
from sklearn.model_selection import train_test_split
#����ѵ���������Լ����ֹ���
from sklearn.model_selection import GridSearchCV   #����GridSearchCV���ι���
from sklearn.svm import SVR
from sklearn import ensemble
from sklearn import linear_model            #��sklearnģ������������ģ�Ͳ���
from sklearn.linear_model import LinearRegression           #�������Իع�ģ��
path = 'D:\\pwork\\bs\\JZG_VISITOR.txt'                     #���������ļ�·��
data = np.loadtxt(path, dtype=float, delimiter=',',skiprows=1,usecols=(1,2,3,4,5,6,7,8,9,10,11,12))
print data
x, y = np.split(data, (11,), axis=1)                    #��������ֵ��Ŀ�����
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.8)                                           #����ѵ���������Լ�
clf = svm.SVR(kernel='rbf',C=110000,gamma=0.0075)  #�趨clfΪSVR�Ľ����������ֱ��ʹ����֮ǰ����SVR��ʱ��õ��Ĳ�����������Ŀǰ���ֵ�׼ȷ����ߵĲ���
clf.fit(x_train,y_train)                 #ѵ��SVR����SVR�Ľ����ѵ���������
print 'svr train accuracy:'
print clf.score(x_train, y_train)                 #���SVRѵ������ϵ�׼ȷ��
y_hat = clf.predict(x_train)                #�趨y_hatΪSVRԤ���ѵ�������
y_hat = y_hat.reshape(-1,1)       #��y_hat��ת�ã������y_hat���н�һ������
print 'svr test accuracy:'
print clf.score(x_test, y_test)                   #���SVR���Լ���ϵ�׼ȷ��
y_hat1 = clf.predict(x_test)               #�趨y_hat1ΪSVRԤ��Ĳ��Լ����
y_hat1 = y_hat1.reshape(-1,1)   #��y_hat1��ת�ã������y_hat1���н�һ������
clf = ensemble.GradientBoostingRegressor(n_estimators=500,max_depth=5,
min_samples_split=3,learning_rate=0.1)  
#�趨clf1ΪGBDT�Ľ������ͬ��ֱ��ʹ����֮ǰ����GBDT��ʱ��õ��Ĳ���
clf1.fit(x_train, y_train)             #ѵ��GBDT����GBDT�Ľ����ѵ���������
print 'gbdt train accuracy:'
print clf1.score(x_train, y_train)               #���GBDTѵ������ϵ�׼ȷ��
y_hat2 = clf1.predict(x_train)            #�趨y_hat2ΪGBDTԤ���ѵ�������
y_hat2 = y_hat2.reshape(-1,1)   #��y_hat2��ת�ã������y_hat2���н�һ������
print 'gbdt test accuracy:'
print clf1.score(x_test, y_test)                 #���GBDT���Լ���ϵ�׼ȷ��
y_hat3 = clf1.predict(x_test)             #�趨y_hat3ΪGBDT���Ե�ѵ�������
y_hat3 = y_hat3.reshape(-1,1)   #��y_hat3��ת�ã������y_hat3���н�һ������
hz = np.vstack((y_hat,y_hat1))
#����y_hat��y_hat1���������γ�SVR��Ԥ���ܽ��
hz1 = np.vstack((y_hat2,y_hat3))
 #����y_hat2��y_hat3���������γ�GBDT��Ԥ���ܽ��
x = np.hstack((hz,hz1))
#����hz��hz1���γ��ܵ���������������Ϊx��������LRԤ��
y = np.vstack((y_train,y_test)) 
#����ѵ������Ӧ�ķ�ʽ��������ֲ���ѵ�����Ͳ��Լ���Ϊy���ϳ��ܵ�Ŀ�����
x_train,x_test = np.split(x,(2358,),axis=0)
#��������������ѵ�����Ͳ��Լ��������ѡ�������ʮ������������Լ�����ǰ������������������ѵ�������۲�С���Լ�������µ�׼ȷ�����
y_train,y_test = np.split(y,(2358,),axis=0)     #����Ŀ�������ѵ�����Ͳ��Լ� 
clf2 = LinearRegression()
#���Ѿ�����õ�x��y������������Ŀ��Ŀ����������Իع�
clf2.fit(x_train,y_train)                  #ѵ��LR����LR�Ľ����ѵ���������
print 'lr train accuracy:'
print clf2.score(x_train, y_train)                 #���LRѵ������ϵ�׼ȷ��
y_hat4 = clf2.predict(x_train)              #�趨y_hat4ΪLRԤ���ѵ�������
y_hat4 = y_hat4.reshape(-1,1)   #��y_hat4��ת�ã������y_hat4���н�һ���۲�
print 'lr test accuracy:'
print clf2.score(x_test, y_test)                   #���LR���Լ���ϵ�׼ȷ��
y_hat5 = clf2.predict(x_test)               #�趨y_hat5ΪLRԤ��Ĳ��Լ����
y_hat5 = y_hat5.reshape(-1,1)   #��y_hat5��ת�ã������y_hat5���н�һ���۲�
print 'x_test:'
print x_test                                      #���SVR��GBDT�Ĳ��Լ����
print 'y_test:'
print y_test                                            #�����ʵ�Ĳ��Լ����
print 'y_predict:'
print y_hat5        #�����LR��SVR��GBDT������ѧϰ��Ĳ��Լ���������жԱ�
