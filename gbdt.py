# -*- coding:utf-8 -*- 
from sklearn import svm                                         #����SVMģ��
from sklearn import ensemble       #����emsembleģ�飬GBDT���㷨λ�ڴ�ģ����
import numpy as np                                            #����numpyģ��
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
path = 'D:\\pwork\\bs\\JZG_VISITOR.txt'                     #�趨�����ļ�·��
data = np.loadtxt(path, dtype=float, delimiter=',',skiprows=1,usecols=(1,2,3,4,5,6,7,8,9,10,11,12))
print data
x, y = np.split(data, (11,), axis=1)                    #��������ֵ��Ŀ�����
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.8)                                           #����ѵ���������Լ�
parameters = {'n_estimators': [100,300,500,700,900], 'max_depth': [3,4,5,6], 'min_samples_split': [0.5,2,3],'learning_rate': [0.01,0.1,0.3,0.5]}                                         #�趨����
gbdt = ensemble.GradientBoostingRegressor()            
#���庯��gbdt=��������ʹ��GBDT�㷨
clf = GridSearchCV(gbdt,parameters)  
#ʹ��parameters��Ĳ�������GridSearchCV����GBDT����
clf.fit(x_train, y_train)              #ѵ��GBDT����GBDT�Ľ����ѵ���������
print 'The parameters of the best model are: '  
print  clf.best_params_                                         #�����Ѳ���
print 'train accuracy:'
print clf.score(x_train, y_train)                     #���ѵ������ϵ�׼ȷ��
y_hat = clf.predict(x_train)               #�趨y_hatΪGBDTԤ���ѵ�������
print 'train predict:'
print y_hat                                             #���Ԥ���ѵ�������
print 'train real:'
print y_train       #�����ʵ��ѵ�������������������Ԥ���ѵ��������Աȹ۲�
print 'test accuracy:'
print clf.score(x_test, y_test)                       #������Լ���Ԥ��׼ȷ��
y_hat = clf.predict(x_test)                #�趨y_hatΪGBDTԤ��Ĳ��Լ����
print 'test predict:'
print y_hat                                             #���Ԥ��Ĳ��Լ����
print 'test real:'
print y_test        #�����ʵ�Ĳ��Լ����������������Ԥ��Ĳ��Լ�����Աȹ۲�
