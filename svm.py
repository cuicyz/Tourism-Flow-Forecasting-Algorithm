# -*- coding:utf-8 -*-
from sklearn import svm  
#��sklearn����ѧϰģ��������SVM�㷨����������������ֱ�ӵ���SVM
import numpy as np                          #����numpy������ģ�飬��дΪnp
from sklearn.model_selection import train_test_split
#��sklearn������ѵ���������Լ����๤��train_test_split
from sklearn.model_selection import GridSearchCV        
#��sklearn��������ι���GridSearchCV
from sklearn.svm import SVR        
#��sklearn.svm�����븺�����ع������SVR��֧�������ع飩
path = 'D:\\pwork\\bs\\JZG_VISITOR.txt'                     #���������ļ�·��
data = np.loadtxt(path, dtype=float, delimiter=',',skiprows=1,usecols=(1,2,3,4,5,6,7,8,9,10,11,12))
#����txt��ʽ�����ļ�������Ϊ�����ļ�·������������=�����ͣ�������һ�У������У���ѡ���1-12�У�������0������У���
print data                                  #չʾ��ȡ�����ݣ��۲��Ƿ����Ҫ��
x, y = np.split(data, (11,), axis=1)
#�ָ�����ֵ��Ŀ�������0-10��Ϊx������ֵ����11��Ϊy��Ŀ�������
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.8)
#�ָ�ѵ�����Ͳ��Լ�������������䣬ѵ����=80%�����Լ�=20%��random_state��ָ���ݷָ����������random_state=1�Ļ���ÿ�ε�ѵ�����Ͳ��Լ����ֳ����Ľ���ǲ���ģ�random_state=0�Ļ���ÿ�λ��ֳ�����ѵ�����Ͳ��Լ�����һ��
parameters={'kernel':('linear','rbf'),'C':[80000,100000,120000],'gamma':[0.0075,0.00755,0.0076,0.00765,0.0077]}#�趨SVM�������˺��������Ժ˻��˹�ˣ��ͷ�ϵ��C��80000��100000��120000���˺���ϵ��gamma��0.0075��0.00755��0.0076��0.00765��0.0077��ϵ���ĵ����ķ�Χ������������Ϊ�ҴӴ�Χ�ĳ�����Ѱ����Ԥ��׼ȷ����Ըߵ�ϵ������ϵ���Ѿ���������ɸѡ
svr = svm.SVR()                             #���庯��svr=��������ʹ��SVR�㷨
clf = GridSearchCV(svr, parameters)     
#ʹ��parameters��Ĳ�������GridSearchCV����SVR����
clf.fit(x_train,y_train)                 #ѵ��SVR����SVR�Ľ����ѵ���������
print 'The parameters of the best model are: '  
print  clf.best_params_                                   #���SVR����Ѳ���
print 'train accuracy:'
print clf.score(x_train, y_train)                     #���ѵ������ϵ�׼ȷ��
y_hat = clf.predict(x_train)                #�趨y_hatΪSVRԤ���ѵ�������
print 'train predict:'
print y_hat                                             #���Ԥ���ѵ�������
print 'train real:'
print y_train       #�����ʵ��ѵ�������������������Ԥ���ѵ��������Աȹ۲�
print 'test accuracy:'
print clf.score(x_test, y_test)                       #������Լ���Ԥ��׼ȷ��
y_hat = clf.predict(x_test)                 #�趨y_hatΪSVRԤ��Ĳ��Լ����
print 'test predict:'
print y_hat                                             #���Ԥ��Ĳ��Լ����
print 'test real:'
print y_test        #�����ʵ�Ĳ��Լ����������������Ԥ��Ĳ��Լ�����Աȹ۲�
