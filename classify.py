import numpy as np
import os
import csv
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import chi2
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split
import time
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import roc_curve

def draw_auc(y, y_pred):
    fpr, tpr, thresholds = roc_curve(y, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % roc_auc)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()

def cross_validation(lenOfBlock , X , Y , num):
    warnings.filterwarnings('ignore')
    if num ==2:
        clf = MLPClassifier(max_iter=500, alpha=1.0, random_state=21,tol=0.000000001)
    elif num == 1:
        clf = KNeighborsClassifier(n_neighbors=3)
    cv = KFold(n_splits = lenOfBlock, random_state = None, shuffle = False)

    TN = 0
    FP = 0
    FN = 0
    TP = 0
    F1 = 0
    Educate = 0.0
    Test = 0.0
    for train_index, test_index in cv.split(X):
        start_time = time.time()
        clf.fit(X[train_index], Y[train_index])
        education_time = time.time() - start_time
        start_time = time.time()
        proba = clf.predict(X[test_index])
        test_time = time.time() - start_time

        Educate += education_time
        Test += test_time
        tn, fp, fn, tp = confusion_matrix(Y[test_index], proba ,labels=[0,1]).ravel()
        TN += tn
        FP += fp
        FN += fn
        TP += tp
        F1 += (f1_score(Y[test_index], proba , average='binary') )
    
    summ = TN + FP + FN + TP
    
    print('TP: ', TP / summ)
    print('TN: ', TN / summ)
    print('FP: ', FP / summ)
    print('FN: ', FN / summ)
    print('Точность (Precision): ', TN / (TN + FN)) 
    print('Полнота(Recall)', TN / (TN + TP))
    print('F-мера: ', F1 / lenOfBlock)
    print('Время обучения: ', Educate) 
    print('Время тестирования: ', Test) 


def element_check(X , Y, num):
    if num ==2:
        clf = MLPClassifier(max_iter=500, alpha=1.0, random_state=21,tol=0.000000001)
    elif num == 1:
        clf = KNeighborsClassifier(n_neighbors=3)
    TN = 0
    FP = 0
    FN = 0
    TP = 0
    F1 = 0
    Educate = 0.0
    Test = 0.0
    count = 0
    loo = LeaveOneOut()
    loo.get_n_splits(X)
    
    for train_index, test_index in loo.split(X):
        start_time = time.time()
        clf.fit(X[train_index], Y[train_index])
        education_time = time.time() - start_time
        start_time = time.time()
        proba = clf.predict(X[test_index])
        test_time = time.time() - start_time
        Educate += education_time
        Test += test_time
    	
        tn, fp, fn, tp = confusion_matrix(Y[test_index], proba ,labels=[0,1]).ravel()
        TN += tn
        FP += fp
        FN += fn
        TP += tp
        count += 1
        F1 += (f1_score(Y[test_index], proba , average='binary') )
    
    summ = TP + TN + FP + FN

    print('TP: ', TP / summ)
    print('TN: ', TN / summ)
    print('FP: ', FP / summ)
    print('FN: ', FN / summ)
    print('Точность (Precision): ', TN / (TN + FN)) 
    print('Полнота(Recall)', TN / (TN + TP))
    print('F-мера: ', F1 / len(Y))
    print('Время обучения: ', Educate) 
    print('Время тестирования: ', Test) 


 
def random_sampling(X , Y, num):
    tn = 0
    fp = 0
    fn = 0
    fp = 0
    education_time = 0.0
    test_time = 0.0
    if num == 2:
        clf = MLPClassifier(max_iter=500, alpha=1.0, random_state=21,tol=0.000000001)
    elif num ==1:
        clf = KNeighborsClassifier(n_neighbors=3)
    X_train, X_test, y_train, y_test = train_test_split(X, Y)
    
    start_time = time.time()
    clf.fit(X_train, y_train)
    education_time = time.time() - start_time
    
    start_time = time.time()
    proba = clf.predict(X_test)
    test_time = time.time() - start_time
    draw_auc(y_test, proba)
    tn, fp, fn, tp = confusion_matrix(y_test, proba,labels=[0,1]).ravel()

    summ = tn + fp + fn + tp

    print('TP: ', tp / summ)
    print('TN: ', tn / summ)
    print('FP: ', fp / summ)
    print('FN: ', fn / summ)
    print('Точность (Precision): ', tn / (tn + fn)) 
    print('Полнота(Recall)', tn / (tn + tp))
    print('F-мера: ', f1_score(y_test, proba, average='binary'))
    print('Время обучения: ', education_time) 
    print('Время тестирования: ', test_time) 


#метод взаимной информации
def mutual_info(data, labels):
    selecter = SelectKBest(score_func=mutual_info_classif, k=20)
    selecter.fit(data, labels)
    string = selecter.get_support()
    return selecter.transform(data),string

#критерий хи2
def method_chi2(data, labels):
    selecter = SelectKBest(score_func=chi2, k=20)
    selecter.fit(data, labels)
    string = selecter.get_support()
    return selecter.transform(data),string

def read_data():
    data = []
    names=[]
    labels=[]
    flag=0
    with open('D:\\parse.csv') as file:
        for line in file:
            if flag ==0:
                names=line.split(';')
                flag=1
            else:
                parameters = line.split(';')
                for i in range(len(parameters)):
                    parameters[i] = int(parameters[i])
                data.append(parameters)
    with open('D:\\flag.csv') as file:
        for line in file:
            labels.append(int(line[0]))
        
    return np.array(data),names,labels

def main():
    dataset,names,labels = read_data()
    print("1.Критерий x^2 и метод k-ближайших соседей\n2.Метод взаимной информации и нейронная сеть")
    num=int(input())
    if num == 1:
        dataset,string = method_chi2(dataset, labels)
        for i in range(0,len(string)):
            if string[i]==True:
                print(names[i])
        
    elif num == 2:
        dataset,string = mutual_info(dataset, labels)
        for i in range(0,len(string)):
            if string[i]==True:
                print(names[i])

    warnings.filterwarnings('ignore')
    buffer_test = minmax_scale(dataset,feature_range=(0, 1),axis = 0)
    nptraining = np.array(buffer_test , "float32")
    nptarget = np.array(labels, "float32")

    print("Перекрестная проверка k = 5:")
    cross_validation(5 , nptraining , nptarget,num)
    print("Перекрестная проверка k = 10:")
    cross_validation(10 , nptraining , nptarget,num)
    print("Перекрестная проверка k = 20:")
    cross_validation(20 , nptraining , nptarget,num)

    print("Случайное сэмплирование:")
    random_sampling(nptraining , nptarget,num)

    print("Поэлементная проверка:")
    element_check(nptraining , nptarget, 

)


main()
