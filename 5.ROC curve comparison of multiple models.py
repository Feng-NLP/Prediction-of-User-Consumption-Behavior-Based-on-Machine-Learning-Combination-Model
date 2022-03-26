from sklearn.metrics import roc_curve, auc
import matplotlib as mpl  
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#定义ROC绘制曲线函数
def plot_roc(labels, predict_probs, titles):
    color = ['c', 'g', 'y', 'r', 'b']                                                                 
    for idx, predict_prob in enumerate(predict_probs):
        false_positive_rate,true_positive_rate,thresholds=roc_curve(labels, predict_prob)
        roc_auc=auc(false_positive_rate, true_positive_rate)
        c = color[idx%len(color)]
        plt.plot(false_positive_rate, true_positive_rate,'b',label='ROC curve class of 1 {}: auc={:.4}'.format(title[idx], roc_auc), color=c,lw=1)  
        plt.legend(loc='lower right')
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.plot([0,1],[0,1],'r--')
        plt.ylabel('TPR')
        plt.xlabel('FPR')
    plt.show()

#读取朴素贝叶斯、支持向量机、随机森林、Logistic以及其组合模型的测试集预测概率
nb_test_prob = pd.read_csv('朴素贝叶斯测试集预测概率.csv')
svm_test_prob = pd.read_csv('支持向量机测试集预测概率.csv')
logit_test_prob = pd.read_csv('Logistic测试集预测概率.csv')
rf_test_prob = pd.read_csv('随机森林测试集预测概率.csv')
con_test_prob = pd.read_csv('随机森林-Logistic测试集预测概率.csv')
true_label = pd.read_csv('测试集真实标签.csv')

#所有模型测试集的预测概率取正类预测概率
nb_test_prob_1 = list(nb_test_prob.iloc[:,2])
svm_test_prob_1 = list(svm_test_prob.iloc[:,2])
logit_test_prob_1 = list(logit_test_prob.iloc[:,2])
rf_test_prob_1 = list(rf_test_prob.iloc[:,2])
con_test_prob_1 = list(con_test_prob.iloc[:,2])
test_label = list(true_label.iloc[:,1])

predict_probs = [nb_test_prob_1,svm_test_prob_1,logit_test_prob_1,rf_test_prob_1,con_test_prob_1]
title = ['     NB      ','    SVM     ','  Logistic  ','      RF      ','Rf-Logistic']

#调用函数绘制ROC曲线
plot_roc(test_label, predict_probs,title)