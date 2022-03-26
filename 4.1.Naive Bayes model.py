import numpy as np
import pandas as pd
import xlwt
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

#数据读取
data = pd.read_csv('平衡数据集B.csv')
x = data.iloc[:,1:21]
y = data.iloc[:,21]

#训练集、测试集的划分
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20,random_state=7)

#训练模型
clf = GaussianNB()
clf.fit(x_train, y_train)

#模型预测
y_pred_prob = clf.predict_proba(x_test)
y_pred_prob = pd.DataFrame(y_pred_prob)
y_predict = clf.predict(x_test)

#保存测试集预测概率，用于绘制ROC曲线
y_pred_prob.to_csv('朴素贝叶斯测试集预测概率.csv')

y_predict = np.array(y_predict)
y_real = np.array(y_test)

y_true = y_test
y_pred = y_predict

#混淆矩阵
classes = list(set(y_true))
classes.sort()
confusion = confusion_matrix(y_pred, y_true)
plt.imshow(confusion, cmap=plt.cm.Blues)
indices = range(len(confusion))
plt.xticks(indices, classes)
plt.yticks(indices, classes)
plt.colorbar()
plt.xlabel('y_pred')
plt.ylabel('y_true')
for first_index in range(len(confusion)):
    for second_index in range(len(confusion[first_index])):
        plt.text(first_index, second_index, confusion[first_index][second_index])
 
plt.show()

#kappa系数
kappa = cohen_kappa_score(y_true,y_pred)
print("kappa:",kappa)

#正确率
print("ACC：",accuracy_score(y_true, y_pred))

#精确率
print("precision：",precision_score(y_true, y_pred))

#F1分数
print("F1_score",f1_score(y_true, y_pred))