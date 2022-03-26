import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import precision_score,accuracy_score
from sklearn.metrics import f1_score

#数据读取
data = pd.read_csv('平衡数据集B.csv')
x = data.iloc[:,1:21]
y = data.iloc[:,21]

#划分训练集与测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20,random_state=7)

#构建并训练logistics模型
clf = LogisticRegression(max_iter=500,solver='newton-cg')
clf.fit(x_train,y_train)

#将训练好的模型用于测试集的预测
y_pred_prob = clf.predict_proba(x_test)
y_pred_prob = pd.DataFrame(y_pred_prob)
y_pred = clf.predict(x_test)

#保存测试集预测概率，用于绘制ROC曲线
y_pred_prob.to_csv('Logistic测试集预测概率.csv')

y_true = y_test.copy()

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