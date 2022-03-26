import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import precision_score,accuracy_score
from sklearn.metrics import f1_score

#数据读取
data_A = pd.read_csv('平衡数据集A.csv')
data_B = pd.read_csv('平衡数据集B.csv')

x_A = data_A.iloc[:,0:20]
y_A = data_A.iloc[:,20]
x_B = data_B.iloc[:,0:21]
y_B = data_B.iloc[:,21]

#使用数据集A训练随机森林模型
clf_1 = RandomForestClassifier(criterion="entropy",n_estimators=100,random_state=0)#
clf_1.fit(x_A,y_A)

#使用训练好的随机森林对平衡数据集B做预测
y_B_pred = clf_1.predict(x_B.iloc[:,1:21])
x_B_y_pred = pd.DataFrame(y_B_pred)

#构建平衡数据集C
x_C = x_B.copy()
y_C = y_B.copy()
x_C['clf_1_pred'] = x_B_y_pred

#对平衡数据集C 按 8:2 划分为训练集、测试集
x_train, x_test, y_train, y_test = train_test_split(x_C, y_C, test_size=0.20,random_state=7)

#使用平衡数据集C的训练集训练logistic模型
clf_2 = LogisticRegression(max_iter=500,solver='newton-cg')
clf_2.fit(x_train.iloc[:,1:22],y_train)

#使用训练好的模型对测试集进行预测
y_pred_prob = clf_2.predict_proba(x_test.iloc[:,1:22])
y_pred_prob = pd.DataFrame(y_pred_prob)
y_pred = clf_2.predict(x_test.iloc[:,1:22])

#保存测试集预测概率，用于绘制ROC曲线
y_pred_prob.to_csv('随机森林-Logistic测试集预测概率.csv')

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

#将测试集的模型预测结果保存为csv文件
test_pred = y_pred.copy()
test_probability = y_pred_prob.iloc[:,1].copy()
user_id = x_test['user_id'].copy()


model_results = pd.DataFrame([list(user_id),list(test_pred),list(test_probability)]).T
model_results.columns=["user_id","result","result_probability"]

model_results.to_csv('model_sample_output.csv',index=None)