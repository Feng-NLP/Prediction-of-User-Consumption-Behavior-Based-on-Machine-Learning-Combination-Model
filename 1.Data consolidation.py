import pandas as pd
import numpy as np

#读取数据
data = pd.read_csv("visit_login.csv")
data_1 = data.copy()

data_2 = pd.read_csv('user_info.csv',encoding='utf_8_sig')
data_3 = pd.read_csv('result.csv')

#将visit_login与user_info合并匹配
union_data_1 = pd.merge(data_1,data_2,how='left',left_on='user_id',right_on='user_id')

#在与result数据合并匹配
union_data_2 = pd.merge(union_data_1,data_3,how='left',left_on='user_id',right_on='user_id')
union_data_2.to_csv('三张表.csv',encoding='utf_8_sig')
