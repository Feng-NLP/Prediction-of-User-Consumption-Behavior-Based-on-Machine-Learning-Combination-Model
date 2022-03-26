import pandas as pd
import numpy as np

#读取数据
data = pd.read_csv("三张表.csv",encoding='utf_8_sig')
df = data.copy()

#登录时长与登录天数的异常值筛选
df_clear = df.drop(data[(data['login_time']==0) & (data['login_day']!=-1)].index)
df_clear.to_csv('登录时长与登录天数_outlier_1.csv',encoding='utf_8_sig')

#最后登录距期末和登录天数的异常值筛选
df_clear_2 = df_clear.drop(df_clear[(df_clear['distance_day']<0) & (df_clear['login_day']!=-1)].index)
df_clear_2.to_csv('最后登录距期末和登录天数_outlier_2.csv',encoding='utf_8_sig')

#开课节数与学习课节数、完成课节数的异常值筛选
df_clear_3 = df_clear_2.drop(df_clear_2[(df_clear_2['camp_num']==0) & (df_clear_2['learn_num']!=0)].index)
df_clear_4 = df_clear_3.drop(df_clear_3[(df_clear_3['camp_num']==0) & (df_clear_3['finish_num']!=0)].index)
df_clear_5 = df_clear_4.drop(df_clear_4[(df_clear_4['learn_num']==0) & (df_clear_4['finish_num']!=0)].index)
df_clear_6 = df_clear_5.drop(df_clear_5[(df_clear_5['learn_num']<df_clear_5['finish_num'])].index)
df_clear_6.to_csv('class_outlier_3.csv',encoding='utf_8_sig')


#领券数量与领券访问数的异常值筛选
df_clear_7 = pd.read_csv('class_outlier_3.csv',encoding='utf_8_sig')
df_2 = df_clear_7.drop(df_clear_7[(df_clear_7['coupon_visit']==0) & (df_clear_7['coupon']!=0)].index)
df_2.to_csv('web_visit_outlier_4.csv',encoding='utf_8_sig')