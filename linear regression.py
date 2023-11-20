# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 16:38:01 2023

@author: OUYANG XI
"""

import scipy.stats as st
import pandas as pd

datas = pd.read_excel('passenger domestic.xlsx') # 读取 excel 数据，引号里面是 excel 文件的位置
y = datas.iloc[:, 1] # 因变量为第 2 列数据
x = datas.iloc[:, 2] # 自变量为第 3 列数据

# 线性拟合，可以返回斜率，截距，r 值，p 值，标准误差
slope, intercept, r_value, p_value, std_err = st.linregress(x, y)

print(slope)# 输出斜率
print(intercept) # 输出截距
print(r_value**2) # 输出 r^2
