from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''分析两类玻璃的分类规律'''
'''特征选择'''


def feature_selection(df):
    model = RandomForestRegressor(random_state=1, max_depth=10)
    df = pd.get_dummies(df)
    model.fit(df, train.Type)

    features = df.columns
    importances = model.feature_importances_
    indices = np.argsort(importances[0:14])
    plt.rcParams['font.sans-serif'] = 'SimHei'  # 设置中文显示
    plt.title('14种化学成分')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('相对重要性')
    plt.show()


train = pd.read_csv("../dataset/table2_all.csv")
df1 = train.drop(['Type'], axis=1)
title1 = '14种化学成分'
df2 = train.drop(['Type', 'PbO'], axis=1)
title2 = '除去PbO的化学成分'

feature_selection(df1)
feature_selection(df2)
