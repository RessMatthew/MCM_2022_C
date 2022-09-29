# -*- coding: utf-8 -*-

import joblib
import numpy as np
import pandas as pd

""" 
    预测附件表三类型
"""


def data_handle(filename):
    read_data = pd.read_csv(filename)
    list_datasets = []
    category_labels = []
    for i in range(len(read_data)):
        list_data = []
        for j in range(len(read_data.iloc[i, :]) - 1):
            row_data = read_data.iloc[i, j]  # 读取每个样本的每个数据i以他人为主
            list_data.append(row_data)  # 将每个数据存入列表
        list_datasets.append(list_data)  # 将每个样本的数据存入列表

        row_data_label = read_data.iloc[i, len(read_data.iloc[i, :]) - 1]  # 读取每个样本的类别标签
        if row_data_label == 0:
            category_labels.append(0)  # 将二分类标签转化为0和1,0代表软件正常，1代表软件缺陷
        else:
            category_labels.append(1)
    return list_datasets, category_labels


def preditc():
    datasets, labels = data_handle('../dataset/table3_q3.csv')  # 对数据集进行处理
    X_test = datasets[:]
    clf0 = joblib.load("../result/temp.pkl")
    y_predict = clf0.predict(X_test)  # 使用分类器对测试集进行预测
    np.savetxt('../result/predict.txt', y_predict)
    print(y_predict)


if __name__ == '__main__':
    preditc()
