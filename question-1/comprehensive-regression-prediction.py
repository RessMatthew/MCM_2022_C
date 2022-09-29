import pandas as pd

'''分别对高钾和铅钡预测风化前化学成分含量进行预测'''
'''综合回归预测'''


def prediction(datapath, filepath, list_xishu, changshu):
    read_data = pd.read_csv(datapath)
    list_datasets = []
    for i in range(len(read_data)):
        list_data = []
        for j in range(len(read_data.iloc[i, :]) - 2):
            row_data = read_data.iloc[i, j]
            list_data.append(row_data)
        list_datasets.append(list_data)

    xishupingfanghe = 0
    for i in range(len(list_xishu)):
        xishupingfanghe += list_xishu[i] * list_xishu[i]

    list_xiangxishuhe = []
    xiangxishuhe = 0
    for i in range(len(list_datasets)):
        for j in range(len(list_datasets[0])):
            xiangxishuhe += list_datasets[i][j] * list_xishu[j]
        list_xiangxishuhe.append(xiangxishuhe)

    list_x = []
    for i in range(len(list_datasets)):
        x = (list_xiangxishuhe[i] + changshu) / xishupingfanghe
        list_x.append(x)

    list_result = []
    for i in range(len(list_datasets)):
        row_result = []
        for j in range(len(list_datasets[0])):
            data = list_datasets[i][j] - float(list_xishu[j]) * float(list_x[i])
            row_result.append(data)
        list_result.append(row_result)

    name = ['SiO2', 'Na2O', 'K2O', 'CaO', 'MgO', 'Al2O3', 'Fe2O3', 'CuO', 'PbO', 'BaO', 'P2O5', 'SrO', 'SnO2', 'SO2']
    test = pd.DataFrame(columns=name, data=list_result)
    print(list_result)
    test.to_csv(filepath, encoding='gbk')


# 高钾类型
list_xishu = [0.1183, 0.2938, 0.1211, 0.0057, 1.2599, -0.0912, 0.2503, 0.2874, 0.3632, -0.5532, -0.1046, 1.5952,
              -0.5967, -2.1349]
changshu = -10.7513
datapath = "../dataset/table2_gj_predict.csv"
filepath = '../result/高钾类风化前预测.csv'
prediction(datapath, filepath, list_xishu, changshu)

# 铅钡类型
list_xishu = [0.0101, 0.0028, -0.3440, -0.0357, 0.0658, 0.0250, 0.1076, 0.0318, 0.0378, 0.0215, 0.0468, -0.6115, 0.3390,
              0.0364]
changshu = -1.4130
datapath = "../dataset/table2_qb_predict.csv"
filepath = '../result/铅钡类风化前预测.csv'
prediction(datapath, filepath, list_xishu, changshu)
