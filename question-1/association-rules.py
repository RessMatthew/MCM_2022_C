from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import pandas as pd

'''分析玻璃文物表面风化和玻璃类型、纹饰以及颜色的关系'''
'''关联规则强度效度'''

# 读取table1，转换为list(list)对象df_arr
df_arr = []
read_data = pd.read_csv("../dataset/table1.csv")
for i in range(len(read_data)):
    list_data = []
    for j in range(len(read_data.iloc[i, :])):
        row_data = read_data.iloc[i, j]  # 读取每个样本的每个数据i
        list_data.append(row_data)  # 将每个数据存入列表
    df_arr.append(list_data)

# 转换为算法可接受模型（布尔值）
te = TransactionEncoder()
df_tf = te.fit_transform(df_arr)
df = pd.DataFrame(df_tf, columns=te.columns_)

# 利用 Apriori,设置支持度求频繁项集
frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)  # min_support为组合出现数/总项数
print(frequent_itemsets)

# 求关联规则,设置最小置信度为0.15
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.15)

rules = rules.drop(rules[rules.lift < 1.0].index)  # 设置最小提升度
rules.rename(columns={'antecedents': 'from', 'consequents': 'to', 'support': 'sup', 'confidence': 'conf'}, inplace=True)
rules = rules[['from', 'to', 'sup', 'conf', 'lift']]
# 排序查询数据，按提升度lift,置信度conf，支持度sup的降序排序
r = rules.sort_values(by=['lift', 'conf', 'sup'], ascending=False)
print(r)