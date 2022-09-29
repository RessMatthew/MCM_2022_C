import numpy as np
import pylab as pl
from sklearn import svm

'''绘制二分类示意图'''

# 生成训练实例
np.random.seed(0)
X = np.r_[np.random.randn(50, 2) - [2, 2], np.random.randn(50, 2) + [2, 2]]
print(X)
Y = [0] * 50 + [1] * 50

# 建立 svm 模型
clf = svm.SVC(kernel='linear')
clf.fit(X, Y)

# 获得划分超平面
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - (clf.intercept_[0]) / w[1]

# 画出支持向量的两条线（斜率相同，截距不同）
b = clf.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = clf.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])

# 查看相关的参数值
print("w: ", w)
print("a: ", a)
print("support_vectors_: ", clf.support_vectors_)

# 绘制超平面，边际平面和样本点
pl.plot(xx, yy, 'k-')
pl.plot(xx, yy_down, 'k--')
pl.plot(xx, yy_up, 'k--')

# 圈出支持向量
clf.support_vectors_[:, 1],
pl.scatter(X[:, 0], X[:, 1], c=Y, cmap=pl.cm.Paired)

pl.axis('tight')
pl.show()
