import numpy as np
import xlrd2
import xlwt

'''z-检验'''
'''比较不同类别之间的化学成分关联关系的相关系数'''
data = xlrd2.open_workbook("../dataset/相关系数表.xlsx")
sheet1 = data.sheet_by_index(0)
sheet2 = data.sheet_by_index(1)
nrows = sheet1.nrows  # 获取行数

x1 = []
x2 = []
for i in range(nrows):
    row1 = sheet1.row_values(i)
    x1.append(row1)
    row2 = sheet2.row_values(i)
    x2.append(row2)

tongji = np.zeros(shape=(14, 14))

for i in range(1, 14):
    for j in range(i):
        r1 = x1[i][j]
        r2 = x2[i][j]
        # if type(x1[i][j]) == str or type(x2[i][j]) == str:
        if type(x1[i][j]) == str and type(x2[i][j]) == str:
            r1 = float(x1[i][j].replace('*', ''))
            z1 = 0.5*np.log((1+r1)/(1-r1))
            r2 = float(x2[i][j].replace('*', ''))
            z2 = 0.5 * np.log((1 + r2) / (1 - r2))
            z = z1-z2
            sigma = np.sqrt(1/(49-3)+1/(18-3))
            tongji[i][j] = abs(z/sigma)

signif = []
for i in range(14):
    for j in range(14):
        if tongji[i][j] != 0:
            if tongji[i][j] > 1.64 and tongji[i][j] < 2.57:  # 0.05
                #print('差异显著:' + str(i+1) + '行,' + str(j+1) + '列 ' + str(tongji[i][j]))
                signif.append([i, j])

signif_very = []
for i in range(14):
    for j in range(14):
        if tongji[i][j] != 0:
            if tongji[i][j] > 2.57:  # 0.005
                #print('差异非常显著:' + str(i + 1) + '行,' + str(j + 1) + '列 ' + str(tongji[i][j]))
                signif_very.append([i, j])

not_signif = []
for i in range(14):
    for j in range(14):
        if tongji[i][j] != 0:
            if tongji[i][j] < 1.64:
                #print('差异不显著:' + str(i+1) + '行,' + str(j+1) + '列 ' + str(tongji[i][j]))
                not_signif.append([i, j])

book = xlwt.Workbook(encoding='utf-8',style_compression=0)
sheet = book.add_sheet('sheet',cell_overwrite_ok=True)

for i in range(len(signif_very)):
        sheet.write(signif_very[i][0], signif_very[i][1], "差异非常显著")

for i in range(len(signif)):
        sheet.write(signif[i][0], signif[i][1], "差异显著")

for i in range(len(not_signif)):
        sheet.write(not_signif[i][0], not_signif[i][1], "差异不显著")

savepath = '../result/z-检验结果.xlsx'
book.save(savepath)