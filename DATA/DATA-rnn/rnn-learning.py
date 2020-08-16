import pandas as pd
import matplotlib.pyplot as plt

data_csv = pd.read_csv('.\data.csv')
data_csv.info()
data_csv = data_csv.dropna()
dataSet = data_csv.values
print(dataSet)

# 1960年12个月的流量情况
year_1960 = dataSet[-12:]
year_1960X,year_1960Y = [],[]
for i in range(len(year_1960)):
    year_1960X.append(int(year_1960[i][0][-2:]))
    year_1960Y.append(float(year_1960[i][1]))

plt.figure()
ax1 = plt.subplot(1,3,1)
plt.title('year:1960')
plt.bar(year_1960X,year_1960Y)

# 1949年至1960年各个月飞机的流量情况
dataX,dataY = [],[]
for i,x in enumerate(dataSet):
    dataX.append(i+1)
    dataY.append(x[1])
ax2 = plt.subplot(1,3,(2,3))
plt.title('year:1949-1960')
ax2.yaxis.set_ticks_position('right')
plt.plot(dataX,dataY)
plt.show()