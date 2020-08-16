import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class lstm_reg(nn.Module):
    def __init__(self, input_size=2, hidden_size=5, output_size=1, num_layers=2):
        super(lstm_reg, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers)  # rnn
        self.reg = nn.Linear(hidden_size, output_size)  # 回归

    def forward(self, x):
        x, _ = self.rnn(x)  # (seq, batch, hidden)
        s, b, h = x.shape
        x = x.view(s * b, h)  # 拉伸为线性层的输入格式
        x = self.reg(x)
        x = x.view(s, b, -1)
        return x
def loadData(file='..\DATA\DATA-rnn\data.csv'):
    dataCsv = pd.read_csv(file)
    dataCsv = dataCsv.dropna()
    dataCsv = dataCsv.values
    data = [x[1] for x in dataCsv]

    maxVal = np.max(data)
    minVal = np.min(data)
    scalar = maxVal - minVal
    data = list(map(lambda x: x / scalar, data))
    return data
def createDataSet(data,look_back = 2):
    dataX, dataY = [], []
    for i in range(len(data) - look_back):
        a = data[i:(i + look_back)]
        dataX.append(a)
        dataY.append(data[i + look_back])
    return np.array(dataX), np.array(dataY)
def generateData(dataX,dataY):
    train_size = int(len(dataX) * 0.7)
    test_size = len(dataX) - train_size
    trainX = dataX[:train_size]
    trainY = dataY[:train_size]
    testX = dataX[train_size:]
    testY = dataY[train_size:]

    # 构造符合要求的数据，如trainX.shape:(99,1,2)。99是指序列长度，1是批次，2是输入向量的维度（两个月）
    trainX = torch.from_numpy(trainX.reshape(-1,1,2)).float()
    trainY = torch.from_numpy(trainY.reshape(-1,1,1)).float()
    testX = torch.from_numpy(testX.reshape(-1,1,2)).float()

    return trainX,trainY,testX,testY

model = lstm_reg()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-2)

if __name__ == "__main__":
    # 加载数据
    data = loadData()
    dataX, dataY = createDataSet(data)
    trainX, trainY, testX, testY = generateData(dataX, dataY)

    # 进行训练
    model.train()
    num_epochs = 1000
    for epoch in range(num_epochs):
        x = Variable(trainX)
        y = Variable(trainY)
        # 前向传播
        out = model(x)
        loss = criterion(out, y)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0: # 每 100 次输出结果
            print('Epoch: {}, Loss: {:.5f}'.format(epoch + 1, loss.item()))

    model.eval()
    x_t = Variable(testX)
    out = model(x_t)

    predict = [float(x[0][0]) for x in out]
    # train_Y = [float(x[0][0]) for x in trainY]

    plt.plot(range(len(testY)),testY)
    plt.plot(range(len(testY)),predict)
    plt.show()
