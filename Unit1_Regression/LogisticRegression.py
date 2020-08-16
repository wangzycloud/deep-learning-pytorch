import torch
import torch.nn as nn
import torch.optim as optimize
from torch.autograd import Variable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def loadData():
    iris = pd.read_csv('..\DATA\DATA-iris\iris.csv', usecols=[1, 2, 3, 4, 5])
    iris = iris.values
    iris = iris[0:100]
    inputVecs = iris[:, [0, 1]].astype('float')
    labels = iris[:, [4]]
    for i in range(labels.shape[0]):
        if labels[i][0] == 'Iris-setosa':
            labels[i][0] = 1
        elif labels[i][0] == 'Iris-versicolor':
            labels[i][0] = 0
    return inputVecs, labels.astype(float)

class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression,self).__init__()
        self.linear = nn.Linear(2,1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out = self.linear(x)
        out = self.sigmoid(out)
        return out

model = LogisticRegression()
criterion = nn.BCELoss()
optimizer = optimize.SGD(model.parameters(),lr=1e-3,momentum=0.9)

if __name__ == "__main__":
    dataX,dataY = loadData()

    x = torch.from_numpy(dataX).float()
    y = torch.from_numpy(dataY).float()

    num_epochs = 10000
    for epoch in range(num_epochs):

        x = Variable(x)
        y = Variable(y)

        out = model(x)
        loss = criterion(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            print('Epoch[{}/{}], loss:{:.6f}'.format(epoch + 1, num_epochs, loss.item()))

    weight = model.linear.weight
    w1 = float(weight[0][0])
    w2 = float(weight[0][1])
    b = float(model.linear.bias[0])

    print('线性方程为:y={:.2f}*x1 + {:.2f}*x2 + {:.2f}'.format(w1, w2, b))

    plot_x = np.arange(4, 7.5, 0.1)
    plot_y = (-w1*plot_x -b )/w2
    plt.plot(plot_x,plot_y)

    x0 = [x for i,x in enumerate(dataX) if dataY[i] == 0]
    x1 = [x for i,x in enumerate(dataX) if dataY[i] == 1]

    sca_x0_0 = [x[0] for x in x0]
    sca_x0_1 = [x[1] for x in x0]
    sca_x1_0 = [x[0] for x in x1]
    sca_x1_1 = [x[1] for x in x1]

    plt.scatter(sca_x0_0,sca_x0_1,label='x0')
    plt.scatter(sca_x1_0,sca_x1_1,label='x1')
    plt.legend(loc='best')
    plt.show()