import torch
import random
import torch.nn as nn
import torch.optim as optimize
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

def loadData(N):
    dataX = []
    for _ in range(N):
        x = random.uniform(-1,1)*10
        dataX.append(round(x,2))

    dataY = []
    for x in dataX:
        y = 2*x + random.gauss(0, 2)
        dataY.append(round(y,2))

    return dataX,dataY

class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression,self).__init__()
        self.linear = nn.Linear(1,1)

    def forward(self, x):
        out = self.linear(x)
        return out

model = LinearRegression()
criterion = nn.MSELoss()
optimizer = optimize.SGD(model.parameters(),lr=1e-3)

if __name__ == "__main__":
    dataX,dataY = loadData(50)
    x = torch.from_numpy(np.mat(np.array(dataX)).T).float()
    y = torch.from_numpy(np.mat(np.array(dataY)).T).float()

    num_epochs = 100
    for epoch in range(num_epochs):

        x = Variable(x)
        y = Variable(y)

        out = model(x)
        loss = criterion(out,y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            print('Epoch[{}/{}], loss:{:.6f}'.format(epoch + 1, num_epochs, loss.item()))

    a = model.linear.weight.item()
    b = model.linear.bias.item()
    print('回归方程为:y={:.2f}x+{:.2f}'.format(a,b))

    plot_x = np.arange(-10,10,0.1)
    plot_y = a*plot_x + b
    plt.plot(plot_x, plot_y, color='red')

    plt.scatter(x,y)
    plt.show()