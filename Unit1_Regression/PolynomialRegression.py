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
        x = random.uniform(-1,1)*5
        dataX.append(round(x,2))

    dataY = []
    for x in dataX:
        y = 2.4*x**3 + 3*x**2 + 1*x + random.gauss(0, 20)
        dataY.append(round(y, 2))

    return dataX, dataY
def get_input(dataX):
    xs = []
    for x in dataX:
        xs.append([x**3,x**2,x])
    return xs

class PolynomialRegression(nn.Module):
    def __init__(self):
        super(PolynomialRegression,self).__init__()
        self.poly = nn.Linear(3,1)

    def forward(self, x):
        out = self.poly(x)
        return out

model = PolynomialRegression()
criterion = nn.MSELoss()
optimizer = optimize.Adam(model.parameters(),lr=1e-2)

if __name__ == "__main__":
    dataX,dataY = loadData(20)
    get_dataX = get_input(dataX)

    x = torch.from_numpy(np.mat(np.array(get_dataX))).float()
    y = torch.from_numpy(np.mat(np.array(dataY)).T).float()

    model.train()
    num_epochs = 20000
    loss_old = 0
    for epoch in range(num_epochs):

        x = Variable(x)
        y = Variable(y)

        out = model(x)
        loss = criterion(out, y)
        loss_new = loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            print('Epoch[{}/{}], loss:{:.6f}'.format(epoch + 1, num_epochs,loss_new))
        if abs(loss_new-loss_old) < 0.0003:
            break
        loss_old = loss_new

    weight = model.poly.weight
    a = float(weight[0][0])
    b = float(weight[0][1])
    c = float(weight[0][2])
    d = float(model.poly.bias[0])

    print('回归方程为:y={:.2f}*x^3 + {:.2f}*x^2 + {:.2f}*x + {:.2f}'.format(a, b, c, d))

    plot_x = np.arange(-5, 5, 0.1)
    plot_y = a*plot_x**3 + b*plot_x**2 + c*plot_x + d
    plt.plot(plot_x, plot_y, color='red')

    plt.scatter(dataX, dataY)
    plt.show()



