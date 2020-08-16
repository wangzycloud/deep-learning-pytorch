import torch
import torch.optim
import torch.nn as nn
from torch.autograd import Variable
from pytorch.DATA.loadDataLoader import trainDataLoader_Mnist,testDataLoader_Mnist
import matplotlib.pyplot as plt

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder,self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3,12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out

net = AutoEncoder()
optimizer = torch.optim.Adam(net.parameters(), lr=0.003)
criterion = nn.MSELoss()

def train(epochs=20):
    net.train()
    for epoch in range(epochs):
        loss_epoch = 0
        for x in trainDataLoader_Mnist:
            x = x[0].view(x[0].size(0),-1)
            x = Variable(x)
            out = net(x)

            loss = criterion(out,x)
            loss_epoch += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('loss :',loss_epoch)
def test():
    net.eval()
    for x in testDataLoader_Mnist:
        x = x[0].view(x[0].size(0), -1)
        x = Variable(x)
        out = net(x)

        # 预测
        predict = out[0].detach().reshape((28,28))
        # plt.imshow(predict)
        # plt.show()

if __name__ == "__main__":
    train()
    test()