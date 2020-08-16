import torch.optim
import torch.nn as nn
from torch.autograd import Variable
from pytorch.DATA.loadDataLoader import trainDataLoader_Mnist,testDataLoader_Mnist

class AutoEncoder_CNN(nn.Module):
    def __init__(self):
        super(AutoEncoder_CNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,16,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(16,8,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8,16,kernel_size=4,stride=2,padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16,8,kernel_size=4,stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(8,1,kernel_size=3,stride=1,padding=1),
            nn.Tanh()
        )
    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out

net = AutoEncoder_CNN()
optimizer = torch.optim.Adam(net.parameters(), lr=0.003)
criterion = nn.MSELoss()

net.train()
for epoch in range(50):
    loss_epoch = 0
    for x,y in trainDataLoader_Mnist:
        x = Variable(x)
        out = net(x)

        loss = criterion(out,x)
        loss_epoch += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('loss :',loss_epoch)