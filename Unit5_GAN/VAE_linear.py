import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from pytorch.DATA.loadDataLoader import trainDataLoader_Mnist,testDataLoader_Mnist

class VAE(nn.Module):
    def __init__(self):
        super(VAE,self).__init__()

        self.fc1 = nn.Linear(28*28,400)
        self.fc2_1 = nn.Linear(400,20)
        self.fc2_2 = nn.Linear(400,20)
        self.fc3 = nn.Linear(20,400)
        self.fc4 = nn.Linear(400,28*28)

    def encode(self,x):
        x = self.fc1(x)
        h1 = F.relu(x)
        return self.fc2_1(h1),self.fc2_2(h1)

    def reParametrize(self,mu,logVar):
        std = logVar.mul(0.5).exp_()
        eps = Variable(torch.FloatTensor(std.size()).normal_())
        return eps.mul(std).add_(mu)

    def decode(self,z):
        z = self.fc3(z)
        h3 = F.relu(z)
        fc4 = self.fc4(h3)
        return torch.sigmoid(fc4)

    def forward(self, x):
        mu,logVar = self.encode(x)
        z = self.reParametrize(mu,logVar)
        out = self.decode(z)
        return out,mu,logVar
class loss(nn.Module):
    def __init__(self):
        super(loss,self).__init__()
    def forward(self, recon_x,x,mu,logVar):
        BCE = torch.nn.functional.binary_cross_entropy_with_logits(recon_x,x)
        KLD_element = mu.pow(2).add_(logVar.exp()).mul_(-1).add_(1).add_(logVar)
        KLD = torch.sum(KLD_element).mul_(-0.5)
        return BCE + KLD

net = VAE()
optimizer = torch.optim.Adam(net.parameters(), lr=0.003)
criterion = loss()

net.train()
for epoch in range(50):
    loss_epoch = 0
    for x,y in trainDataLoader_Mnist:
        x = x[0].view(x[0].size(0), -1)
        x = Variable(x)
        out,mu,logVar = net(x)

        loss = criterion(out,x,mu,logVar)
        loss_epoch += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('loss :',loss_epoch)