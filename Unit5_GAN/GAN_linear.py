import torch
import torch.optim
import torch.nn as nn
from torch.autograd import Variable
from pytorch.DATA.loadDataLoader import trainDataLoader_Mnist,testDataLoader_Mnist

class Discriminator_linear(nn.Module):
    def __init__(self):
        super(Discriminator_linear,self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(28*28,256),
            nn.LeakyReLU(0.2),
            nn.Linear(256,256),
            nn.LeakyReLU(0.2),
            nn.Linear(256,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.dis(x)
        return out
class Generator_liner(nn.Module):
    def __init__(self,z_dimension=10):
        super(Generator_liner,self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dimension,256),
            nn.ReLU(True),
            nn.Linear(256,256),
            nn.ReLU(True),
            nn.Linear(256,28*28),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.gen(x)
        return out

z_dimension = 20
D = Discriminator_linear()
G = Generator_liner(z_dimension)

criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(),lr=0.0003)
g_optimizer = torch.optim.Adam(G.parameters(),lr=0.0003)

def train(epochs=30):
    for epoch in range(epochs):
        for x in trainDataLoader_Mnist:

            # ---------------------
            #  Train Discriminator
            # ---------------------
            # load real images
            x = x[0].view(x[0].size(0),-1)
            real_img = Variable(x)
            real_score = D(real_img)
            real_label = Variable(torch.ones(real_score.shape[0],real_score.shape[1]))
            real_loss = criterion(real_score,real_label)

            # load fake images
            z = Variable(torch.randn(real_score.shape[0],z_dimension))
            fake_img = G(z)
            fake_score = D(fake_img)
            fake_label = Variable(torch.zeros(fake_score.shape[0],fake_score.shape[1]))
            fake_loss = criterion(fake_score,fake_label)

            # 判别器参数更新
            d_loss = real_loss+fake_loss
            print('d_loss:',d_loss.item())
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # -----------------
            #  Train Generator
            # -----------------
            z = Variable(torch.randn(real_score.shape[0], z_dimension))
            fake_img = G(z)
            score = D(fake_img)

            # 生成器参数更新
            g_loss = criterion(score,real_label)
            print('g_loss:',g_loss.item(),'\n')
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

        print('\n-------------epoch:{}-------------\n'.format(epoch))

if __name__ == "__main__":
    train()
