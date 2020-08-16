import torch
import torch.optim
import torch.nn as nn
from torch.autograd import Variable
from pytorch.DATA.loadDataLoader import trainDataLoader_Mnist,testDataLoader_Mnist

class Discriminator_CNN(nn.Module):
    def __init__(self):
        super(Discriminator_CNN,self).__init__()
        self.c1 = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=5,padding=2),
            nn.LeakyReLU(0.2,inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=5,padding=2),
            nn.LeakyReLU(0.2,inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64*7*7,512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.c1(x)
        out = self.c2(out)
        out = out.view(out.size(0),-1)
        out = self.fc(out)
        return out
class Generator_CNN(nn.Module):
    def __init__(self,z_dimension=10,num_features=3136):
        super(Generator_CNN,self).__init__()
        self.fc = nn.Linear(z_dimension,num_features)
        self.br = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )
        self.downSample1 = nn.Sequential(
            nn.Conv2d(1,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.downSample2 = nn.Sequential(
            nn.Conv2d(64,32,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.downSample3 = nn.Sequential(
            nn.Conv2d(32,1,kernel_size=2,stride=2),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.fc(x)
        out = out.view(out.size(0),1,56,56)
        out = self.br(out)
        out = self.downSample1(out)
        out = self.downSample2(out)
        out = self.downSample3(out)
        return out

z_dimension = 20
D = Discriminator_CNN()
G = Generator_CNN(z_dimension)

d_criterion = nn.BCELoss()
g_criterion = nn.MSELoss()

d_optimizer = torch.optim.Adam(D.parameters(),lr=0.0003)
g_optimizer = torch.optim.Adam(G.parameters(),lr=0.0003)

def train(epochs=30):
    for epoch in range(epochs):
        for x,y in trainDataLoader_Mnist:

            # ---------------------
            #  Train Discriminator
            # ---------------------
            # load real images
            real_img = Variable(x)
            real_score = D(real_img)
            real_label = Variable(torch.ones(real_score.shape[0],real_score.shape[1]))
            real_loss = d_criterion(real_score,real_label)

            # load fake images
            z = Variable(torch.randn(real_score.shape[0],z_dimension))
            fake_img = G(z)
            fake_score = D(fake_img)
            fake_label = Variable(torch.zeros(fake_score.shape[0],fake_score.shape[1]))
            fake_loss = d_criterion(fake_score,fake_label)

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
            g_loss_x = g_criterion(fake_img,x)

            score = D(fake_img)
            g_loss_d = d_criterion(score,real_label)

            g_loss = g_loss_x + g_loss_d
            print('g_loss:',g_loss.item(),'\n')
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

        print('\n-------------epoch:{}-------------\n'.format(epoch))

if __name__ == "__main__":
    train()
