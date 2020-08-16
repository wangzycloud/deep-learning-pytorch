import os
import struct
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optimize
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader,random_split

PATH = '../DATA/DATA-mnist/'
def load_MNIST(path, kind='t10k'):
    # 将字节数据加载至Numpy
    labels_path = os.path.join(path,'%s-labels.idx1-ubyte'% kind)
    images_path = os.path.join(path,'%s-images.idx3-ubyte'% kind)

    with open(labels_path, 'rb') as lbPath:
        magic, n = struct.unpack('>II',lbPath.read(8))
        labels = np.fromfile(lbPath,dtype=np.uint8)

    with open(images_path, 'rb') as imgPath:
        magic, num, rows, cols = struct.unpack('>IIII',imgPath.read(16))
        images = np.fromfile(imgPath,dtype=np.uint8).reshape(len(labels), 784)

    return images, labels
class Datasets(Dataset):
    def __init__(self,dataX,dataY):
        self.dataX = dataX
        self.dataY = dataY

    def __len__(self):
        return len(self.dataX)

    def __getitem__(self, idx):
        x = self.dataX[idx]
        x = torch.from_numpy(x).float()

        y = self.dataY[idx]
        y = np.array(y)
        y = torch.from_numpy(y).long()
        return x,y
class FullyConnectedNet(nn.Module):
    def __init__(self):
        super(FullyConnectedNet,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(784,300),
            nn.BatchNorm1d(300),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(300,100),
            nn.BatchNorm1d(100),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(100,10)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out

model = FullyConnectedNet()
criterion = nn.CrossEntropyLoss()
optimizer = optimize.Adam(model.parameters(),lr=1e-3)

if __name__ == "__main__":
    img, lab = load_MNIST(PATH)
    DATA = Datasets(img, lab)
    trainSize = int(0.7*len(DATA))
    testSize = len(DATA) - trainSize
    trainDATA, testDATA = random_split(DATA, [trainSize,testSize])

    # 训练过程
    trainDataLoader = DataLoader(trainDATA, batch_size=64, shuffle=True)
    model.train()
    num_epochs = 50
    loss_old = 0
    loss_new = 0

    for epoch in range(num_epochs):

        for data in trainDataLoader:
            x,y = data
            x = Variable(x)
            y = Variable(y)

            out = model(x)
            loss = criterion(out, y)
            loss_new = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch[{}/{}], loss:{:.6f}'.format(epoch + 1, num_epochs,loss_new))
        if abs(loss_new-loss_old) < 0.0003:
            break
        loss_old = loss_new

    # 测试过程
    testDataLoader = DataLoader(testDATA, batch_size=64, shuffle=True)
    num_correct = 0
    model.eval()

    for data in testDataLoader:
        x,y = data
        x = Variable(x)
        y = Variable(y)

        out = model(x)
        loss = criterion(out,y)

        _,predict = torch.max(out,1)
        correct = (predict == y).sum()
        num_correct += correct.item()
    print('test Acc: {:.6f}'.format(num_correct/testSize))
    print('model\n',model)
    print('model.children\n',list(model.children()))
    # print('model.named_children\n',list(model.named_children()))
    # print('model.named_modules\n',list(model.named_modules()))

    for param in model.named_parameters():
        print(param[0])