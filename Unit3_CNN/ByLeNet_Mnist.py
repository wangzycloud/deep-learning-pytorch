import torch
import torch.nn as nn
import torch.optim as optimize
from torch.autograd import Variable
from pytorch.DATA.loadDataLoader import trainDataLoader_Mnist,testDataLoader_Mnist,size_Mnist
from CNN_LeNet import CNN_LeNet

model = CNN_LeNet()
criterion = nn.CrossEntropyLoss()
optimizer = optimize.Adam(model.parameters(),lr=1e-3)

trainDataLoader = trainDataLoader_Mnist
testDataLoader = testDataLoader_Mnist

if __name__ == "__main__":
    # 训练过程
    model.train()
    num_epochs = 100
    loss_old = 0
    loss_new = 0

    for epoch in range(num_epochs):
        for data in trainDataLoader:
            x, y = data
            x = Variable(x)
            y = Variable(y)

            out = model(x)
            loss = criterion(out, y)
            loss_new = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch[{}/{}], loss:{:.6f}'.format(epoch + 1, num_epochs, loss_new))
        if abs(loss_new - loss_old) < 0.0003:
            break
        loss_old = loss_new
    # 测试过程
    model.eval()
    num_correct = 0
    for data in testDataLoader:
        x, y = data
        x = Variable(x)
        y = Variable(y)

        out = model(x)
        loss = criterion(out, y)

        _, predict = torch.max(out, 1)
        correct = (predict == y).sum()
        num_correct += correct.item()
    print('test Acc: {:.6f}'.format(num_correct / size_Mnist[1]))
    # print('model\n', model)