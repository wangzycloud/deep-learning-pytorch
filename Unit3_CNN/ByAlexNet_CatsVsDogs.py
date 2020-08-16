import torch
import torch.nn as nn
import torch.optim as optimize
from torch.autograd import Variable
from pytorch.DATA.loadDataLoader import trainDataLoader_CatVsDog,testDataLoader_CatVsDog,size_CatVsDog
from CNN_AlexNet import CNN_AlexNet

model = CNN_AlexNet(2)
# model = CNN_VGG16(2)
criterion = nn.CrossEntropyLoss()
optimizer = optimize.Adam(model.parameters(),lr=1e-3)
trainDataLoader = trainDataLoader_CatVsDog
testDataLoader = testDataLoader_CatVsDog

if __name__ == "__main__":
    # 训练过程
    model.train()
    num_epochs = 20
    loss_old = 0
    loss_new = 0

    for epoch in range(num_epochs):
        loss_epoch = 0
        for data in trainDataLoader:
            x, y = data
            x = Variable(x)
            y = Variable(y)

            out = model(x)

            loss = criterion(out, y)
            loss_new = loss.item()
            loss_epoch += loss_new

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch[{}/{}], loss:{:.6f}'.format(epoch + 1, num_epochs, loss_epoch))
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
        print(loss.item())

        _, predict = torch.max(out, 1)
        correct = (predict == y).sum()
        num_correct += correct.item()
    print('test Acc: {:.6f}'.format(num_correct / size_CatVsDog))
    # print('model\n', model)