import torch
import torch.nn as nn
import torch.optim as optimize
from torch.autograd import Variable
from pytorch.DATA.loadDataLoader import trainDataLoader_Mnist,testDataLoader_Mnist,size_Mnist
from RNN_LSTM_simple import RNN_lstm_simple

model = RNN_lstm_simple(28)
criterion = nn.CrossEntropyLoss()
optimizer = optimize.Adam(model.parameters(),lr=1e-3)

trainDataLoader = trainDataLoader_Mnist
testDataLoader = testDataLoader_Mnist

if __name__ == "__main__":
    # 训练过程
    model.train()
    num_epochs = 50
    loss_old = 0
    loss_new = 0

    for epoch in range(num_epochs):
        for data in trainDataLoader:
            x, y = data
            x = Variable(x.squeeze(1))
            # 这里降维的目的，是要符合模型输入的维度，输入该模型的数据要有以下三个维度：
            # （批次，输入向量x的序列长度，输入向量x的维度）
            # 我们知道，现在使用的dataLoader是CNN时的加载方式，数据维度为：（B=16,C=1,H=28,W=28）
            # 现在可以把读取到的C=1的图像数据，在通道层次上进行降维，
            # 之后将图片数据理解成28个28*1维的向量。
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
        x = Variable(x.squeeze(1))
        y = Variable(y)

        out = model(x)
        loss = criterion(out, y)

        _, predict = torch.max(out, 1)
        correct = (predict == y).sum()
        num_correct += correct.item()
    print('test Acc: {:.6f}'.format(num_correct / size_Mnist[1]))
    # print('model\n', model)