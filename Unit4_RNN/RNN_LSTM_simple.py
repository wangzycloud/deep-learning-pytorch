import torch
import torch.nn as nn
from torch.autograd import Variable
class RNN_lstm_simple(nn.Module):
    def __init__(self,in_dim,n_class=10,n_layer=2,hidden_dim=32):
        super(RNN_lstm_simple,self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layer = n_layer
        self.n_class = n_class

        self.rnn = nn.LSTM(in_dim,hidden_dim,n_layer,batch_first=True)
        self.classifier = nn.Linear(hidden_dim,n_class)

    def forward(self, x):
        # 生成初始隐藏状态h0、初始记忆单元；批次s.size(0)
        h0 = Variable(torch.zeros(self.n_layer,x.size(0),self.hidden_dim))
        c0 = Variable(torch.zeros(self.n_layer,x.size(0),self.hidden_dim))

        out,_ = self.rnn(x,(h0,c0))
        # 取输出序列中的最后一个元素作为rnn的输出结果
        out = out[:,-1,:]
        out = self.classifier(out)
        return out