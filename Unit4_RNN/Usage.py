import torch
import torch.nn as nn
from torch.autograd import Variable

# # 定义一个2层的RNN网络
# 20是输入向量的维度（这个输入向量是20*1维的），50是隐藏层维度
basic_rnn = nn.RNN(input_size=20,hidden_size=50,num_layers=2)

# # 生成输入数据以及h0隐藏状态
# 100是输入序列的长度，20是输入向量的维度，32是批次
# 在RNN网络中，批次是第二维度，这与CNN差别很大，可以通过batch_first=True将批次放在第一个维度
toy_input = Variable(torch.randn(100,32,20))
h_0 = Variable(torch.randn(2,32,50))

# # 得到输出结果
# 输出向量的维度是50
toy_output,h_n = basic_rnn(toy_input,h_0)

# # 如果在传入网络的时候不特别注明隐藏状态h0，那么初始的隐藏状态默认是0
# toy_output,h_n = basic_rnn(toy_input)
print('toy_output：',toy_output.size())
print('h_n.size：',h_n.size())
print('----------------------------')

# # 定义一个2层的LSTM网络
lstm = nn.LSTM(input_size=20,hidden_size=10,num_layers=2)
h_0 = Variable(torch.randn(2,32,10))
c_0 = Variable(torch.randn(2,32,10))
lstm_out,(h_n,c_n) = lstm(toy_input,(h_0,c_0))
print('lstm_out.size：',lstm_out.size())
print('h_n.size：',h_n.size())
print('c_n.size：',c_n.size())
print('----------------------------')

gru = nn.GRU(input_size=20,hidden_size=10,num_layers=2)
gru_out,(h_n,c_n) = lstm(toy_input)
print('gru_out.size：',gru_out.size())
print('h_n.size：',h_n.size())
print('c_n.size：',c_n.size())