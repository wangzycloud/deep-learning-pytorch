# GoogleNet
# 调用Inception模块，输入前指定输入通道数目C
# 得到[B 256 H W]，也就是输出通道数目为C=256
import torch
import torch.nn as nn
import torch.nn.functional as F
class BasicConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,**kwargs):
        super(BasicConv2d,self).__init__()
        self.conV = nn.Conv2d(in_channels,out_channels,bias=False,**kwargs)
        self.bn = nn.BatchNorm2d(out_channels,eps=0.001)

    def forward(self, x):
        x = self.conV(x)
        x = self.bn(x)
        out = F.relu(x)
        return out

class Inception(nn.Module):
    def __init__(self,in_channels=3,pool_features=32):
        super(Inception,self).__init__()
        self.branch1x1 = BasicConv2d(in_channels,64,kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels,48,kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48,64,kernel_size=5,padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels,64,kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64,96,kernel_size=3,padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96,96,kernel_size=3,padding=1)

        self.branch_pool = BasicConv2d(
            in_channels,pool_features,kernel_size=1
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl_1 = self.branch3x3dbl_1(x)
        branch3x3dbl_2 = self.branch3x3dbl_2(branch3x3dbl_1)
        branch3x3dbl_3 = self.branch3x3dbl_3(branch3x3dbl_2)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        out = (branch1x1,branch5x5,branch3x3dbl_3,branch_pool)
        return torch.cat(out,1)