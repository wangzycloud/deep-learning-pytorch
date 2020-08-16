import torch.nn as nn
class CNN_LeNet(nn.Module):
    def __init__(self):
        super(CNN_LeNet,self).__init__()
        self.layer1_C1 = nn.Conv2d(3,6,kernel_size=5)
        self.layer2_S2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Sigmoid()
        )
        self.layer3_C3 = nn.Conv2d(6,16,kernel_size=5)
        self.layer4_S4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Sigmoid()
        )
        self.layer5_C5 = nn.Linear(400,120)
        self.layer6_F6 = nn.Linear(120,84)
        self.layer7_OUTPUT = nn.Linear(84,10)
    def forward(self, x):
        x = self.layer1_C1(x)       # C1:[32, 6, 28, 28]
        x = self.layer2_S2(x)       # S2:[32, 6, 14, 14]
        x = self.layer3_C3(x)       # C3:[32, 16, 10, 10]
        x = self.layer4_S4(x)       # S4:[32, 400]
        x = x.view(x.size(0),-1)
        x = self.layer5_C5(x)       # C5:[32, 120]
        x = self.layer6_F6(x)       # F6:[32, 84]
        out = self.layer7_OUTPUT(x) # OUTPUT:[32, 10]
        return out