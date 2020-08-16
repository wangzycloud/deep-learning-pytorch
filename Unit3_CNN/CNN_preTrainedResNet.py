import torch.nn as nn
from torchvision import models

class preTrainedResNet(nn.Module):

    preTrained_net = models.resnet18(pretrained=True)
    for param in preTrained_net.parameters():
        param.requires_grad = True

    def __init__(self,n_class):
        super(preTrainedResNet,self).__init__()
        self.preNet = self.preTrained_net

        self.init = nn.Sequential(*list(self.preNet.children())[0:3])
        # 1/2 (去掉了原有最大池化层)
        self.layer1 = list(self.preNet.children())[4]
        self.layer2 = list(self.preNet.children())[5]
        self.layer3 = list(self.preNet.children())[6]
        self.layer4 = list(self.preNet.children())[7]
        self.adjust = nn.Sequential(
            nn.Conv2d(512,256,kernel_size=1,stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.avgPool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 2 * 2, 100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 10),
            nn.ReLU(),
            nn.Linear(10, n_class)
        )

    def forward(self, x):
        init = self.init(x)
        enc1 = self.layer1(init)
        enc2 = self.layer2(enc1)
        enc3 = self.layer3(enc2)
        enc4 = self.layer4(enc3)

        out = self.adjust(enc4)
        out = self.avgPool(out)

        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return out