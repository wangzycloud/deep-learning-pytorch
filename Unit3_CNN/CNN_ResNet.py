import torch.nn as nn

def conv1x1(in_channels,out_channels,stride=1):
    return nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride)
def conv3x3(in_channels,out_channels,stride=1):
    # 3x3 convolution with padding
    return nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1)

class BasicBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,downSample=None):
        super(BasicBlock,self).__init__()
        self.downSampleSelf = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1),
            nn.BatchNorm2d(out_channels)
        )

        self.conv1 = conv3x3(in_channels,out_channels,stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = conv3x3(out_channels,out_channels,stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downSample = downSample if downSample else self.downSampleSelf
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downSample is not None:
            residual = self.downSample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    # 输出维度是输入维度的四倍，这里out_channels决定不了输出维度。
    # 输出维度由expansion决定。
    expansion = 4
    def __init__(self,in_channels,out_channels,stride=1,downSample=None):
        super(Bottleneck,self).__init__()
        self.downSampleSelf = nn.Sequential(
            nn.Conv2d(in_channels, in_channels*self.expansion, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_channels*self.expansion)
        )

        self.conv1 = conv1x1(in_channels,out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = conv3x3(out_channels,out_channels,stride)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = conv1x1(out_channels,in_channels*self.expansion)
        self.bn3 = nn.BatchNorm2d(in_channels*self.expansion)

        self.relu = nn.ReLU()
        self.downSample = downSample if downSample else self.downSampleSelf
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downSample is not None:
            identity = self.downSample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downSample = None
        if stride != 1 or self.in_channels != planes * block.expansion:
            downSample = nn.Sequential(
                conv1x1(self.in_channels, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, planes, stride, downSample))
        self.in_channels = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxPool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgPool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x