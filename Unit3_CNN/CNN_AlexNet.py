import torch.nn as nn
class CNN_AlexNet(nn.Module):
    def __init__(self,n_class):
        super(CNN_AlexNet,self).__init__()
        self.layer1_C1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=11,stride=4,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        self.layer1_C2 = nn.Sequential(
            nn.Conv2d(64,192,kernel_size=5,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        self.layer1_C3 = nn.Sequential(
            nn.Conv2d(192,384,kernel_size=3,stride=1,padding=1),
            nn.ReLU()
        )
        self.layer1_C4 = nn.Sequential(
            nn.Conv2d(384,256,kernel_size=3,stride=1,padding=1),
            nn.ReLU()
        )
        self.layer1_C5 = nn.Sequential(
            nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256*6*6,100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100,10),
            nn.ReLU(),
            nn.Linear(10,n_class)
        )
    def forward(self, x):
        x = self.layer1_C1(x)
        x = self.layer1_C2(x)
        x = self.layer1_C3(x)
        x = self.layer1_C4(x)
        x = self.layer1_C5(x)
        x = x.view(x.size(0),256*6*6)
        out = self.classifier(x)
        return out