import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset

def getDataLoder(DATA,rate):
    from torch.utils.data import DataLoader,random_split
    trainSize = int(rate * len(DATA))
    testSize = len(DATA) - trainSize
    trainDATA, testDATA = random_split(DATA, [trainSize, testSize])
    trainDataLoader = DataLoader(trainDATA, batch_size=16, shuffle=True)
    testDataLoader = DataLoader(testDATA, batch_size=1, shuffle=True)

    return trainDataLoader,testDataLoader,(trainSize,testSize)

####加载MNIST数据####
PATH_Mnist = '../DATA/DATA-mnist/'
def loadMnist(path, kind='t10k'):
    import struct
    # 将字节数据加载至Numpy
    labels_path = os.path.join(path,'%s-labels.idx1-ubyte'% kind)
    images_path = os.path.join(path,'%s-images.idx3-ubyte'% kind)

    with open(labels_path, 'rb') as lbPath:
        magic, n = struct.unpack('>II',lbPath.read(8))
        labels = np.fromfile(lbPath,dtype=np.uint8)

    with open(images_path, 'rb') as imgPath:
        magic, num, rows, cols = struct.unpack('>IIII',imgPath.read(16))
        images = np.fromfile(imgPath,dtype=np.uint8).reshape(len(labels), 784)

    return images, labels
class DatasetsMnist(Dataset):
    def __init__(self,dataX,dataY):
        self.dataX = dataX
        self.dataY = dataY
        self.tfs = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize([0.485],[0.229])]
        )

    def __len__(self):
        return len(self.dataX)

    def __getitem__(self, idx):
        x = self.dataX[idx]
        x = x.reshape(28,28)
        # x = cv2.resize(x,(32,32))
        x = self.tfs(x)
        x = np.array(x)
        x = torch.from_numpy(x).float()

        y = self.dataY[idx]
        y = np.array(y)
        y = torch.from_numpy(y).long()
        return x,y

images_Mnist,labels_Mnist = loadMnist(PATH_Mnist)
DATA_Mnist = DatasetsMnist(images_Mnist,labels_Mnist)
trainDataLoader_Mnist,testDataLoader_Mnist,size_Mnist = getDataLoder(DATA_Mnist,0.7)


####加载Cifar10数据####
PATH_Cifar10 = '../DATA/DATA-cifar10/test_batch'
def loadCifar10(file):
    # 将字节数据加载至Numpy
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    images = []
    labels = []
    for i in range(10000):
        img = np.reshape(dict[b'data'][i],(3,32,32))
        img = img.transpose(1, 2, 0)
        images.append(img)
        lab = int(dict[b'labels'][i])
        labels.append(lab)

    return np.array(images), np.array(labels)
class DatasetsCifar10(Dataset):
    def __init__(self,dataX,dataY):
        self.dataX = dataX
        self.dataY = dataY
        self.tfs = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )

    def __len__(self):
        return len(self.dataX)

    def __getitem__(self, idx):
        x = self.dataX[idx]
        x = self.tfs(x)
        x = np.array(x)
        x = torch.from_numpy(x).float()

        y = self.dataY[idx]
        y = np.array(y)
        y = torch.from_numpy(y).long()
        return x,y

images_Cifar10,labels_Cifar10 = loadCifar10(PATH_Cifar10)
DATA_Cifar10 = DatasetsCifar10(images_Cifar10,labels_Cifar10)
trainDataLoader_Cifar10,testDataLoader_Cifar10,size_Cifar10 = getDataLoder(DATA_Cifar10,0.7)


####加载catVSdog数据####
PATH_CatVsDog = '../DATA/DATA-CatVsDog/test_set/'
def loadCatVsDog(PATH):
    catsDir = PATH+'cats/'
    dogsDir = PATH+'dogs/'

    cats = os.listdir(catsDir)
    cats = [catsDir+x for x in cats]
    dogs = os.listdir(dogsDir)
    dogs = [dogsDir+x for x in dogs]

    catsLabels = [0]*len(cats)
    dogsLabels = [1]*len(dogs)

    images = cats + dogs
    labels = catsLabels + dogsLabels

    return np.array(images), np.array(labels)
class DatasetsCatVsDog(Dataset):
    def __init__(self, dataX, dataY):
        self.dataX = dataX
        self.dataY = dataY
        self.tfs = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )

    def __len__(self):
        return 20

    def __getitem__(self, idx):
        x = self.dataX[idx]
        x = cv2.imread(x)
        x = cv2.resize(x, (224, 224))
        x = self.tfs(x)
        x = np.array(x)
        x = torch.from_numpy(x).float()

        y = self.dataY[idx]
        y = np.array(y)
        y = torch.from_numpy(y).long()
        # y = y.unsqueeze(dim=0)
        return x, y

images_CatVsDog,labels_CatVsDog = loadCatVsDog(PATH_CatVsDog)
DATA_CatVsDog = DatasetsCatVsDog(images_CatVsDog,labels_CatVsDog)
trainDataLoader_CatVsDog,testDataLoader_CatVsDog,size_CatVsDog = getDataLoder(DATA_CatVsDog,0.7)
