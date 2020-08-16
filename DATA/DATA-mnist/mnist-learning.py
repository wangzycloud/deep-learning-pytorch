import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

PATH = './'
PATH_SAVE = './mnist/'
def load_MNIST(path, kind='t10k'):
    # 将字节数据加载至Numpy
    import struct
    labels_path = os.path.join(path,'%s-labels.idx1-ubyte'% kind)
    images_path = os.path.join(path,'%s-images.idx3-ubyte'% kind)

    with open(labels_path, 'rb') as lbPath:
        magic, n = struct.unpack('>II',lbPath.read(8))
        labels = np.fromfile(lbPath,dtype=np.uint8)

    with open(images_path, 'rb') as imgPath:
        magic, num, rows, cols = struct.unpack('>IIII',imgPath.read(16))
        images = np.fromfile(imgPath,dtype=np.uint8).reshape(len(labels), 784)

    return images, labels
img,lab = load_MNIST(PATH)

def visualizing1(img,lab):
    # 可视化1：展示10张不同数字的图片（0-10）
    fig, ax = plt.subplots(nrows=2,ncols=5)
    ax = ax.flatten()
    for i in range(10):
        t = img[lab == i][0].reshape(28, 28)
        ax[i].imshow(t, cmap='Greys', interpolation='nearest')

    plt.tight_layout()
    plt.show()
#visualizing1(img,lab)

def visualizing2(img,lab):
    # 可视化2：同一数字，展示10张图片
    fig, ax = plt.subplots(nrows=2,ncols=5)
    ax = ax.flatten()
    for i in range(10):
        t = img[lab == 7][i].reshape(28, 28)
        ax[i].imshow(t, cmap='Greys', interpolation='nearest')

    plt.tight_layout()
    plt.show()
#visualizing2(img,lab)

def MNIST_save(img,lab):
    # 将Numpy数据保存为bmp图片，以标签为文件夹
    for i,item in enumerate(lab):
        dir = PATH_SAVE+str(item)
        isExists = os.path.exists(dir)
        if not isExists:
            os.mkdir(dir)
        cv2.imwrite(dir+'/'+ str(i) +'.bmp',img[i].reshape(28,28))
#MNIST_save(img,lab)