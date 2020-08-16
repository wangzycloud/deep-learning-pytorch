import os
import cv2
import numpy as np

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

testXtr = unpickle("./test_batch")
print(testXtr.keys())

for i in range(0, 200):
    dir = './cifar/'+ str(testXtr[b'labels'][i])
    isExists = os.path.exists(dir)
    if not isExists:
        os.mkdir(dir)
    # 读取图片文件名及数据
    imgName = str(testXtr[b'filenames'][i], encoding='utf8')
    img = np.reshape(testXtr[b'data'][i], (3, 32, 32))
    img = img.transpose(1,2,0)

    cv2.imwrite(dir+'/'+imgName,img)
