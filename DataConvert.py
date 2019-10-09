import pickle as p
import numpy as np
from PIL import Image
import os
import shutil

# 加载数据
def load_CIFAR_batch(filename):
    with open(filename, 'rb') as f:
        datadict = p.load(f, encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32)
        Y = np.array(Y)
        return X, Y

def getLabelType(typeIndex, label):
    index = label[typeIndex]
    if index == 0:
        return "airplane"
    if index == 1:
        return "automobile"
    if index == 2:
        return "bird"
    if index == 3:
        return "cat"
    if index == 4:
        return "deer"
    if index == 5:
        return "dog"
    if index == 6:
        return "frog"
    if index == 7:
        return "horse"
    if index == 8:
        return "ship"
    if index == 9:
        return "truck"


# 创建每个batch图片保存的文件夹
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("创建文件夹" + path)
        return 1
    else:
        return 0


# 保存图片和标签
def saveImgAndLabel(tag):
    saveDir = image + "batch_images_" + str(tag)
    print(saveDir)
    if mkdir(saveDir) == 0:
        return
    if tag == "test":
        imgX, labelX = load_CIFAR_batch(root + "test_batch")  # 加载测试数据集
    else:
        imgX, labelX = load_CIFAR_batch(root + "data_batch_" + str(tag))  # 加载训练数据集
    for i in range(0, 10000):
        imgs = imgX[i]
        img0 = imgs[0]
        img1 = imgs[1]
        img2 = imgs[2]
        # 生成image对象RGB
        i0 = Image.fromarray(img0)
        i1 = Image.fromarray(img1)
        i2 = Image.fromarray(img2)
        img = Image.merge("RGB", (i0, i1, i2))
        # 这里是生成英文标签的名字
        label = getLabelType(i, labelX)
        name = str(i) + '_' + label + '.png'
        img.save(saveDir + "/" + name, "png")
        f = open(labelTxt,'a')   # 读取label.txt文件，没有则创建，‘a’表示再次写入时不覆盖之前的内容
        f.write(str(saveDir) + "/" + str(name) + ' ' + str(labelX[i]))
        f.write('\n')               # 实现换行的功能

# 程序入口
if __name__ == "__main__":
    # cifar10数据库存放地址
    root = "/home/gao/PycharmProjects/Adv_Improve/data/cifar-10-batches-py/"
    # 转换生成的图片存放的地址
    image = "/home/gao/PycharmProjects/Adv_Improve/data/image/"
    # 转换生成的标签文件
    labelTxt = 'dir/cifar.txt'
    # 如果这些文件存在先删除
    if not os.path.exists('dir/'):
        os.makedirs('dir/')
    if os.path.exists(image):
        shutil.rmtree(image)
    if os.path.exists(labelTxt):
        os.remove(labelTxt)
    for i in range(1, 7):
        if i == 6:
            saveImgAndLabel('test')
        else:
            saveImgAndLabel(i)
    print("保存完毕.")
