import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import random
import os
import argparse
import numpy as np
import scipy.misc
from models import *
from utils import progress_bar
import shutil
import matplotlib.pyplot as plt


class MyDataset(torch.utils.data.Dataset):
    # 初始化一些需要传入的参数
    def __init__(self, datatxt, transform=None, target_transform=None):
        fh = open(datatxt, 'r')  # 按照传入的路径和txt文本参数，打开这个文本，并读取内容
        imgs = []  # 创建一个名为img的空列表，一会儿用来装东西
        for line in fh:  # 按行循环txt文本中的内容
            line = line.rstrip()  # 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
            words = line.split()  # 通过指定分隔符对字符串进行切片，默认为所有的空字符，包括空格、换行、制表符等
            imgs.append((words[0], int(words[1])))  # 把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定

        # 很显然，根据我刚才截图所示txt的内容，words[0]是图片信息，words[1]是lable
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
    def __getitem__(self, index):
        fn, label = self.imgs[index]  # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        img = Image.open(fn).convert('RGB')  # 按照path读入图片from PIL import Image # 按照路径读取图片
        if self.transform is not None:
            img = self.transform(img)  # 是否进行transform
        return img, label  # return很关键，return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)


def val_test():
    net.eval()
    correct = 0
    total = 0
    image_list = []  # 判断错误的图片在txt中对应的行(从0开始)

    for idx, (inputs, targets) in enumerate(valloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if predicted != targets:
            image_list.append(idx)
    val_ap = 100 * float(correct) / total
    return round(val_ap, 2), image_list


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def test():
    global BEST_ACC
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > BEST_ACC:
        print('Saving..')
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(net.state_dict(), 'checkpoint/model.pkl')
        BEST_ACC = acc


"""
生成一个与输入图像x类似的fooling image,但是这个image会被分类成target_y

Inputs:
- x: Input image
- target_y: An integer in the range [0, 10)
- model: A pretrained CNN

Returns:
- x_fooling: An image that is close to x, but that is classifed as target_y by the model.
"""


def make_fooling_image(x, target_y, model):
    # Initialize our fooling image to the input image, and wrap it in a Variable.
    x = x.cuda()
    x_fooling = x.clone()
    x_fooling_var = Variable(x_fooling, requires_grad=True)

    learning_rate = 0.8

    for i in range(100):
        scores = model(x_fooling_var).cuda()  # 前向操作
        _, index = scores.data.max(dim=1)
        if index[0] == target_y:  # 当成功fool的时候break.
            break
        target_score = scores[0, target_y]  # Score for the target class.
        target_score.backward()
        im_grad = x_fooling_var.grad.data  # 得到正确分数对应输入图像像素点的梯度

        # torch.norm(input, p=2) → float 返回输入张量input的p范数
        x_fooling_var.data += learning_rate * (im_grad / torch.norm(im_grad, 2))  # 通过正则化的梯度对所输入图像进行更新
        x_fooling_var.grad.data.zero_()  # 梯度更新清零，否则梯度会进行累积
    return x_fooling


# 得到txt文件中每一行数据末尾为0和1的分界线
def get_index(path):
    f = open(path, 'r')
    for num, value in enumerate(f):
        value = str(value).strip()
        if value[-1] == str(1):
            bound = num
            break
    f.close()
    return bound


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
args = parser.parse_args()

# Data
print('==> Preparing data..')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 生成的两个种类图片存放的目录
gen_class0 = '/home/gao/PycharmProjects/Adv_Improve/data/fooling_img/class0/'
gen_class1 = '/home/gao/PycharmProjects/Adv_Improve/data/fooling_img/class1/'
# 训练集第一类的标签目录
class0Label = 'dir/train0.txt'
# 训练集第二类的标签目录
class1Label = 'dir/train1.txt'
# 训练集标签目录
trainLabel = 'dir/train0_1.txt'
# 测试集的标签目录
testLabel = 'dir/test0_1.txt'
# 验证集的标签目录
valLabel = 'dir/val0_1.txt'
# 模型的数目
N = 14

for k in range(N):
    # 删除存放生成的图片的目录文件夹
    if os.path.exists(gen_class0):
        shutil.rmtree(gen_class0)  # 递归删除文件夹
    if os.path.exists(gen_class1):
        shutil.rmtree(gen_class1)  
        
    # 删除存放生成的图片的地址的txt文件
    if os.path.exists('dir/retrain.txt'):
        os.remove('dir/retrain.txt')

    os.makedirs('data/fooling_img/class0')
    os.makedirs('data/fooling_img/class1')

    # 创建存放生成的图片的地址的txt文件
    f = open('dir/retrain.txt', 'w')
    f.close()

    BEST_ACC = 0  # 最高的准确率

    ii = 1
    testAp = []
    valAp = []
    trainNum = []
    errNum = []

    for j in range(50):  # 生成50轮图片继续训练
        """
        数据分类,得到所需要的训练集 测试集 验证集数据
        根据训练集cat和dog所有图片生成新的训练集和验证集标签
        训练集中每一类的图片占80%,测试集为20%
        生成的新的训练集标签文件为train3_5,验证集文件为val3_5
        生成的其它文件为程序中的临时文件
        retrain文件为生成的对抗样本的标签文件
        """
        if os.path.exists(trainLabel):
            os.remove(trainLabel)
        if os.path.exists(valLabel):
            os.remove(valLabel)
        if os.path.exists('dir/temp.txt'):
            os.remove('dir/temp.txt')
        if os.path.exists('dir/train0r.txt'):
            os.remove('dir/train0r.txt')
        if os.path.exists('dir/train1r.txt'):
            os.remove('dir/train1r.txt')

        f0 = open(class0Label, 'r')
        f0_list = f0.readlines()
        fr = open('dir/retrain.txt', 'r')
        fr_list = fr.readlines()
        f1 = open(class1Label, 'r')
        f1_list = f1.readlines()
        f = open('dir/temp.txt', 'a')
        for i in range(len(f0_list)):
            f.writelines(f0_list[i])
        for i in range(len(fr_list)):
            f.writelines(fr_list[i])
        for i in range(len(f1_list)):
            f.writelines(f1_list[i])
        f.close()
        f0.close()
        f1.close()
        fr.close()
        temp_sort = ''.join(
            sorted(open('dir/temp.txt'), key=lambda s: s.split()[1], reverse=0))
        f = open('dir/temp.txt', 'w')
        f.write(temp_sort)
        f.close()
        f = open('dir/temp.txt', 'r')
        f_list = f.readlines()
        f.close()
        index = get_index('dir/temp.txt')
        print('0和1的分界线索引:' + str(index))
        f0r = open('dir/train0r.txt', 'a')
        f1r = open('dir/train1r.txt', 'a')
        for i in range(0, index):
            f0r.writelines(f_list[i])
        for i in range(index, len(f_list)):
            f1r.writelines(f_list[i])
        f0r.close()
        f1r.close()
        f0r = open('dir/train0r.txt', 'r')
        f0_list = f0r.readlines()
        f1r = open('dir/train1r.txt', 'r')
        f1_list = f1r.readlines()
        f0r.close()
        f1r.close()
        random.shuffle(f0_list)
        random.shuffle(f1_list)
        fv = open(valLabel, 'a')
        for i in range(int(0.2 * len(f0_list))):
            fv.writelines(f0_list[i])
        for i in range(int(0.2 * len(f1_list))):
            fv.writelines(f1_list[i])
        ft = open(trainLabel, 'a')
        for i in range(int(0.2 * len(f0_list)), len(f0_list)):
            ft.writelines(f0_list[i])
        for i in range(int(0.2 * len(f1_list)), len(f1_list)):
            ft.writelines(f1_list[i])
        fv.close()
        ft.close()
        f = open(trainLabel, 'r')
        f_list = f.readlines()
        f.close()
        trainNum.append(len(f_list))
        print('训练集数目:' + str(trainNum))
        trainset = MyDataset(trainLabel, transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
        testset = MyDataset(testLabel, transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=True, num_workers=2)
        valset = MyDataset(valLabel, transform)
        valloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=True, num_workers=2)
        classes = ('cat', 'dog')

        # Model
        print('==> Building model..')

        if ii == 1:  # 只有在ii = 1的时候才会选择一个模型训练并保存,否则加载已保存的模型
            if k == 0:
                net = VGG('VGG19')
            elif k == 1:
                net = ResNet18()
            elif k == 2:
                net = GoogLeNet()
            elif k == 3:
                net = DenseNet121()
            elif k == 4:
                net = ResNeXt29_2x64d()
            elif k == 5:
                net = MobileNet()
            elif k == 6:
                net = MobileNetV2()
            elif k == 7:
                net = DPN92()
            elif k == 9:
                net = ShuffleNetG2()
            elif k == 10:
                net = SENet18()
            elif k == 11:
                net = ShuffleNetV2(1)
            elif k == 12:
                net = EfficientNetB0()
            elif k == 13:
                net = PreActResNet18()
            net = net.cuda()
        else:  # 在ii不等于1的时候加载一个已有的模型
            net.load_state_dict(torch.load('checkpoint/model.pkl'))
            net.cuda()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

        BEST_ACC = 0
        # Training
        for epoch in range(0, 50):
            train(epoch)
            test()

        testAp.append(BEST_ACC)
        print('测试集准确率:' + str(testAp))

        val_ap, image_list = val_test()
        valAp.append(val_ap)
        print('验证集准确率:' + str(valAp))
        if os.path.exists('dir/error_img.txt'):
            os.remove('dir/error_img.txt')

        # 按找到的判断错误的图片从原txt里面提取图片信息到新的txt
        with open(valLabel, 'r') as f:  # 打开文件
            lines = f.readlines()  # 读取所有行
            ff = open('dir/error_img.txt', 'a')
            for i in range(len(image_list)):
                # 读取第image_list[i]行
                error_line = lines[image_list[i]]
                ff.write(str(error_line))
            ff.close()
        f.close()

        # 当生成的图片,或者说是error_img文本文件之中有两个类别的图片都有或者只有标签为1的图片时
        try:
            index = get_index('dir/error_img.txt')  # 只有标签为0的图片时会报错
            trainset = MyDataset('dir/error_img.txt', transform)
            f = open('dir/error_img.txt', 'r')
            raw_list = f.readlines()
            f.close()
            errNum.append(len(raw_list))
            print('错误图片数量:' + str(errNum))
            for i in range(0, index):
                x = trainset[i][0]  # 输入数据
                x = torch.unsqueeze(x, 0)  # 给数据增加一个维度
                y = trainset[i][1]  # 输入数据的标签
                y = np.array([y])  # Convert y from Torch Tensors to numpy arrays
                target_y = 0  # fooling image的目标类型
                x_fooling = make_fooling_image(x, target_y, net)

                scores = net(x_fooling)
                if target_y == scores.data.max(1)[1][0].item():
                    # 按行取最大值并返回索引
                    x_fooling = x_fooling.squeeze(0)  # 去除某个维度
                    x_fooling_np = x_fooling.cpu().numpy()
                    x_fooling_np = np.transpose(x_fooling_np, (1, 2, 0))
                    img_name = str(ii) + '_' + str(i) + '.png'
                    path = os.path.join(gen_class0, img_name)
                    scipy.misc.imsave(path, x_fooling_np)
            for i in range(index, len(raw_list)):
                x = trainset[i][0]  # 输入数据
                x = torch.unsqueeze(x, 0)  # 给数据增加一个维度
                y = trainset[i][1]  # 输入数据的标签
                y = np.array([y])  # Convert y from Torch Tensors to numpy arrays
                target_y = 1  # fooling image的目标类型
                x_fooling = make_fooling_image(x, target_y, net)
                scores = net(x_fooling)
                if target_y == scores.data.max(1)[1][0].item():
                    # 按行取最大值并返回索引
                    x_fooling = x_fooling.squeeze(0)  # 去除某个维度
                    x_fooling_np = x_fooling.cpu().numpy()
                    x_fooling_np = np.transpose(x_fooling_np, (1, 2, 0))
                    img_name = str(ii) + '_' + str(i) + '.png'
                    path = os.path.join(gen_class1, img_name)
                    scipy.misc.imsave(path, x_fooling_np)

        # 当生成的图片,或者说是error_img文本文件之中只有标签为0的图片时
        except UnboundLocalError:
            trainset = MyDataset('dir/error_img.txt', transform)
            f = open('dir/error_img.txt', 'r')
            raw_list = f.readlines()
            f.close()
            errNum.append(len(raw_list))
            for i in range(len(raw_list)):
                x = trainset[i][0]  # 输入数据
                x = torch.unsqueeze(x, 0)  # 给数据增加一个维度
                y = trainset[i][1]  # 输入数据的标签
                y = np.array([y])  # Convert y from Torch Tensors to numpy arrays
                target_y = 0  # fooling image的目标类型
                x_fooling = make_fooling_image(x, target_y, net)
                scores = net(x_fooling)
                if target_y == scores.data.max(1)[1][0].item():
                    # 按行取最大值并返回索引
                    x_fooling = x_fooling.squeeze(0)  # 去除某个维度
                    x_fooling_np = x_fooling.cpu().numpy()
                    x_fooling_np = np.transpose(x_fooling_np, (1, 2, 0))
                    img_name = str(ii) + '_' + str(i) + '.png'
                    path = os.path.join(gen_class0, img_name)
                    scipy.misc.imsave(path, x_fooling_np)

        ii = ii + 1

        # 建立列表，用于保存图片信息
        file_list = []
        # 读取图片文件，并将图片地址、图片名和标签写到txt文件中
        write_file_name = 'dir/retrain.txt'
        write_file = open(write_file_name, 'w')  # 以只写方式打开write_file_name文件
        for file in os.listdir(gen_class0):  # file为current_dir当前目录下图片名
            if file.endswith('.png'):  # 如果file以png结尾
                write_name = file + ' ' + str(0)
            file_list.append(write_name)  # 将write_name添加到file_list列表最后
            number_of_lines = len(file_list)  # 列表中元素个数
        # 将图片信息写入txt文件中，逐行写入
        for current_line in range(number_of_lines):
            write_file.write(gen_class0 + file_list[current_line] + '\n')
        # 关闭文件
        write_file.close()  # 存放原始图片地址

        # 建立列表，用于保存图片信息
        file_list = []
        # 读取图片文件，并将图片地址、图片名和标签写到txt文件中
        write_file_name = 'dir/retrain.txt'
        write_file = open(write_file_name, 'a')  # 以只写方式打开write_file_name文件
        for file in os.listdir(gen_class1):  # file为current_dir当前目录下图片名
            if file.endswith('.png'):  # 如果file以png结尾
                write_name = file + ' ' + str(1)
            file_list.append(write_name)  # 将write_name添加到file_list列表最后
            number_of_lines = len(file_list)  # 列表中元素个数
        # 将图片信息写入txt文件中，逐行写入
        for current_line in range(number_of_lines):
            write_file.write(gen_class1 + file_list[current_line] + '\n')
        # 关闭文件
        write_file.close()
        
    if not os.path.exists('results/'):
        os.makedirs('results/')        
    
    plt.plot(testAp)
    plt.savefig('results/testAp' + str(k) + '.png')
    plt.close()
    plt.plot(valAp)
    plt.savefig('results/valAp' + str(k) + '.png')
    plt.close()
    plt.plot(trainNum)
    plt.savefig('results/trainNum' + str(k) + '.png')
    plt.close()
    plt.plot(errNum)
    plt.savefig('results/errNum' + str(k) + '.png')
    plt.close()
