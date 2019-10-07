import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import os
import argparse
from models import *
from utils import progress_bar


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

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
args = parser.parse_args()

# Data
print('==> Preparing data..')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 训练集第一类的标签目录
class0Label = 'dir/train0.txt'
# 训练集第二类的标签目录
class1Label = 'dir/train1.txt'
# 训练集标签目录
trainLabel = 'dir/train0_1.txt'
# 测试集的标签目录
testLabel = 'dir/test0_1.txt'

# 模型的数目
N = 14

for k in range(N):

    BEST_ACC = 0  # 最高的准确率

    testAp = []

    if os.path.exists(trainLabel):
        os.remove(trainLabel)

    f0 = open(class0Label, 'r')
    f0_list = f0.readlines()
    f1 = open(class1Label, 'r')
    f1_list = f1.readlines()
    f0.close()
    f1.close()
    ft = open(trainLabel, 'a')
    for i in range(0, len(f0_list)):
        ft.writelines(f0_list[i])
    for i in range(0, len(f1_list)):
        ft.writelines(f1_list[i])
    ft.close()
    trainset = MyDataset(trainLabel, transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = MyDataset(testLabel, transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    classes = ('cat', 'dog')

    # Model
    print('==> Building model..')

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

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    BEST_ACC = 0
    # Training
    for epoch in range(0, 200):
        train(epoch)
        test()

    testAp.append(BEST_ACC)
    print('测试集准确率:' + str(testAp))