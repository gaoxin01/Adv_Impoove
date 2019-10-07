import os

# cifar10数据库的标签
cifar = "dir/cifar.txt"
# 训练集的标签
Train = 'dir/Train.txt'
# 测试集的标签
Test = 'dir/Test.txt'
# 按标签顺序重新排序
train_sort = 'dir/trainsort.txt'
test_sort = 'dir/testsort.txt'
# 训练集第一类的标签目录
class0Label = 'dir/train0.txt'
# 训练集第二类的标签目录
class1Label = 'dir/train1.txt'
# 测试集的标签目录
testLabel = 'dir/test0_1.txt'

# 如果这些文件存在先删除
if os.path.exists(Train):
    os.remove(Train)
if os.path.exists(Test):
    os.remove(Test)
if os.path.exists(train_sort):
    os.remove(train_sort)
if os.path.exists(test_sort):
    os.remove(test_sort)
if os.path.exists(class0Label):
    os.remove(class0Label)
if os.path.exists(class1Label):
    os.remove(class1Label)
if os.path.exists(testLabel):
    os.remove(testLabel)

# 将cifar10标签60000标签分成训练集和测试集两个txt文件
fc = open(cifar, 'r')
fc_list = fc.readlines()
fc.close()
f = open(Train, 'a')
for i in range(0, 50000):
    f.writelines(fc_list[i])
f.close()
f = open(Test, 'a')
for i in range(50000, len(fc_list)):
    f.writelines(fc_list[i])
f.close()

# 将生成的训练集和测试集的标签按顺序排序(从大到小)
f = open(train_sort, 'w')
trainSort = ''.join(sorted(open(Train), key=lambda s: s.split()[1], reverse=1))
f.write(trainSort)
f.close()
f = open(test_sort, 'w')
testSort = ''.join(sorted(open(Test), key=lambda s: s.split()[1], reverse=1))
f.write(testSort)
f.close()

# 读取cifar10中训练集标签所有数据
f = open(train_sort, 'r')
f_list = f.readlines()
f.close()
# 得到cifar10中训练集需要的dog类
f = open(class1Label, 'w')
for i in range(20000, 25000):
    f.writelines(f_list[i])
f.close()
# 得到cifar10中训练集需要的cat类
f = open(class0Label, 'w')
for i in range(30000, 35000):
    f.writelines(f_list[i])
f.close()
# 读取cifar10中测试集标签所有数据
f = open(test_sort, 'r')
f_list = f.readlines()
f.close()
# 得到cifar10中测试集需要的dog类
f = open(testLabel, 'a')
for i in range(4000, 5000):
    f.writelines(f_list[i])
f.close()
# 得到cifar10中测试集需要的cat类
f = open(testLabel, 'a')
for i in range(6000, 7000):
    f.writelines(f_list[i])
f.close()

# 将train0.txt中的标签3替换为0
f = open(class0Label, "r")
content = f.read()
content = content.replace(".png 3", ".png 0")
f.close()
f = open(class0Label, 'w')
f.write(content)
f.close()
# 将train1.txt中的标签5替换为1
f = open(class1Label, "r")
content = f.read()
content = content.replace(".png 5", ".png 1")
f.close()
f = open(class1Label, 'w')
f.write(content)
f.close()
# 将test0_1.txt中的标签3替换为0,标签5替换为0
f = open(testLabel, "r")
content = f.read()
content = content.replace(".png 3", ".png 0")
f.close()
# 覆盖原来的内容
f = open(testLabel, 'w')
f.write(content)
f.close()
f = open(testLabel, "r")
content = f.read()
content = content.replace(".png 5", ".png 1")
f.close()
f = open(testLabel, 'w')
f.write(content)
f.close()