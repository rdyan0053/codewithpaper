import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


def main():
    # 图像预处理（Compose是一个类，将两个预处理方法打包）
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 50000张训练图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=36,
                                               shuffle=True, num_workers=0)

    # 10000张验证图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=4,
                                              shuffle=False, num_workers=0)

    # 这两行代码没看懂
    test_data_iter = iter(test_loader)
    test_image, test_label = test_data_iter.next()  # 调用该迭代器上的__next__()方法以获取第一次迭代。再次运行next()将获得迭代器的第二个条目

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 简单展示一下数据集中的图片
    # # print labels
    # print(' '.join('%5s' % classes[test_label[j]] for j in range(4)))
    # # show images
    # imshow(torchvision.utils.make_grid(test_image))  # make_grid制作图像网格

    net = LeNet()  # 实例化模型
    loss_function = nn.CrossEntropyLoss()  # 定义损失函数
    optimizer = optim.Adam(net.parameters(), lr=0.001)  # 定义优化器，parameters是网络需要训练的参数，lr是学习率learning rate

    # 具体训练过程
    for epoch in range(5):  # loop over the dataset multiple times

        running_loss = 0.0  # 累加训练过程中的损失
        # enumerate:枚举, 通过enumerate将可迭代对象train_loader返回一个tuple,其中包含一个计数(这里是step),另一个就是获取的值data
        for step, data in enumerate(train_loader, start=0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data  # 将data赋值给inputs和labels

            # zero the parameter gradients(每计算一个batch,就要调用一次optimizer.zero_grad()清除历史梯度,如果不去除就会对历史梯度进行累加)
            optimizer.zero_grad()  # 解释链接https://www.zhihu.com/question/303070254
            # forward + backward + optimize
            outputs = net(inputs)  # 将数据输入到模型,得到预测结果
            loss = loss_function(outputs, labels)  # outputs是网络预测的值,labels是真实值
            loss.backward()  # 将loss进行反向传播
            optimizer.step()  # 通过优化器的step函数进行参数的更新

            # print statistics(打印过程)
            running_loss += loss.item()  # 累加损失， 因为我们希望每500次迭代计算一个损失
            if step % 500 == 499:  # print every 500 mini-batches,每训练500次之后就会进入这里验证
                with torch.no_grad():  # with是一个上下文管理器,这里禁用的计算是下面模型推理步骤的梯度，也就是模型预测步骤的梯度，此时模型实例化，就会生成一张计算图，而这种资源的浪费，是没必要的
                    outputs = net(test_image)  # [batch, 10]
                    predict_y = torch.max(outputs, dim=1)[1]  # dim=1代表取outputs的第2个维度,后面的1是索引(max返回两个值0是最大值 1是位置)
                    # 判断预测值和真实值是否相符,然后求和,求和之后转换为int,再除以验证样本的数目,得到准确率
                    accuracy = torch.eq(predict_y, test_label).sum().item() / test_label.size(0)

                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                          (epoch + 1, step + 1, running_loss / 500, accuracy))
                    running_loss = 0.0

    print('Finished Training')

    save_path = './Lenet.pth'
    torch.save(net.state_dict(), save_path)


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    # Use transpose(a, argsort(axes)) to invert the transposition of tensors when using the axes keyword argument.
    # 使用axes关键字参数时反转张量的转置
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    main()
