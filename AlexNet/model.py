import torch.nn as nn
import torch


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        # 提取图像特征部分
        self.features = nn.Sequential(  # Sequential将一系列的层结构打包成新的结构，这里取名为features,代表专门提取图像特征的结构
            # 卷积核个数out_channels不是原网络的98，而是48以减少运算，彩色图片in_channels为3
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=11, stride=4, padding=2),
            # 相比之前每创建一个层结构都要"self.模块名称"要方便
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=48, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        # 分类部分，即网络最后的全连接层
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),  # Dropout一般是放在全连接层之间
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        # 这里只是讲解下自己写的时候初始化权重的方法，如果不初始化，默认会使用何恺明初始化
        if init_weights:
            self._initialize_weights()

    # 定义正向传播过程，forward函数，x代表输入图像
    def forward(self, x):
        x = self.features(x)  # 先对x进行特征提取
        # 得到输出的特征图之后进行展平处理，start_dim的含义是从第一个维度展平（四个维度，分别是batch,channel,width,height）
        # 第0个维度，即batch不用动；所以这里是从channel这个维度进行展平
        x = torch.flatten(x, start_dim=1)  # 也开始使用view函数进行展平
        x = self.classifier(x)  # 然后是全连接层部分（分类）
        return x

    # 初始化权重的方法
    def _initialize_weights(self):
        for m in self.modules():  # 遍历每一层网络（modules的作用：Returns an iterator over all modules in the network）
            if isinstance(m, nn.Conv2d):  # 判断层结构是否为卷积层
                # 如果是卷积层，则使用kaiming_normal_方法来初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    # 如果该层的偏置不为None，则初始化为0
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 如果是全连接层，则使用正态分布（均值为0，方差为0.01）初始化，同时初始化偏置为0
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
