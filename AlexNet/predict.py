import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import AlexNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # load image
    img_path = "./yjx.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)  # 使用PIL库加载图片

    plt.imshow(img)  # 展示图片
    # [N, C, H, W]
    img = data_transform(img)   # 对图片进行预处理
    # expand batch dimension（扩充batch维度）
    img = torch.unsqueeze(img, dim=0)   # 给图片扩充维度，原本读取的图片只有height,width,channel

    # read class_indict（读取json文件，也就是 索引对应类别 的那个文件）
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)     # 加载成我们需要的字典

    # create model（初始化网络）
    model = AlexNet(num_classes=5).to(device)

    # load model weights（载入网络模型参数）
    weights_path = "./AlexNet.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path))

    model.eval()    # 进入eval模式，也就是关闭掉dropout(train是打开dropout)
    with torch.no_grad():   # 不计算损失梯度
        # predict class
        output = torch.squeeze(model(img.to(device)))   # 将图片交给cpu设备处理，然后传入model，然后使用squeeze压缩维度,去掉batch得到输出
        predict = torch.softmax(output, dim=0)          # 使用softmax处理，变成概率分布
        predict_cla = torch.argmax(predict).numpy()     # 使用argmax获取概率最大处所对应的索引值

    # 打印类别名称以及对应概率
    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)], predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)], predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    main()
