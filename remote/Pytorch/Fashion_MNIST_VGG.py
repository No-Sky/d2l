import torch
from torch import nn
from d2l import torch as d2l
import utils


def vgg_block(num_convs, in_channels, out_channels):
    """
    返回一个VGG块的网络序列
    :param num_convs: 卷积层的数量
    :param in_channels: 输入通道的数量
    :param out_channels: 输出通道的数量
    :return:
    """
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)

def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # 全连接层部分
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))

"""
原始 VGG 网络有 5 个卷积块，其中前两个块各有一个卷积层，
后三个块各包含两个卷积层。 第一个模块有 64 个输出通道，
每个后续模块将输出通道数量翻倍，直到该数字达到 512。
由于该网络使用 8 个卷积层和 3 个全连接层，因此它通常被称为 VGG-11
"""
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
net = vgg(conv_arch)

lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = utils.load_data_fashion_mnist(batch_size, resize=224)
utils.train(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())