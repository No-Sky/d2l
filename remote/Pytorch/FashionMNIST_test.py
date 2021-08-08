import torch
from torch import nn
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
import utils

d2l.use_svg_display()

"""
Fashion-MNIST中包含的10个类别分别为
t-shirt（T恤）、trouser（裤子）、pullover（套衫）、
dress（连衣裙）、coat（外套）、sandal（凉鞋）、shirt（衬衫）、
sneaker（运动鞋）、bag（包）和ankle boot（短靴）。
以下函数用于在数字标签索引及其文本名称之间进行转换
"""

# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式
# 并除以255使得所有像素的数值均在0到1之间
# trans = transforms.ToTensor()
# mnist_train = torchvision.datasets.FashionMNIST(root="./data", train=True,
#                                                 transform=trans,
#                                                 download=True)
# mnist_test = torchvision.datasets.FashionMNIST(root="./data", train=False,
# transform=trans, download=True)

"""
Fashion-MNIST中包含的10个类别分别为
t-shirt（T恤）、trouser（裤子）、pullover（套衫）、
dress（连衣裙）、coat（外套）、sandal（凉鞋）、shirt（衬衫）、
sneaker（运动鞋）、bag（包）和ankle boot（短靴）。
以下函数用于在数字标签索引及其文本名称之间进行转换
"""


def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签。"""
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt',
        'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  # @save
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


# 测试数据读取
# X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
# show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))


# 小批量数据读取
# batch_size = 256


def get_dataloader_workers():  # @save
    """使用4个进程来读取数据。"""
    return 4


# train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
#                              num_workers=get_dataloader_workers())

# 小批量数据读取测试
# timer = d2l.Timer()
# for X, y in train_iter:
#     continue
# print(f'{timer.stop():.2f} sec')

# 测试整合函数，并通过resize测试调整图像大小
# train_iter, test_iter = utils.load_data_fashion_mnist(32, resize=64)
# for X, y in train_iter:
#     print(X.shape, X.dtype, y.shape, y.dtype)
#     break