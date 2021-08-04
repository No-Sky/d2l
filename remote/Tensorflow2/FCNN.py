import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

train_path = './mnist_image_label/mnist_train_jpg_60000/'
train_txt = './mnist_image_label/mnist_train_jpg_60000.txt'
x_train_savepath = './mnist_image_label/mnist_x_train.npy'
y_train_savepath = './mnist_image_label/mnist_y_train.npy'

test_path = './mnist_image_label/mnist_test_jpg_10000/'
test_txt = './mnist_image_label/mnist_test_jpg_10000.txt'
x_test_savepath = './mnist_image_label/mnist_x_test.npy'
y_test_savepath = './mnist_image_label/mnist_y_test.npy'


def generateds(path, txt):
    f = open(txt, 'r')  # 以只读形式打开txt文件
    contents = f.readlines()  # 读取文件中所有行
    f.close()  # 关闭txt文件
    x, y_ = [], []  # 建立空列表
    for content in contents:  # 逐行取出
        value = content.split()  # 以空格分开，图片路径为value[0] , 标签为value[1] , 存入列表
        img_path = path + value[0]  # 拼出图片路径和文件名
        img = Image.open(img_path)  # 读入图片
        img = np.array(img.convert('L'))  # 图片变为8位宽灰度值的np.array格式
        img = img / 255.  # 数据归一化 （实现预处理）
        x.append(img)  # 归一化后的数据，贴到列表x
        y_.append(value[1])  # 标签贴到列表y_
        print('loading : ' + content)  # 打印状态提示

    x = np.array(x)  # 变为np.array格式
    y_ = np.array(y_)  # 变为np.array格式
    y_ = y_.astype(np.int64)  # 变为64位整型
    return x, y_  # 返回输入特征x，返回标签y_

if os.path.exists(x_train_savepath) and os.path.exists(y_train_savepath) and os.path.exists(
        x_test_savepath) and os.path.exists(y_test_savepath):
    print('-------------Load Datasets-----------------')
    x_train_save = np.load(x_train_savepath)
    y_train = np.load(y_train_savepath)
    x_test_save = np.load(x_test_savepath)
    y_test = np.load(y_test_savepath)
    x_train = np.reshape(x_train_save, (len(x_train_save), 28, 28))
    x_test = np.reshape(x_test_save, (len(x_test_save), 28, 28))
else:
    print('-------------Generate Datasets-----------------')
    x_train, y_train = generateds(train_path, train_txt)
    x_test, y_test = generateds(test_path, test_txt)

    print('-------------Save Datasets-----------------')
    x_train_save = np.reshape(x_train, (len(x_train), -1))
    x_test_save = np.reshape(x_test, (len(x_test), -1))
    np.save(x_train_savepath, x_train_save)
    np.save(y_train_savepath, y_train)
    np.save(x_test_savepath, x_test_save)
    np.save(y_test_savepath, y_test)

# # 数据增强函数设置
# image_gen_train = ImageDataGenerator(
#     rescale=1. / 255,
#     rotation_range=45,
#     width_shift_range=.15,
#     height_shift_range=.15,
#     horizontal_flip=False,
#     zoom_range=0.5
# )
# # 数据增强操作
# x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
# image_gen_train.fit(x_train)

"""
# print("xtrain",x_train.shape)
# x_train_subset1 = np.squeeze(x_train[:12])
# print("xtrain_subset1",x_train_subset1.shape)
# print("xtrain",x_train.shape)
# x_train_subset2 = x_train[:12]  # 一次显示12张图片
# print("xtrain_subset2",x_train_subset2.shape)

# fig = plt.figure(figsize=(20, 2))
# plt.set_cmap('gray')
# 显示原始图片
# for i in range(0, len(x_train_subset1)):
#     ax = fig.add_subplot(1, 12, i + 1)
#     ax.imshow(x_train_subset1[i])
# fig.suptitle('Subset of Original Training Images', fontsize=20)
# plt.show()

# 显示增强后的图片
# fig = plt.figure(figsize=(20, 2))
# for x_batch in image_gen_train.flow(x_train_subset2, batch_size=12, shuffle=False):
#     for i in range(0, 12):
#         ax = fig.add_subplot(1, 12, i + 1)
#         ax.imshow(np.squeeze(x_batch[i]))
#     fig.suptitle('Augmented Images', fontsize=20)
#     plt.show()
#     break;
"""

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])


checkpoint_save_path = "./checkpoint/mnist.ckpt"
# 读取模型
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

# 保存模型
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)
# 增强数据喂入使用flow函数，batch_size改为128加快收敛速度
# history = model.fit(image_gen_train.flow(x_train, y_train, batch_size=128), epochs=5, validation_data=(x_test, y_test), validation_freq=1,
#                     callbacks=[cp_callback])
history = model.fit(x_train, y_train, batch_size=128, epochs=5, validation_data=(x_test, y_test), validation_freq=1,
                    callbacks=[cp_callback])
model.summary()

acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()