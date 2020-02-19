# -*- encoding: utf-8 -*-
# @Author  :   YNatsu 
# @Time    :   2020/2/19 11:48
# @Title:  : 

import models
import mxnet
from mxnet import gluon
from mxnet.gluon import data as gdata, loss as gloss, Trainer
from mxnet.gluon.nn import *

dataset = gdata.vision.ImageFolderDataset(root=r'../resource/mask')
transform_train = gdata.vision.transforms.Compose(
    [  # Compose将这些变换按照顺序连接起来
        # 将图片放大成高和宽各为 40 像素的正方形。
        gdata.vision.transforms.Resize(40),
        # 随机对高和宽各为 40 像素的正方形图片裁剪出面积为原图片面积 0.64 到 1 倍之间的小正方
        # 形，再放缩为高和宽各为 32 像素的正方形。
        gdata.vision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0), ratio=(1.0, 1.0)),
        # 随机左右翻转图片。
        gdata.vision.transforms.RandomFlipLeftRight(),
        # 将图片像素值按比例缩小到 0 和 1 之间，并将数据格式从“高 * 宽 * 通道”改为“通道 * 高 * 宽”。
        gdata.vision.transforms.ToTensor(),
        # 对图片的每个通道做标准化。
        gdata.vision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ]
)
dataloader = gluon.data.DataLoader(
    dataset=dataset.transform_first(transform_train), shuffle=True, batch_size=16
)


class Net(Block):
    def __init__(self, classes=10):
        super(Net, self).__init__()

        fun = 'relu'
        layer = Sequential()
        layer.add(

            Conv2D(channels=6, kernel_size=5, activation=fun),
            MaxPool2D(pool_size=2, strides=2),

            Conv2D(channels=16, kernel_size=5, activation=fun),
            MaxPool2D(pool_size=2, strides=2),

            Dense(120, activation=fun),
            Dense(84, activation=fun),
        )

        self.features = layer
        self.output = Dense(classes)

    def forward(self, x):
        for block in self._children.values():
            x = block(x)
        return x

    def summary(self):
        for block in self._children.values():
            print(block)


if __name__ == '__main__':

    net = Net(classes=2)
    net.initialize(ctx=mxnet.gpu())

    classification = models.Classification(net=net, ctx=mxnet.gpu())
    history = classification.fit(train_data=dataloader, epochs=16)

    history.plot(['acc'])
