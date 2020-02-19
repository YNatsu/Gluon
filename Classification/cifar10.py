from mxnet import gluon
from mxnet.gluon.data.vision import transforms
from mxnet.gluon import data as gdata

import mxnet as mx

from gluoncv.model_zoo import get_model
from gluoncv.data import transforms as gcv_transforms

from models import Classification

import matplotlib.pyplot as plt

transform_fn = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

transform_train = transforms.Compose([
    gcv_transforms.RandomCrop(32, pad=4),
    transforms.RandomFlipLeftRight(),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

cifar_train = gdata.vision.CIFAR10(train=True, root=r'../resource/cifar10')
cifar_test = gdata.vision.CIFAR10(train=False, root=r'../resource/cifar10')

batch_size = 16

train_data = gdata.DataLoader(
    cifar_train.transform_first(transform_train),
    batch_size=batch_size, shuffle=True
)

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

val_data = gluon.data.DataLoader(
    cifar_test.transform_first(transform_test),
    batch_size=batch_size, shuffle=False
)

if __name__ == '__main__':
    ctx = mx.gpu()
    net = get_model('cifar_resnet20_v1', classes=10, pretrained=True)
    net.collect_params().reset_ctx(ctx)
    net.initialize(ctx=ctx)

    model = Classification(net=net, ctx=ctx)

    model.summary()

    history = model.fit(train_data, 1, val_data)

    history.plot()
    plt.legend()
    plt.show()
