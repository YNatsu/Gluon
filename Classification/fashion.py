import mxnet as mx

from mxnet.gluon import loss as gloss, data as gdata, Trainer
from mxnet.gluon.nn import *
from models import Classification

import seaborn

seaborn.set()


def get_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


class Net(Block):
    def __init__(self, classes=10, ):
        super(Net, self).__init__()
        fun = 'sigmoid'
        net = Sequential()
        net.add(
            BatchNorm(),
            Conv2D(channels=6, kernel_size=5, activation=fun),
            MaxPool2D(pool_size=2, strides=2),

            BatchNorm(),
            Conv2D(channels=16, kernel_size=5, activation=fun),
            MaxPool2D(pool_size=2, strides=2),

            Dense(120, activation=fun),
            Dense(84, activation=fun),
        )

        self.features = net
        self.output = Dense(classes)

    def forward(self, x):
        x = self.features(x)
        x = self.output(x)
        return x




mnist_train = gdata.vision.FashionMNIST(train=True, root=r'../resource/fashion')
mnist_test = gdata.vision.FashionMNIST(train=False, root=r'../resource/fashion')

transform = gdata.vision.transforms.ToTensor()
train_iter = gdata.DataLoader(dataset=mnist_train.transform_first(transform), shuffle=True, batch_size=128)
test_iter = gdata.DataLoader(mnist_test.transform(transform), batch_size=128)

if __name__ == '__main__':
    ctx = mx.gpu( )
    net = Net(classes=10)
    net.initialize(ctx=ctx)
    print(net)

    trainer = Trainer(net.collect_params(), 'adam', {'learning_rate': 0.01})
    fun = gloss.SoftmaxCrossEntropyLoss()

    model = Classification(neural=net, fun=fun, opt=trainer)

    model.train(mnist_train.transform_first(transform), batch_size=256, epochs=32)
