from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

from mxnet import nd, autograd
from mxnet.gluon import data as gdata, Trainer, loss as gloss
from mxnet.gluon.nn import *
from mxnet.gluon.rnn import LSTM

import matplotlib.pyplot as plt
import seaborn

seaborn.set()

n_features = 16
n_targets = 1

x, y = make_regression(n_samples=10000, n_features=n_features, n_targets=n_targets, noise=1)

x = x.reshape([-1, 1, n_features])

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.25, shuffle=True)


class Net(Block):

    def __init__(self, n):
        super(Net, self).__init__()

        layer = Sequential()
        layer.add(Dense(16, activation='relu'))
        layer.add(Dense(32, activation='relu'))
        # layer.add(LSTM(32))
        layer.add(Dense(n))

        self.layer = layer

    def forward(self, i): return self.layer(i)

    def summary(self):
        for block in self._children.values():
            print(block)


class Regression:

    def __init__(self, neural, opt, fun):
        self.lossList = []
        self.accList = []
        self.neural = neural
        self.opt = opt
        self.fun = fun

    def train(self, dataset, batch_size, epochs=8):
        dataloader = gdata.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

        length = len(dataloader)

        for epoch in range(1, epochs + 1):
            step = 1
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            for data, target in dataloader:
                data = data.astype('float32')
                target = target.astype('float32')

                with autograd.record():
                    result = self.neural(data)
                    loss = self.fun(result, target)

                loss.backward()

                self.opt.step(batch_size)

                loss = loss.asnumpy().mean()
                acc = self.accuracy(target, result)

                print(f'Epoch : {epoch} - step : {step}/{length} - acc : {acc} - loss : {loss} ')
                step += 1

                self.accList.append(acc)
                self.lossList.append(loss)

    def history(self):
        return self.accList, self.lossList

    @staticmethod
    def accuracy(y, y_):
        y, y_ = y.asnumpy(), y_.asnumpy()
        return r2_score(y, y_)

    def score(self, x, y):
        y_ = self.neural(x)
        loss = self.fun(y_, y).asnumpy().mean()
        acc = self.accuracy(y, y_)
        return [acc, loss]


if __name__ == '__main__':
    net = Net(n_targets)
    net.initialize()
    net.summary()

    dataset = gdata.ArrayDataset(x_train, y_train)
    x_test, y_test = nd.array(x_test).astype('float32'), nd.array(y_test)

    trainer = Trainer(net.collect_params(), 'adam', {'learning_rate': 0.01})
    criterion = gloss.L2Loss()

    model = Regression(net, trainer, criterion)

    model.train(dataset, batch_size=128, epochs=32)
    print(model.score(x_test, y_test))
    acc, loss = model.history()

    plt.subplot(1, 2, 1)
    plt.plot(acc, label='acc')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='loss')
    plt.legend()
    plt.show()
