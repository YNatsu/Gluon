from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

from mxnet import nd, autograd
from mxnet.gluon import data as gdata, Trainer, loss as gloss
from mxnet.gluon.nn import *
from mxnet.gluon.rnn import LSTM, RNN

import matplotlib.pyplot as plt
import seaborn

seaborn.set()

time_step = 16

_, data = make_regression(n_samples=1000, n_features=1, n_targets=1, noise=1)

scaler = MinMaxScaler()
data = scaler.fit_transform(data.reshape([-1, 1]))


def load_data(data, time_step):
    x_list = []
    y_list = []

    for i in range(len(data) - time_step):
        x = data[i: i + time_step]
        y = data[i + time_step]
        x_list.append(x)
        y_list.append(y)

    return nd.array(x_list).astype('float32').reshape([-1, time_step, 1]), nd.array(y_list).astype('float32')


x, y = load_data(data, time_step)

split = int(len(y) * 3 / 4)

x_train, x_test, y_train, y_test = x[:split], x[split:], y[:split], y[split:]


# print(y.shape, x_train.shape)

class Net(Block):
    def __init__(self):
        super(Net, self).__init__()

        self.l1 = RNN(16)
        self.l2 = Sequential()
        self.l2.add(
            # Dense(32, activation='relu'),
            Dense(1, activation='relu')
        )

    def forward(self, i):
        i = self.l1(i)[:, -1, :]
        i = self.l2(i)
        return i

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

        h = nd.normal(shape=[1, 16, 32])

        for epoch in range(1, epochs + 1):
            step = 1
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

    def predict(self, i):
        return self.neural(i).asnumpy()


if __name__ == '__main__':
    net = Net()
    net.initialize()

    dataset = gdata.ArrayDataset(x_train, y_train)

    x_test, y_test = nd.array(x_test).astype('float32'), nd.array(y_test)

    trainer = Trainer(net.collect_params(), 'adam', {'learning_rate': 0.01})
    criterion = gloss.L2Loss()

    model = Regression(net, trainer, criterion)
    model.train(dataset, batch_size=128, epochs=160)
    print(model.score(x_test, y_test))

    y_ = model.predict(x_train)
    y = y_train.asnumpy()

    acc, loss = model.history()

    plt.subplot(2, 2, 1)
    plt.plot(acc, label='acc')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(loss, label='loss')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(y, label='y')
    plt.plot(y_, label='y_')

    plt.show()
