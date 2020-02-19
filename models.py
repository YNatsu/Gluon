import mxnet as mx
from mxnet.gluon import loss as gloss, Trainer
from mxnet import autograd
from gluoncv.utils import plot_history

import time


class Model:
    __slots__ = ('net', 'ctx', 'trainer', 'loss_fun', 'metric', 'history')

    def __init__(self, net, ctx):
        self.net = net
        self.ctx = ctx

        self.trainer = Trainer(net.collect_params(), 'adam', {'learning_rate': 0.01})

        self.loss_fun = None
        self.metric = None

        self.history = plot_history.TrainingHistory(['acc'])

    def compile(self, trainer, loss_fun, metric):
        self.trainer = trainer
        self.loss_fun = loss_fun
        self.metric = metric

    def test(self, dataloader):

        self.metric.reset()

        for data, label in dataloader:

            if self.ctx != mx.cpu():
                data = data.copyto(self.ctx)
                label = label.copyto(self.ctx)

            outputs = self.net(data)
            loss = self.loss_fun(outputs, label)

            self.metric.update(labels=label, preds=outputs)
            loss = loss.asnumpy().mean()

        return loss, self.metric.get()

    def summary(self):
        print(self.net)

    def fit(self, train_data, epochs, val_data=None):

        if val_data:
            self.history = plot_history.TrainingHistory(['acc_train', 'acc_val'])

        length = len(train_data)

        for epoch in range(1, 1 + epochs):

            self.metric.reset()
            tic = time.time()

            for step, (data, label) in enumerate(train_data):

                if self.ctx != mx.cpu():
                    data = data.copyto(self.ctx)
                    label = label.copyto(self.ctx)

                batch_size = data.shape[0]

                with autograd.record():
                    outputs = self.net(data)
                    loss = self.loss_fun(outputs, label)

                loss.backward()
                self.trainer.step(batch_size)
                self.metric.update(labels=label, preds=outputs)

                loss = loss.asnumpy().mean()

                _, acc = self.metric.get()
                t = time.time()
                step += 1

                if val_data:

                    loss, (_, val_acc) = self.test(val_data)
                    print(
                        f'Epoch:{epoch}/{epochs}  step:{step}/{length}  acc:{acc}  val_acc:{val_acc} loss:{loss} time:{t - tic}'
                    )

                    self.history.update([acc, val_acc])
                else:
                    print(
                        f'Epoch:{epoch}/{epochs}  step:{step}/{length}  acc:{acc}  loss:{loss} time:{t - tic}'
                    )

                    self.history.update([acc])

        return self.history


class Classification(Model):

    def __init__(self, net, ctx=mx.cpu()):
        super(Classification, self).__init__(net=net, ctx=ctx)
        self.loss_fun = gloss.SoftmaxCrossEntropyLoss()
        self.metric = mx.metric.Accuracy()


class Regression(Model):
    def __init__(self, net, ctx=mx.cpu()):
        super(Regression, self).__init__(net=net, ctx=ctx)
        self.loss_fun = gloss.L2Loss()
        self.metric = mx.metric.PearsonCorrelation()
