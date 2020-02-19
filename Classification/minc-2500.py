from mxnet.gluon import data as gdata, loss as gloss, Trainer, nn
from mxnet.gluon.data.vision import transforms
import mxnet
from mxnet import autograd, init

from gluoncv import model_zoo, utils
import time

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomFlipLeftRight(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

root = r'../resource/minc-2500-tiny/minc-2500-tiny/'

train_path = f'{root}train'
val_path = f'{root}val'
test_path = f'{root}test'

batch_size = 8
classes = 23
epochs = 16
ctx = mxnet.gpu()

train_data = gdata.DataLoader(
    gdata.vision.ImageFolderDataset(train_path).transform_first(transform_train),
    shuffle=True, batch_size=batch_size
)

val_data = gdata.DataLoader(
    gdata.vision.ImageFolderDataset(val_path).transform_first(transform_train),
    shuffle=False, batch_size=batch_size
)

test_data = gdata.DataLoader(
    gdata.vision.ImageFolderDataset(test_path).transform_first(transform_train),
    shuffle=False, batch_size=batch_size
)

if __name__ == '__main__':

    model_name = 'ResNet50_v2'
    net = model_zoo.get_model(model_name, pretrained=True)
    net.output = nn.Dense(classes)
    net.output.initialize(ctx=ctx, init=init.Xavier())
    net.collect_params().reset_ctx(ctx)
    net.hybridize()

    # net = model_zoo.get_model('cifar_resnet20_v1', classes=classes)
    # net.initialize(ctx=ctx)

    print(net)

    trainer = Trainer(net.collect_params(), 'adam', {'learning_rate': 0.01})
    loss_fun = gloss.SoftmaxCrossEntropyLoss()
    metric = mxnet.metric.Accuracy()
    history = utils.plot_history.TrainingHistory(['train_acc', 'val_acc'])

    for epoch in range(1, 1 + epochs):
        tic = time.time()
        metric.reset()
        for step, (data, label) in enumerate(train_data):
            data = data.copyto(ctx)
            label = label.copyto(ctx)

            with autograd.record():
                outputs = net(data)
                loss = loss_fun(outputs, label)

            loss.backward()
            trainer.step(batch_size)

            metric.update(label, outputs)

            print(metric.get())
