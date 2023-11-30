import decimal
import zipfile
from math import sin

import numpy as np
import paddle
from matplotlib import pyplot as plt
from paddle import nn
from paddle.nn import Linear, Conv2D, Layer
from paddle.static.nn import fc

from base import save_model, train_test_split, normalize, load_model, denormalize


def get_data(x):
    c, r = x.shape
    y = np.sin(x * 3.14) + 1 + (0.02 * (2 * np.random.rand(c, r) - 1))
    return (y)


class Regressor(paddle.nn.Layer):
    def __init__(self):
        super(Regressor, self).__init__()

        # 使用全连接 激活函数为relu
        self.fc = Linear(in_features=1, out_features=16)
        self.relu = paddle.nn.ReLU()

        self.fc1 = Linear(in_features=16, out_features=64)
        self.relu1 = paddle.nn.ReLU()

        self.fc2 = Linear(in_features=64, out_features=16)
        self.relu2 = paddle.nn.ReLU()

        self.fc3 = Linear(in_features=16, out_features=1)

    def forward(self, x):
        # print("x",x.shape)
        y1 = self.fc(x)
        y2 = self.relu(y1)

        y3 = self.fc1(y2)
        y4 = self.relu1(y3)

        y5 = self.fc2(y4)
        y6 = self.relu2(y5)

        y7 = self.fc3(y6)

        return y7


def get_data3(val_number):
    x_val = np.linspace(-np.pi, np.pi, val_number).reshape(val_number, 1)
    return x_val, np.sin(x_val) * 2


def load_data():
    # path = './data/sin_data.zip'
    # x_data = np.load("./data/sin_data/x.npy")
    # y_data = np.load("./data/sin_data/y.npy")

    x_data = np.arange(-3, 3, 0.01).reshape(-1, 1)
    x_data, y_data = get_data3(500)

    combined_data = np.concatenate((x_data, y_data), axis=1)

    global max_values
    global min_values
    global avg_values

    # combined_data, max_values, min_values = normalize(combined_data)

    # train_data, test_data = train_test_split(combined_data, 0.2)

    return combined_data, []


def train(model):
    iters = []
    losses = []
    model = Regressor()
    tran_data, test_data = load_data()
    # x_data = np.array(tran_data[:, :-1]).astype("float32")
    # x_data1 = np.array(tran_data[:, :-1])
    # y_data = np.array(tran_data[:, -1:]).astype("float32")
    #
    # plt.plot(x_data, y_data, 'ro')
    # plt.show()
    # return
    model.train()

    epoch = 500
    BATCH_SIZE = 10
    # 使用adma优化器

    # opt = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())
    # opt = paddle.optimizer.Momentum(learning_rate=0.001, momentum=0.9, parameters=model.parameters())
    # opt = paddle.optimizer.Adagrad(learning_rate=0.01, parameters=model.parameters())
    # opt = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
    # opt = paddle.optimizer.Adam(learning_rate=0.001, weight_decay=paddle.regularizer.L2Decay(coeff=1e-3),parameters=model.parameters())
    opt = paddle.optimizer.Adam(learning_rate=0.01, parameters=model.parameters())

    for i in range(epoch):
        # 打乱数据
        # np.random.shuffle(tran_data)
        # 根据BATCH_SIZE进行分割
        mini_batches = [tran_data[k:k + BATCH_SIZE] for k in range(0, len(tran_data), BATCH_SIZE)]

        for iter, mini_batch in enumerate(mini_batches):
            x_data = np.array(mini_batch[:, :-1]).astype("float32")  # 获得当前批次训练数据
            y_data = np.array(mini_batch[:, -1:]).astype("float32")  # 获得当前批次训练标签（真实数据）
            # print("x_data",x_data.shape)
            # print(x_data)
            # print("y_data",y_data.shape)
            x_data = paddle.to_tensor(x_data)
            y_data = paddle.to_tensor(y_data)

            y_predict = model(x_data)
            # 方差
            loss = paddle.square(y_predict - y_data) * 0.5
            # loss = paddle.nn.functional.square_error_cost(y_predict, y_data)
            # loss = paddle.nn.functional.square_error_cost(y_predict, label=y_data)

            avg_loss = paddle.mean(loss)
            iters.append(iter + i * len(y_data))
            losses.append(avg_loss)

            if iter % 4 == 0:
                print("epoch: {}, iter: {}, loss is: {}".format(i, iter, float(avg_loss)))

            avg_loss.backward()

            # 最小化loss,更新参数
            opt.step()
            # 清除梯度
            opt.clear_grad()
        if(len(iters) % 100 == 0):
            plt.xlabel("iter", fontsize=14)
            plt.ylabel("loss", fontsize=14)
            plt.plot(iters, losses, color='red', label='train loss')
            plt.show()
            losses = []
            iters = []


# 推理
def validation(model):
    load_model(model)
    model.eval()
    # 随机取样100个样本
    test_data = load_data()[1]

    x_data = np.array(test_data[:, :-1]).astype("float32")
    x_data1 = np.array(test_data[:, :-1])
    y_data = np.array(test_data[:, -1:]).astype("float32")
    x_data = paddle.to_tensor(x_data)
    y_data = paddle.to_tensor(y_data)
    predicts = model(x_data)

    # plt画图

    # predicts=denormalize(predicts, max_values, min_values)
    # x_data1=denormalize(x_data1, max_values, min_values)
    # y_data=denormalize(y_data, max_values, min_values)

    for i in range(len(predicts)):
        print("因变量", x_data1[i], "预测值：", predicts[i].numpy(), "真实值：", y_data[i].numpy(), "sin",
              np.sin(x_data1[i]))
    # 坐标图

    plt.plot(x_data1, y_data.numpy(), 'r-', label='真实值')
    plt.plot(x_data1, predicts.numpy(), 'b-', label='预测值')
    plt.show()


def test():
    model = Regressor()
    model.load_dict(paddle.load('r2.pdparams'))

    model.eval()

    # x_data = np.arange(-3, 3, 0.01).reshape(-1, 1)
    # labels = get_data(x_data)

    val_number = 500  # 验证点的个数

    X_val = np.linspace(-np.pi, np.pi, val_number).reshape(val_number, 1)
    Y_val = np.sin(X_val) * 2

    print(X_val.shape)
    predicts = []
    for i in X_val:
        i = i.astype("float32")
        x = paddle.to_tensor(i)
        y = model(x)
        predicts.append(y.numpy())

    plt.plot(X_val, Y_val, 'r-', label='真实值')
    plt.plot(X_val, predicts, 'b-', label='预测值')
    plt.show()


def doTrain():
    model = Regressor()
    train(model)
    paddle.save(model.state_dict(), 'r2.pdparams');
    # save_model(model)


#
if __name__ == '__main__':
    doTrain()
    # x_data,y_data = get_data3(500)
    #
    # print(x_data)
    #
    # plt.plot(x_data, y_data, 'ro')
    # plt.show()

    # test()
    # save_model(model)
    # paddle.save(model.state_dict(), 'r1.pdparams');

    # load_model(model)
    # validation(model)
