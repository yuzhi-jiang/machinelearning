import numpy as np
import paddle
import pandas as pd
from matplotlib import pyplot as plt
from paddle.nn import Linear
import paddle.nn.functional as F

def train_test_split(data, ratio):
    offset = int(data.shape[0] * ratio)
    training_data = data[:offset]
    test_data = data[offset:]
    return training_data, test_data


def load_data():
    data = pd.read_csv('./data/data.csv')
    # 返回 训练集合验证集合 80% 20%


    data=data.values
    # data = data.reshape([data.shape[0] /2, 2])

    train_data, test_data = train_test_split(data, 0.8)


    # 归一化
    # 计算train数据集的最大值，最小值，平均值
    maximums, minimums, avgs = train_data.max(axis=0), train_data.min(axis=0), train_data.sum(axis=0) / \
                                                                               train_data.shape[0]

    # 记录数据的归一化参数，在预测时对数据做归一化
    global max_values
    global min_values
    global avg_values
    max_values = maximums
    min_values = minimums
    avg_values = avgs

    for i in range(2):
        data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i])

    train_data, test_data = train_test_split(data, 0.8)

    return train_data, test_data


# 定义线性回归模型
class Regressor(paddle.nn.Layer):
    def __init__(self):
        super(Regressor, self).__init__()
        self.fc = Linear(in_features=1, out_features=1)

    def forward(self, x):
        # print("x",x.shape)
        y = self.fc(x)
        return y


def train(model):
    # 加载数据
    train_data, test_data = load_data()
    # 定义模型
    model = Regressor()
    model.train()
    BATCH_SIZE = 10
    # 定义随机梯度下降优化器
    # opt = paddle.optimizer.SGD(learning_rate=0.001, parameters=model.parameters())
    opt = paddle.optimizer.Adam(learning_rate=0.01, weight_decay=paddle.regularizer.L2Decay(coeff=1e-5), parameters=model.parameters())



    for epoch in range(100):
        # 小批次训练,分割数据集,每个批次10个样本
        # 在每轮迭代开始之前，将训练数据的顺序随机的打乱
        np.random.shuffle(train_data)
        # 将训练数据进行拆分，每个batch包含10条数据
        mini_batches = [train_data[k:k + BATCH_SIZE] for k in range(0, len(train_data), BATCH_SIZE)]
        for iter, mini_batch in enumerate(mini_batches):
            x_data= np.array(mini_batch[:, :-1]).astype("float32")  # 获得当前批次训练数据
            y_data = np.array(mini_batch[:, -1:]).astype("float32")  # 获得当前批次训练标签（真实房价）
            # print(x_data)
            # x_data=x_data.reshape(1,-1)
            # y_data=y_data.reshape(1,-1)
            x_data = paddle.to_tensor(x_data)
            y_data = paddle.to_tensor(y_data)
            # 前向传播
            predicts = model(x_data)
            # 计算损失
            loss = F.square_error_cost(predicts, label=y_data)
            avg_loss = paddle.mean(loss)
            if iter % 20 == 0:
                print("epoch: {}, iter: {}, loss is: {}".format(epoch, iter, avg_loss.numpy()))

            # 反向传播
            avg_loss.backward()
            # 最小化loss,更新参数
            opt.step()
            # 清除梯度
            opt.clear_grad()
    save_model(model)


def save_model(model):
    # 保存模型参数，文件名为LR_model.pdparams
    paddle.save(model.state_dict(), 'LR_model.pdparams')
    print("模型保存成功，模型参数保存在LR_model.pdparams中")


def load_model(model):
    # 加载模型参数，文件名为LR_model.pdparams
    model.load_dict(paddle.load('LR_model.pdparams'))
    print("模型加载成功，模型参数保存在LR_model.pdparams中")


def getModel():
    model = Regressor()
    return model


# 测试
def validation():
    model = Regressor()
    load_model(model)
    model.eval()
    # 随机取样100个样本
    test_data = load_data()[1]
    print("test_data[0]",test_data[0])
    x_data = np.array(test_data[:, :-1]).astype("float32")
    x_data1 = np.array(test_data[:, :-1])
    y_data = np.array(test_data[:, -1:]).astype("float32")
    x_data = paddle.to_tensor(x_data)
    y_data = paddle.to_tensor(y_data)
    predicts = model(x_data)
    #反归一化
    for i in range(len(predicts)):
        predicts[i] = predicts[i] * (max_values[1] - min_values[1]) + avg_values[1]

    for i in range(len(y_data)):
        y_data[i] = y_data[i] * (max_values[1] - min_values[1]) + avg_values[1]
        y_data[i]=float(y_data[i].numpy())

    for i in range(len(x_data1)):
        x_data1[i] = x_data1[i] * (max_values[0] - min_values[0]) + avg_values[0]
    #plt画图

    # for i in range(len(predicts)):
        # print("因变量",x_data1[i],"预测值：", predicts[i].numpy(), "真实值：", y_data[i].numpy(),"公式：",2.0*x_data1[i]+3)
    # 坐标图
    print(x_data1[0:10])
    print(y_data[0:10])
    plt.xlabel("iter", fontsize=14)
    plt.ylabel("loss", fontsize=14)
    # print(x_data1.shape)
    # print(predicts.shape)
    # plt.plot(x_data1, y_data.numpy(), 'r-', label='真实值')
    plt.plot(x_data1, predicts.numpy(), 'b-', label='预测值')
    plt.show()
if __name__ == '__main__':
    # a,b=load_data()
    # module=getModel()
    # train(module)
    validation()
