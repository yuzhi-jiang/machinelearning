import keras
import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt


def train_test_split(data, ratio):
    train_data = data.sample(frac=ratio)
    test_data = data.drop(train_data.index)
    return train_data, test_data


def getData():
    data = pd.read_csv('data/1_film.csv')
    return data

def load_data():
    data=getData()
    train_data, test_data = train_test_split(data, 0.2)
    return train_data, test_data


def getLabel(data):
    label = data['filmnum']
    return label


def get_workData(data):
    data_y=getLabel(data)
    data_x=np.array(data[data.columns[1:]])
    return data_x,data_y



def getModel1():
    model = Sequential()
    # model.add(tf.keras.layers.LSTM(hidden_neurons, batch_input_shape=(None, length_of_sequences, in_out_neurons),
    #                                return_sequences=False))
    model.add(Dense(1,input_shape=(3,)))

    # model.add(Activation("linear"))
    # model.add(Dense(4,activation='relu'))
    # model.add(Dense(4,activation='relu'))
    # model.add(Dense(1))
    # model.add(Activation("linear"))
    # model.compile(loss="mean_squared_error", optimizer="sgd")
    model.compile(loss="mse", optimizer="adam")
    return model
def getModel2():
    model = Sequential()

    model.add(Dense(3, activation='relu', input_shape=(3,)))
    # model.add(keras.layers.BatchNormalization())
    # model.add(Dense(16, activation='relu'))
    # model.add(keras.layers.BatchNormalization())
    # model.add(Dense(16, activation='relu'))
    # model.add(keras.layers.BatchNormalization())
    model.add(Dense(1))
    # model.compile(loss="mse", optimizer="adam")
    # 编译模型
    model.compile(optimizer='rmsprop', loss='mse', metrics=['acc'])
    return model


def train(model,data):
    data_x,data_y=data
    history=model.fit(data_x, data_y, epochs=epochsNum, batch_size=batch_size)
    epochs=range(len(history.history['loss']))
    print(history.history)
    # plt.figure()
    # plt.plot(epochs,history.history['acc'],'b',label='Training acc')
    # # plt.plot(epochs,history.history['val_acc'],'r',label='Validation acc')
    # plt.title('Traing and Validation accuracy')
    # plt.legend()

    plt.figure()
    plt.plot(epochs,history.history['loss'],'b',label='Training loss')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    print('训练完成')


def doTrain(model,modelName):
    data=getData()
    data,data_test=train_test_split(data,0.8)

    data_x,data_y=get_workData(data)

    data1=(data_x,data_y)
    train(model,data1)
    model.save("model\\"+modelName+".h5")


df=getData()
y=df.loc[:,'filmnum']
x=df.loc[:,['filmsize','ratio','quality']]
ytest=[];
for i in range(101,126):
    ytest.append(y[i]);
xtest=[];
for i in range(101,126):
    x1=x.loc[i]
    x2=[];
    for c in range(0,3):
        x2.append(x1[c]);
    xtest.append(x2);
def doTest(modelName):
    model=keras.models.load_model("model\\"+modelName+".h5")
    data=getData()
    data=data[101:]
    print(data)
    # data,data_test=train_test_split(data,0.8)
    data_x,data_y=get_workData(data)

    x=range(101,126);
    predicted=model.predict(data_x)

    plt.clf();
    plt.plot(x,predicted,'ro')
    plt.plot(x,data_y,'bo')
    plt.savefig("散点图.png")
    plt.show()




    # print(predicted)
    #
    # # for i in range(0, 10):
    # #     print(data_x[i],predicted[i],data_y[i])
    #
    # dataf = pd.DataFrame(predicted[:200])
    # dataf.columns = ["predict"]
    # dataf["input"] = data_y[:200]
    # dataf.plot(figsize=(15, 5))
    # plt.show()

model=getModel1()
epochsNum=600
batch_size=20
modelName = "model1-600"
if __name__ == '__main__':

    doTrain(model,modelName)
    doTest(modelName)