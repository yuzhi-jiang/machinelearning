import keras
import numpy as np
import pandas as pd
from keras import Sequential
import tensorflow as tf
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from matplotlib import pyplot as plt

length_of_sequences = 100
in_out_neurons = 4
hidden_neurons = 300


from data_generate import get_data, train_test_split, get_workData

data=get_data(2000)

tran_data,test_data=train_test_split(data,0.8)

X_train=tran_data[:,:-1]
y_train=tran_data[:,-1:]
X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)

X_test=test_data[:,:-1]
y_test=test_data[:,-1:]
X_test = np.array(X_test, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32)


def getModel1():
    model = Sequential()
    # model.add(tf.keras.layers.LSTM(hidden_neurons, batch_input_shape=(None, length_of_sequences, in_out_neurons),
    #                                return_sequences=False))
    model.add(Dense(1,input_shape=(4,)))
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

    model.add(Dense(4, activation='relu', input_shape=(4,)))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam")
    # 编译模型
    # model.compile(optimizer='rmsprop', loss='mse', metrics=['acc'])
    return model


def getModel3():
    model = Sequential()
    model.add(Dense(units=10, input_dim=4))
    model.add(Activation('tanh'))
    model.add(Dense(units=1))
    model.add(Activation('tanh'))
    model.summary()
    sgd = SGD(lr=0.3)
    model.compile(optimizer=sgd, loss='mse')
    return model

def getModel4():
    model = Sequential()
    model.add(Dense(1, input_dim=4, activation='sigmoid'))
    # model.add(Dense(1))
    from keras import optimizers
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=adam, loss='mse',metrics=['acc'])
    return model


def plt_loss(history):
    print(history.history)
    epochs=range(len(history.history['acc']))
    plt.figure()
    plt.plot(epochs,history.history['acc'],'b',label='Training acc')
    # plt.plot(epochs,history.history['val_acc'],'r',label='Validation acc')
    plt.title('Traing and Validation accuracy')
    plt.legend()
    # plt.savefig('/root/notebook/help/figure/model_V3.1_acc.jpg')

    plt.figure()
    plt.plot(epochs,history.history['loss'],'b',label='Training loss')
    plt.plot(epochs,history.history['val_loss'],'r',label='Validation val_loss')
    plt.title('Traing and Validation loss')
    plt.legend()
    # plt.savefig('/root/notebook/help/figure/model_V3.1_loss.jpg')


def train():
    model = getModel1()
    history=model.fit(X_train, y_train, batch_size=100, epochs=500,validation_data=(X_test, y_test))
    model.save('model_V3.6.h5')




def main():
    # print(X_train.shape)
    # print(X_train.shape)

    # train()
    #
    # print(X_train)

    model=keras.models.load_model('model_V3.6.h5')



    test_data1=get_data(2000)

    for i in range(0,100):
        test_data1[i]=[0,0,0,0,100]

    test_data1[101]=[0,0,0,0,100]
    test_data1[102]=[0,0,0,0,3]
    test_data1[103]=[0,0,0,0,1]
    test_data1[104]=[0,0,0,0,2]

    test_data1[0][0]=12
    test_data1[0][4]=get_workData(test_data1[0][:-1])
    for i in range(1,100):
        test_data1[i]==test_data1[i-1]
        test_data1[i][0]=i+1
        test_data1[i][4]=get_workData(test_data1[i][:-1])


    X_test1=test_data1[:,:-1]
    y_test1=test_data1[:,-1:]
    X_test1 = np.array(X_test1, dtype=np.float32)
    y_test1 = np.array(y_test1, dtype=np.float32)

    predicted=model.predict(X_test1)


    for i in range(0, 10):
        print(X_test1[i],predicted[i],y_test1[i])

    dataf = pd.DataFrame(predicted[:200])
    dataf.columns = ["predict"]
    dataf["input"] = y_test1[:200]
    dataf.plot(figsize=(15, 5))
    plt.show()

if __name__ == '__main__':
    main()