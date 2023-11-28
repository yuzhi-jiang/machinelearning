# tensorflow_version 2.0
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

steps_per_cycle = 80
number_of_cycles = 50

# 在“t”列存放1、2、3……50*80+1
df = pd.DataFrame(np.arange(steps_per_cycle * number_of_cycles + 1), columns=["t"])
# 在“sin_t”列存放sin(t+噪声)
df["sin_t"] = df.t.apply(lambda x: np.sin(x * (2 * np.pi / steps_per_cycle)+ random.uniform(-1.0, +1.0) * 0.05))
# 以“t”为横轴，“sin_t”为纵轴绘图
# df[["sin_t"]].head(steps_per_cycle * 2).plot()
# plt.show()

def load_data(data, n_prev = 100):
	docX, docY = [], []
	for i in range(len(data)-n_prev):
		docX.append(data.iloc[i:i+n_prev].values)
		docY.append(data.iloc[i+n_prev].values)
	alsX = np.array(docX)
	alsY = np.array(docY)
	return alsX, alsY

# 划分出90%的数据用于训练
def train_test_split(df, test_size = 0.1, n_prev = 100):
	ntrn = round(len(df) * (1 - test_size))
	ntrn = int(ntrn)
	X_train, y_train = load_data(df.iloc[0:ntrn], n_prev)
	X_test, y_test = load_data(df.iloc[ntrn:], n_prev)
	return (X_train, y_train), (X_test, y_test)

length_of_sequences = 100
(X_train, y_train), (X_test, y_test) = train_test_split(df[["sin_t"]], n_prev = length_of_sequences)

# 建立神经网络
in_out_neurons = 1
hidden_neurons = 300

#gpus=tf.config.experimental.list_physical_devices('GPU')
#print(tf.config.experimental.list_physical_devices())
tf.config.experimental.set_visible_devices(tf.config.experimental.list_physical_devices('GPU')[0], 'GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)


model = Sequential()
model.add(tf.keras.layers.LSTM(hidden_neurons, batch_input_shape=(None, length_of_sequences, in_out_neurons), return_sequences=False))
model.add(Dense(in_out_neurons))
model.add(Activation("linear"))
model.compile(loss="mean_squared_error", optimizer="rmsprop")
model.fit(X_train, y_train, batch_size=600, epochs=15, validation_split=0.05)

# 预测
predicted = model.predict(X_test)

dataf = pd.DataFrame(predicted[:200])
dataf.columns = ["predict"]
dataf["input"] = y_test[:200]
dataf.plot(figsize=(15, 5))
plt.show()
