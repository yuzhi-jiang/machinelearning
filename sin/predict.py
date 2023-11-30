import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

steps_per_cycle = 80
number_of_cycles = 50
length_of_sequences = 100

# 在“t”列存放1、2、3……50*80+1
df = pd.DataFrame(np.arange(steps_per_cycle * number_of_cycles + 1), columns=["t"])
# 在“sin_t”列存放sin(t+噪声)
df["sin_t"] = df.t.apply(lambda x: np.sin(x * (2 * np.pi / steps_per_cycle) + random.uniform(-1.0, +1.0) * 0.05))

def load_data(data, n_prev=100):
    docX, docY = [], []
    for i in range(len(data) - n_prev):
        docX.append(data.iloc[i:i + n_prev].values)
        docY.append(data.iloc[i + n_prev].values)
    alsX = np.array(docX)
    alsY = np.array(docY)
    return alsX, alsY
def train_test_split(df, test_size=0.1, n_prev=100):
    ntrn = round(len(df) * (1 - test_size))
    ntrn = int(ntrn)
    X_train, y_train = load_data(df.iloc[0:ntrn], n_prev)
    X_test, y_test = load_data(df.iloc[ntrn:], n_prev)
    return (X_train, y_train), (X_test, y_test)
model = Sequential()
(X_train, y_train), (X_test, y_test) = train_test_split(df[["sin_t"]], n_prev=length_of_sequences)


# model.save("aa.h5")
model = keras.models.load_model('aa.h5')

# 预测
predicted = model.predict(X_test)
print(predicted)
dataf = pd.DataFrame(predicted[:200])
dataf.columns = ["predict"]
dataf["input"] = y_test[:200]
dataf.plot(figsize=(15, 5))
plt.show()