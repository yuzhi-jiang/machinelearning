import numpy as np
from keras.optimizers import Adam
np.random.seed(1337)
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import matplotlib.pyplot as plt
X = np.random.randint(0, 100, (200, 2))
a = [[0.7], [0.5]]
Y = np.mat(X) * np.mat(a).getA() + 2 + np.random.normal(0, 0.05, (200, 1))
X_train, Y_train = X[:160], Y[:160]
X_test, Y_test = X[160:], Y[160:]

print(X_train.shape, X_train.shape)
model = Sequential()
model.add(Dense(10, input_dim=2))
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=adam, loss='mse')
history = model.fit(X_train, Y_train, 40, 1000)
print('\nTesting ------------')
cost = model.evaluate(X_test, Y_test, batch_size=40)
print('test cost:', cost)
W, b = model.layers[0].get_weights()
print('Weights=', W, '\nbiases=', b)
print(history.history)
loss = history.history['loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.title('Training loss')
plt.legend()
plt.figure()
plt.show()
