import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

class TimeSeriesPredictionModel:
    def __init__(self, steps_per_cycle=80, number_of_cycles=50, length_of_sequences=100,
                 hidden_neurons=300, batch_size=600, epochs=15):
        self.steps_per_cycle = steps_per_cycle
        self.number_of_cycles = number_of_cycles
        self.length_of_sequences = length_of_sequences
        self.hidden_neurons = hidden_neurons
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None

    def generate_sin_wave_data(self):
        df = pd.DataFrame(np.arange(self.steps_per_cycle * self.number_of_cycles + 1), columns=["t"])
        df["sin_t"] = df.t.apply(lambda x: np.sin(x * (2 * np.pi / self.steps_per_cycle) +
                                                 random.uniform(-1.0, +1.0) * 0.05))
        return df

    def load_data(self, data, n_prev=100):
        docX, docY = [], []
        for i in range(len(data)-n_prev):
            docX.append(data.iloc[i:i+n_prev].values)
            docY.append(data.iloc[i+n_prev].values)
        alsX = np.array(docX)
        alsY = np.array(docY)
        return alsX, alsY

    def train_test_split(self, df, test_size=0.1, n_prev=100):
        ntrn = round(len(df) * (1 - test_size))
        ntrn = int(ntrn)
        X_train, y_train = self.load_data(df.iloc[0:ntrn], n_prev)
        X_test, y_test = self.load_data(df.iloc[ntrn:], n_prev)
        return (X_train, y_train), (X_test, y_test)

    def build_model(self):
        self.model = Sequential()
        self.model.add(LSTM(self.hidden_neurons, batch_input_shape=(None, self.length_of_sequences, 1),
                            return_sequences=False))
        self.model.add(Dense(1))
        self.model.add(Activation("linear"))
        self.model.compile(loss="mean_squared_error", optimizer="rmsprop")

    def train_model(self, X_train, y_train, validation_split=0.05):
        self.model.fit(X_train, y_train, batch_size=self.batch_size, epochs=self.epochs, validation_split=validation_split)

    def save_model(self, filename="time_series_model.h5"):
        self.model.save(filename)

    def load_model(self, filename="time_series_model.h5"):
        self.model = tf.keras.models.load_model(filename)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def plot_results(self, predicted, y_test, num_points=200):
        dataf = pd.DataFrame(predicted[:num_points])
        dataf.columns = ["predict"]
        dataf["input"] = y_test[:num_points]
        dataf.plot(figsize=(15, 5))
        plt.show()

    def test_model(self):
        # Generate synthetic data
        df = self.generate_sin_wave_data()

        # Split data and train model
        (X_train, y_train), (X_test, y_test) = self.train_test_split(df[["sin_t"]])
        self.build_model()
        self.train_model(X_train, y_train)

        # Save and load model
        self.save_model()
        self.load_model()

        # Predict and plot results
        predicted = self.predict(X_test)
        self.plot_results(predicted, y_test)

# Example usage:
model_instance = TimeSeriesPredictionModel()
model_instance.test_model()
