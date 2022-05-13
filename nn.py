import generator
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.preprocessing.sequence import TimeseriesGenerator


EPOCHS = 200

look_back = 48

X_train = np.linspace(0, 48*2, 48*4)
y_train = generator.generate(freq=2)
y_train = np.concatenate((y_train, y_train))

train_series = y_train.reshape((len(y_train), 1))

train_generator = TimeseriesGenerator(train_series, train_series,
                                      length=look_back,
                                      sampling_rate=1,
                                      stride=1,
                                      batch_size=8)

neurons = 16

model = Sequential()
model.add(LSTM(neurons, input_shape=(look_back, 1), return_sequences=True))
model.add(LSTM(1, input_shape=(neurons,), return_sequences=True))
model.add(LSTM(neurons, input_shape=(1,)))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

history = model.fit(train_generator, epochs=EPOCHS)

model.save('model.h5')

plt.plot(history.history['loss'])
plt.title('model loss')
plt.show()
