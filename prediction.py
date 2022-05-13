from keras.models import load_model
import matplotlib.pyplot as plt
import generator
import numpy as np


def add(array, element):
    new = np.zeros(array.shape)
    new[0, :-1, 0] = array[0, 1:, 0]
    new[0, -1, 0] = element
    return new


def predict(model, data, future=144):
    y_hat = []
    for _ in range(future):
        pred = model.predict(data)[0, 0]

        y_hat.append(pred)
        data = add(data, pred)

    return y_hat


x = np.linspace(0, 72, 144)
y = generator.generate(freq=2)
y = y.reshape((1, len(y), 1))
y2 = np.concatenate((y, y, y))

model = load_model('model.h5')

y_hat = predict(model, y, future=96)

x2 = np.linspace(24, 72, 96)
plt.plot(x, y2.flatten(), label='actual')
plt.plot(x2, y_hat, linestyle=':', c='k', label='predicted')
plt.legend()
plt.show()
