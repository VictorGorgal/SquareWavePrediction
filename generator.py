from scipy.signal import square
from numpy import linspace, zeros, insert, pi


def generate(freq):
    t = linspace(0, 16, 32, endpoint=False)
    ar = zeros(shape=(16,))
    ar = insert(ar, 14, square(pi / 8 * freq * t - 1) + 1)
    return ar


if __name__ == '__main__':
    import matplotlib.pyplot as plot

    freq = 2

    x = linspace(0, 24, 48, endpoint=False)
    ar = generate(freq)
    plot.plot(x, ar)
    plot.title(f'spikes: {freq}')
    plot.xlabel('Time (h)')
    plot.ylabel('Amplitude')
    plot.show()
