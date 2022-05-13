from scipy.signal import square
from numpy import linspace, zeros, insert, pi
import clean


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

    # Plot the square wave signal
    plot.plot(x, ar)

    # Give a title for the square wave plot
    plot.title(f'spikes: {freq}')

    # Give x axis label for the square wave plot
    plot.xlabel('Time (h)')

    # Give y axis label for the square wave plot
    plot.ylabel('Amplitude')
    plot.grid(True, which='both')

    # Provide x axis and line color
    plot.axhline(y=0, color='k')

    # Set the max and min values for y axis
    plot.ylim(0, 5)

    # Display the square wave drawn
    plot.show()

    # new = clean.clean_signal(24, 144, ar, threshold=0.1, debug=True)
    # plot.plot(x, new)
    # plot.show()
