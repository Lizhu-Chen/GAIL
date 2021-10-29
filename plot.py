import matplotlib.pyplot as plt
import numpy as np


def loss_plot(y1, y2, y3, y1_name, y2_name, y3_name):
    x1 = np.arange(len(y1))
    x2 = np.arange(len(y2))
    x3 = np.arange(len(y3))
    fig = plt.figure(1)
    ax1 = plt.subplot(3, 1, 1)
    plt.plot(x1, y1)
    plt.ylabel(y1_name)
    ax2 = plt.subplot(3, 1, 2)
    plt.plot(x2, y2)
    plt.ylabel(y2_name)
    ax3 = plt.subplot(3, 1, 3)
    plt.plot(x3, y3)
    plt.ylabel(y3_name)
    plt.xlabel('Iterations')
    plt.show()
