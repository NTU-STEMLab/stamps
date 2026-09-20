import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import math

def plot_marginal(m, v, zs=None, i=0, show=True):
    mu = m[i]
    variance = v[i]
    sigma = math.sqrt(variance)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    plt.plot(x, mlab.normpdf(x, mu, sigma), label='general marginal pdf')

    if zs:
        zs_i = zs[i]
        mu = zs_i[1]
        variance = zs_i[2]
        sigma = math.sqrt(variance)
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        plt.plot(x, mlab.normpdf(x, mu, sigma), label='specific soft pdf')
    plt.title('index={i}'.format(i=str(i)))
    plt.legend()
    if show:
        plt.show()
    return plt