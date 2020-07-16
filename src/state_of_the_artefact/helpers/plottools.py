import numpy as np
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, classes, title, xlabel, ylabel,
                          cmap=plt.cm.Blues, colorbar=True,
                          annotate=False, figsize=(6, 6)):
    """ Draw a confusion matrix. """

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    maxv = np.max(cm)

    if annotate:
        for x, row in enumerate(cm):
            for y, col in enumerate(row):
                c = 'white' if (col / maxv) < .5 else 'black'
                ax.annotate(f'{col}', xy=(y, x), color=c,
                            horizontalalignment='center',
                            verticalalignment='center')

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    if colorbar:
        cax = fig.add_axes([ax.get_position().x1 + 0.02,
                            ax.get_position().y0,
                            0.022,
                            ax.get_position().height])
        cbar = fig.colorbar(im, cax=cax)

    ax.set_title(title)

    ax.set_xlabel(xlabel)
    ax.xaxis.set_major_locator(ticker.FixedLocator(classes))

    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_locator(ticker.FixedLocator(classes))

    return fig
