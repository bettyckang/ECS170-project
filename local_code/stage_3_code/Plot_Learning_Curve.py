'''
Learning curve plots for the report
'''

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

SERIES_1 = '#2a78d6'  # blue: training set
SERIES_2 = '#eb6834'  # orange: testing set
TEXT = '#0b0b0b'
TEXT_MUTED = '#52514e'
GRID = '#e4e3df'


def style_axis(ax, title, ylabel):
    ax.set_title(title, loc='left', fontsize=11, color=TEXT)
    ax.set_xlabel('Training epoch', color=TEXT_MUTED)
    ax.set_ylabel(ylabel, color=TEXT_MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(colors=TEXT_MUTED)
    # epochs are whole numbers, so no 2.5 / 7.5 ticks
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    for side in ['left', 'bottom']:
        ax.spines[side].set_color(GRID)


def plot_learning_curve(history, save_path, title):
    # loss and accuracy have different scales, so they get separate panels instead of two y-axes;
    # final values go in the legend so they never collide when the two lines end close together
    epochs = history['epoch']
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 3.8))
    fig.suptitle(title, x=0.01, ha='left', fontsize=12, color=TEXT)

    ax_loss.plot(epochs, history['train_loss'], color=SERIES_1, linewidth=2,
                 label='Training set (final %.3f)' % history['train_loss'][-1])
    ax_acc.plot(epochs, history['train_accuracy'], color=SERIES_1, linewidth=2,
                label='Training set (final %.4f)' % history['train_accuracy'][-1])
    if history['test_loss']:
        ax_loss.plot(epochs, history['test_loss'], color=SERIES_2, linewidth=2,
                     label='Testing set (final %.3f)' % history['test_loss'][-1])
        ax_acc.plot(epochs, history['test_accuracy'], color=SERIES_2, linewidth=2,
                    label='Testing set (final %.4f)' % history['test_accuracy'][-1])
    ax_loss.set_ylim(bottom=0)
    style_axis(ax_loss, 'Loss', 'Cross-entropy loss')
    style_axis(ax_acc, 'Accuracy', 'Accuracy')
    ax_loss.legend(frameon=True, framealpha=0.9, edgecolor='none', loc='best', labelcolor=TEXT)
    ax_acc.legend(frameon=True, framealpha=0.9, edgecolor='none', loc='best', labelcolor=TEXT)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print('saved learning curve to', save_path)
