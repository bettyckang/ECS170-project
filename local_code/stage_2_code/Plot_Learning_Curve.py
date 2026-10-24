'''
Learning curve plots for the report
'''

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SERIES_1 = '#2a78d6'  # blue: training
SERIES_2 = '#eb6834'  # orange: testing
TEXT = '#0b0b0b'
TEXT_MUTED = '#52514e'
GRID = '#e4e3df'


def style_axis(ax, title, ylabel):
    ax.set_title(title, loc='left', fontsize=11, color=TEXT)
    ax.set_xlabel('Epoch', color=TEXT_MUTED)
    ax.set_ylabel(ylabel, color=TEXT_MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(colors=TEXT_MUTED)
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    for side in ['left', 'bottom']:
        ax.spines[side].set_color(GRID)


def plot_learning_curve(history, save_path, title='MLP learning curve'):
    epochs = history['epoch']
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 3.8))
    fig.suptitle(title, x=0.01, ha='left', fontsize=12, color=TEXT)

    # loss and accuracy have different scales, so they get separate panels instead of two y-axes
    ax_loss.plot(epochs, history['train_loss'], color=SERIES_1, linewidth=2)
    style_axis(ax_loss, 'Training loss', 'Loss')

    ax_acc.plot(epochs, history['train_accuracy'], color=SERIES_1, linewidth=2, label='Train')
    if history['test_accuracy']:
        ax_acc.plot(epochs, history['test_accuracy'], color=SERIES_2, linewidth=2, label='Test')
        ax_acc.annotate('Test %.4f' % history['test_accuracy'][-1], (epochs[-1], history['test_accuracy'][-1]),
                        xytext=(-4, -14), textcoords='offset points', ha='right', fontsize=9, color=TEXT)
    ax_acc.annotate('Train %.4f' % history['train_accuracy'][-1], (epochs[-1], history['train_accuracy'][-1]),
                    xytext=(-4, 6), textcoords='offset points', ha='right', fontsize=9, color=TEXT)
    style_axis(ax_acc, 'Accuracy', 'Accuracy')
    ax_acc.legend(frameon=False, loc='lower right', labelcolor=TEXT)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print('saved learning curve to', save_path)


def plot_loss_curve(history, save_path):
    # the report's convergence plot (3.5): loss on the full training and testing sets after every epoch
    epochs, train, test = history['epoch'], history['train_loss_eval'], history['test_loss']
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(epochs, train, color=SERIES_1, lw=2, marker='o', ms=3.5, label='Training set (60,000 images)')
    ax.plot(epochs, test, color=SERIES_2, lw=2, marker='o', ms=3.5, label='Testing set (10,000 images)')
    ax.set_xlim(0.5, epochs[-1] + 0.5)
    ax.set_ylim(0, max(max(train), max(test)) * 1.06)
    ax.set_xticks([1] + list(range(5, epochs[-1] + 1, 5)))
    style_axis(ax, 'Final MLP, loss measured on each full dataset after every epoch', 'Cross-entropy loss')
    ax.title.set_fontsize(9.5)
    ax.title.set_color(TEXT_MUTED)
    ax.set_xlabel('Training epoch', color=TEXT_MUTED)
    for value in [train[-1], test[-1]]:
        ax.annotate('%.3f' % value, (epochs[-1], value), xytext=(-2, 8), textcoords='offset points',
                    ha='right', fontsize=9, color=TEXT)
    ax.annotate('%.3f' % train[0], (epochs[0], train[0]), xytext=(8, 0), textcoords='offset points',
                va='center', fontsize=9, color=TEXT)
    ax.legend(frameon=False, loc='upper right', labelcolor=TEXT)
    fig.suptitle('Training loss converges from %.3f to %.3f over %d epochs' % (train[0], train[-1], epochs[-1]),
                 x=0.01, ha='left', fontsize=12, color=TEXT)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print('saved loss curve to', save_path)
