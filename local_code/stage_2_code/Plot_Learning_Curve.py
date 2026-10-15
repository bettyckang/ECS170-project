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
