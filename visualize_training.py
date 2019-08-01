import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def plot_history(metrics):
    """
    Plot evoluction of training and validation loss over the training period.
    :param metrics: dictionary containing training and validation loss
    """

    fig, axs = plt.subplots(1, 2)
    plt.subplots_adjust(top=0.75, bottom=0.25, wspace=0.4)

    train_loss = metrics['train_loss']
    val_loss = metrics['val_loss']

    batch_size = 6
    epochs = batch_size * (len(train_loss) + len(val_loss)) // 6481

    for ax_id, ax in enumerate(axs):
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.set_yscale('log')
        ax.grid(True)

        if ax_id == 0:
            ax.set_ylim([0.5 * min(train_loss), max(train_loss)])
            ax.set_xlim([0.0, len(train_loss)])
            ax.set_title('Training Loss')
            step_size = 3
            ticks = np.arange(0, len(train_loss), step_size * len(train_loss) // (epochs))
            labels = np.arange(1, epochs + 1, step_size)
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels)
            ax.plot(train_loss)
        else:
            ax.set_ylim([0.5 * min(val_loss), max(val_loss)])
            ax.set_xlim([0.0, len(val_loss)])
            ax.set_title('Validation Loss')
            step_size = 3
            ticks = np.arange(0, len(val_loss), step_size * len(val_loss) // (epochs))
            labels = np.arange(1, epochs + 1, step_size)
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels)
            ax.plot(val_loss)

    plt.show()


if __name__ == '__main__':

    metrics = np.load('Metrics/metrics_17.npz', allow_pickle=True)['history'].item()
    plot_history(metrics)

