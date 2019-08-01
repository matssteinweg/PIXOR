import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def plot_precision_recall_curve(eval_dicts):
    """
    Plot precision recall curve using all available metrics stored in the evaluation dictionaries.
    :param eval_dicts: evaluation dictionary
    """

    # set up figure for precision-recall curve
    # one subplot for each distance range evaluated
    max_distance_ranges = np.max([len([key for key in eval_dict.keys() if not isinstance(key, str)][::-1]) for eval_dict in eval_dicts])
    fig, axs = plt.subplots(1, max_distance_ranges)
    for ax in axs:
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_ylim([0.0, 1.05])
        ax.set_xlim([0.0, 1.05])
        ax.grid(True)
        ax.set_aspect(0.8)
    colors = [(80/255, 127/255, 255/255), 'k', 'r', 'g']
    description = ['-', '--', '-.', ':']
    plt.subplots_adjust(wspace=0.4)

    # iterate over all provided evaluation dictionaries
    for eval_id, eval_dict in enumerate(eval_dicts):

        distance_ranges = [key for key in eval_dict.keys() if not isinstance(key, str)][::-1]

        # iterate over all evaluated distance ranges
        for range_id, distance_range in enumerate(distance_ranges):

            distance_dict = eval_dict[distance_range]

            # store plots for legend
            plots = []

            # get IoU thresholds used for evaluation
            thresholds = [key for key in distance_dict.keys() if not isinstance(key, str)]

            # loop over all IoU-thresholds
            plot = None
            for threshold in thresholds:
                recall = distance_dict[threshold]['recall']
                precision = distance_dict[threshold]['precision']
                plot = axs[range_id].plot(recall, precision, linewidth=1, linestyle=description[eval_id],
                                color=colors[eval_id], label='mAP[0.5:0.9] = {0:0.2f}'.format(distance_dict['mAP']))[0]
            plots.append(plot)

            # set subplot title and display legend
            axs[range_id].set_title('Range 0-{0:d}m'.format(distance_range))
            labels = [plot.get_label() for plot in plots]
            axs[range_id].legend(plots, labels, loc='lower right')

    plt.show()


if __name__ == '__main__':

    eval_dict = np.load('Eval/eval_dict_epoch_17.npz', allow_pickle=True)['eval_dict'].item()

    eval_dicts = [eval_dict]
    plot_precision_recall_curve(eval_dicts)



