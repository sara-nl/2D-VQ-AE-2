import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    for n_embeddings in (128, 256, 512, 1024):
        histograms = [
            np.load(f) for f in (
                f'embedding_idx_histogram_{n_embeddings}_{train}.npy'
                for train in ('train', 'validation', 'test')
            )
        ]
        train_sorted_idx = np.argsort(histograms[0])[::-1]  # we want high to low

        histograms = [hist[train_sorted_idx] for hist in histograms]  # sorted
        histograms = np.round([100 * hist / hist.sum() for hist in histograms], 0)  # normed
        breakpoint()
        x = np.arange(len(train_sorted_idx))
        width = 0.95

        fig, ax = plt.subplots()
        fig.set_size_inches((32, 18))

        rects1 = ax.bar(x - width / 3, histograms[0], width / 3, label='train')
        rects2 = ax.bar(x, histograms[1], width / 3, label='validation')
        rects3 = ax.bar(x + width / 3, histograms[2], width / 3, label='test')

        ax.set_ylabel('Marginal probability %')
        ax.set_title(f'Marginal distribution of {n_embeddings} embedding indices')

        # plt.xticks(x, labels=train_sorted_idx)
        ax.legend()
        #
        ax.bar_label(rects1, padding=1)
        ax.bar_label(rects2, padding=1)
        ax.bar_label(rects3, padding=1)

        plt.tight_layout()
        plt.show()
        plt.savefig(f'embedding_idx_histogram_{n_embeddings}.png')
