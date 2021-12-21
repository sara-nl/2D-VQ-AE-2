import numpy as np
import matplotlib.pyplot as plt


histograms = [
    np.load(f) for f in (
        'histogram_train.npy',
        'histogram_val.npy',
        'histogram_test.npy',
    )
]

histograms_normed = np.round([hist / hist.sum() for hist in histograms], 4)
histograms_normed2 = np.round([hist[1:] / hist[1:].sum() for hist in histograms], 4)

x = np.arange(3)
x2 = np.arange(2)
width = 0.95

fig, ax = plt.subplots()

rects1 = ax.bar(x - width / 3, histograms_normed[0], width / 3, label='train')
rects2 = ax.bar(x, histograms_normed[1], width / 3, label='validation')
rects3 = ax.bar(x + width / 3, histograms_normed[2], width / 3, label='test')

ax.set_ylabel('Marginal probability')
ax.set_title('Marginal distribution of background/tissue/cancer')

plt.xticks(x, labels=['background', 'tissue', 'cancer'])
ax.legend()

ax.bar_label(rects1, padding=1)
ax.bar_label(rects2, padding=1)
ax.bar_label(rects3, padding=1)

plt.savefig('hist')

fig, ax = plt.subplots()

rects1 = ax.bar(x2 - width / 3, histograms_normed2[0], width / 3, label='train')
rects2 = ax.bar(x2, histograms_normed2[1], width / 3, label='validation')
rects3 = ax.bar(x2 + width / 3, histograms_normed2[2], width / 3, label='test')

ax.set_ylabel('Marginal probability')
ax.set_title('Marginal distribution of tissue/cancer, excluding background')

plt.xticks(x2, labels=['tissue', 'cancer'])
ax.legend()

ax.bar_label(rects1, padding=1)
ax.bar_label(rects2, padding=1)
ax.bar_label(rects3, padding=1)

plt.savefig('hist2')