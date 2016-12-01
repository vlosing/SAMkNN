
import numpy as np

class ClassifierPlotter(object):
    def __init__(self):
        pass

    def plot(self, samples, labels, fig, subplot, title, colors, XRange, YRange):

        fig.hold(True)
        # subplot = fig.add_subplot(111, aspect='equal')
        if len(labels) > 0:
            self.plotSamples(subplot, samples, labels, colors)
        subplot.set_title(title, fontsize=20)
        subplot.get_axes().xaxis.set_ticks([])
        subplot.set_xlim([XRange[0], XRange[1]])
        subplot.set_ylim([YRange[0], YRange[1]])
        subplot.get_axes().xaxis.set_ticks([])
        subplot.get_axes().yaxis.set_ticks([])

    def plotSamples(self, subplot, samples, samplesLabels, colors, size=5):
        subplot.scatter(samples[:, 0], samples[:, 1], s=size, c=colors[samplesLabels.astype(int)],
                        edgecolor=colors[samplesLabels.astype(int)])








