__author__ = 'vlosing'
from ClassifierListener import ClassifierListener
import matplotlib.pyplot as plt
import numpy as np
import os
import time
def getClassColors():
    """
    Returns various different colors.
    """
    return np.array(['#0000FF', '#FF0000', '#00CC01', '#2F2F2F', '#8900CC', '#0099CC',
                     '#ACE600', '#D9007E', '#FFCCCC', '#5E6600', '#FFFF00', '#999999',
                     '#FF6000', '#00FF00', '#FF00FF', '#00FFFF', '#FFFF0F', '#F0CC01',
                     '#9BC6ED', '#915200',
                     '#0000FF', '#FF0000', '#00CC01', '#2F2F2F', '#8900CC', '#0099CC',
                     '#ACE600', '#D9007E', '#FFCCCC', '#5E6600', '#FFFF00', '#999999',
                     '#FF6000', '#00FF00', '#FF00FF', '#00FFFF', '#FFFF0F', '#F0CC01',
                     '#9BC6ED', '#915200'])

class ClassifierVisualizer(ClassifierListener):
    """
    Classifier visualizer implemented as listener.
    """
    DRAW_STM = True
    DRAW_LTM = True
    DRAW_FIXED_SLIDING_WINDOW = True
    FIXED_SLIDING_WINDOW_SIZE = 500
    def __init__(self, X, y, drawInterval=200, datasetName=''):
        super(ClassifierVisualizer, self).__init__()
        self.X = X
        self.y = y
        self.drawInterval = drawInterval
        self.minX = np.min(X[:,0])
        self.maxX = np.max(X[:,0])
        self.minY = np.min(X[:,1])
        self.maxY = np.max(X[:,1])
        self.datasetName = datasetName

        plt.ion()
        subplotCount = ClassifierVisualizer.DRAW_STM + ClassifierVisualizer.DRAW_LTM + ClassifierVisualizer.DRAW_FIXED_SLIDING_WINDOW
        self.fig = plt.figure(figsize=(16, 8))
        #self.subplots = self.fig.add_subplot(subplotCount*100+11, aspect='equal')
        subplotIdx = 0
        if ClassifierVisualizer.DRAW_FIXED_SLIDING_WINDOW:
            self.subplotSliding = self.fig.add_subplot(311, aspect='equal')
        if ClassifierVisualizer.DRAW_STM:
            self.subplotSTM = self.fig.add_subplot(312, aspect='equal')
        if ClassifierVisualizer.DRAW_LTM:
            self.subplotLTM = self.fig.add_subplot(313, aspect='equal')

    def draw(self, classifier, trainStep):
        self.fig.suptitle('%s #instance %d' % (self.datasetName, trainStep), fontsize=20)
        if ClassifierVisualizer.DRAW_STM:
            self.subplotSTM.clear()
            self.plot(classifier.STMSamples, classifier.STMLabels, self.fig,  self.subplotSTM,
                                        'STM size %d' % classifier.STMSamples.shape[0], getClassColors(), XRange=[self.minX, self.maxX], YRange=[self.minY, self.maxY])
        if ClassifierVisualizer.DRAW_LTM:
            self.subplotLTM.clear()
            self.plot(classifier.LTMSamples[:,:], classifier.LTMLabels[:], self.fig,  self.subplotLTM,
                                        'LTM size %d' % classifier.LTMSamples.shape[0], getClassColors(), XRange=[self.minX, self.maxX], YRange=[self.minY, self.maxY])

        if ClassifierVisualizer.DRAW_FIXED_SLIDING_WINDOW:
            self.subplotSliding.clear()
            startIdx = max(classifier.trainStepCount-ClassifierVisualizer.FIXED_SLIDING_WINDOW_SIZE, 0)
            self.plot(self.X[startIdx:trainStep, :], self.y[startIdx:trainStep], self.fig,  self.subplotSliding,
                                        'Fixed Sliding Window size %d' % ClassifierVisualizer.FIXED_SLIDING_WINDOW_SIZE, getClassColors(), XRange=[self.minX, self.maxX], YRange=[self.minY, self.maxY])
        self.fig.canvas.draw()
        plt.pause(0.001)

    def onNewTrainStep(self, classifier, classificationResult, trainStep):
        if trainStep % (self.drawInterval) == 0:
            self.draw(classifier, trainStep)

    def plot(self, samples, labels, fig, subplot, title, colors, XRange, YRange):
        fig.hold(True)
        if len(labels) > 0:
            subplot.scatter(samples[:, 0], samples[:, 1], s=10, c=colors[labels.astype(int)],
                            edgecolor=colors[labels.astype(int)])
        subplot.set_title(title, fontsize=20)
        subplot.get_axes().xaxis.set_ticks([])
        subplot.set_xlim([XRange[0], XRange[1]])
        subplot.set_ylim([YRange[0], YRange[1]])
        subplot.get_axes().xaxis.set_ticks([])
        subplot.get_axes().yaxis.set_ticks([])
