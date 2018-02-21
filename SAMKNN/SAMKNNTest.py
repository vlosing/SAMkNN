__author__ = 'viktor'
import pandas as pd
import numpy as np
from SAMKNN import SAMKNN
from sklearn.metrics import accuracy_score
from ClassifierVisualizer import ClassifierVisualizer
from ClassifierListener import DummyClassifierListener
import logging

def run(X, y, hyperParams, visualize=False):
    """
    Test function for SAMKNN
    """
    if visualize:
        visualizer = ClassifierVisualizer(X, y, drawInterval=200, datasetName='Moving Squares')
    else:
        visualizer = DummyClassifierListener()
    classifier = SAMKNN(n_neighbors=hyperParams['nNeighbours'], maxSize=hyperParams['maxSize'], knnWeights=hyperParams['knnWeights'],
                        recalculateSTMError=hyperParams['recalculateSTMError'], useLTM=hyperParams['useLTM'], listener=[visualizer])

    logging.info('applying model on dataset')
    predictedLabels, complexity, complexityNumParameterMetric = classifier.trainOnline(X, y, np.unique(y))
    accuracy = accuracy_score(y, predictedLabels)
    logging.info('error rate %.2f%%' % (100-100*accuracy))

if __name__ == '__main__':
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    hyperParams ={'maxSize': 1000, 'nNeighbours': 5, 'knnWeights': 'distance', 'recalculateSTMError': False, 'useLTM': True}
    #hyperParams = {'windowSize': 5000, 'nNeighbours': 5, 'knnWeights': 'distance', 'STMSizeAdaption': None,
    #               'useLTM': False}


    logging.info('loading dataset')
    #X=pd.read_csv('../datasets/NEweather_data.csv', sep=',', header=None).values
    #y=pd.read_csv('../datasets/NEweather_class.csv', sep=',', header=None, dtype=np.int8).values.ravel()
    X = np.loadtxt('../datasets/movingSquares.data')
    y = np.loadtxt('../datasets/movingSquares.labels', dtype=np.uint8)
    logging.info('%d samples' % X.shape[0])
    logging.info('%d dimensions' % X.shape[1])
    run(X, y, hyperParams, visualize=False)

