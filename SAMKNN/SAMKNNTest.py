__author__ = 'viktor'
import pandas as pd
import numpy as np
from SAMKNN import SAMKNN
from sklearn.metrics import accuracy_score
from ClassifierVisualizer import ClassifierVisualizer
from ClassifierListener import DummyClassifierListener

def run(X, y, hyperParams, visualize=False):
    if visualize:
        visualizer = ClassifierVisualizer(X, y, drawInterval=200, datasetName='Moving Squares')
    else:
        visualizer = DummyClassifierListener()
    classifier = SAMKNN(n_neighbors=hyperParams['nNeighbours'], totalWindowSize=hyperParams['windowSize'], knnWeights=hyperParams['knnWeights'],
                       STMSizeAdaption=hyperParams['STMSizeAdaption'], useLTM=hyperParams['useLTM'], listener=[visualizer])

    predictedLabels, complexity, complexityNumParameterMetric = classifier.trainOnline(X, y, np.unique(y))
    accuracy = accuracy_score(y, predictedLabels)
    print accuracy

if __name__ == '__main__':
    hyperParams ={'windowSize': 1000, 'nNeighbours': 5, 'knnWeights': 'distance', 'STMSizeAdaption': 'maxACCApprox', 'useLTM': True}
    #hyperParams = {'windowSize': 5000, 'nNeighbours': 5, 'knnWeights': 'distance', 'STMSizeAdaption': None,
    #               'useLTM': False}



    #X=pd.read_csv('../datasets/NEweather_data.csv', sep=',', header=None).values
    #y=pd.read_csv('../datasets/NEweather_class.csv', sep=',', header=None, dtype=np.int8).values.ravel()

    X = np.loadtxt('../datasets/movingSquares.data')
    y = np.loadtxt('../datasets/movingSquares.labels', dtype=np.uint8)

    run(X, y, hyperParams, visualize=False)

