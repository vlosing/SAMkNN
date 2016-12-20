__author__ = 'vlosing'
class BaseClassifier(object):
    """
    Base class for classifier.
    """
    def __init__(self):
        pass

    def fit(self, samples, labels, epochs):
        raise NotImplementedError()

    def partial_fit(self, samples, labels, classes):
        raise NotImplementedError()

    def alternateFitPredict(self, samples, labels, classes):
        raise NotImplementedError()

    def predict(self, samples):
        raise NotImplementedError()

    def predict_proba(self, samples):
        raise NotImplementedError()

    def getInfos(self):
        raise NotImplementedError()

    def getComplexity(self):
        raise NotImplementedError()

    def getComplexityNumParameterMetric(self):
        raise NotImplementedError()

    def trainOnline(self, X, y, classes):
        predictedLabels = self.alternateFitPredict(X, y, classes)
        return predictedLabels, self.getComplexity(), self.getComplexityNumParameterMetric()
