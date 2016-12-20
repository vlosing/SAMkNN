__author__ = 'vlosing'
class ClassifierListener(object):
    """
    Base class for classifier listener.
    """

    def __init__(self):
        pass

    def onNewTrainStep(self, classifier, classificationResult, trainStep):
        raise NotImplementedError()

class DummyClassifierListener(ClassifierListener):
    """
    Dumy listener
    """
    def onNewTrainStep(self, classifier, classificationResult, trainStep):
        pass