__author__ = 'vlosing'
class ClassifierListener(object):
    def __init__(self):
        pass

    def onNewTrainStep(self, classifier, classificationResult, trainStep):
        raise NotImplementedError()

class DummyClassifierListener(ClassifierListener):
    def onNewTrainStep(self, classifier, classificationResult, trainStep):
        pass