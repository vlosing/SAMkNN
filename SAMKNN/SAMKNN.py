__author__ = 'vlosing'
from BaseClassifier import BaseClassifier
import numpy as np
import libNearestNeighbor
from sklearn.cluster import KMeans
from collections import deque
import logging

class SAMKNN(BaseClassifier):
    """
    Self Adjusting Memory (SAM) coupled with the k Nearest Neighbor classifier.

    Parameters
    ----------
    n_neighbors : int, optional (default=5)
        number of evaluated nearest neighbors.
    knnWeights: string, optional (default='distance')
        Type of weighting of the nearest neighbors.
         It must be either 'distance' or 'uniform' (majority voting).
    maxSize : int, optional (default=5000)
         Maximum number of overall stored data points.
    LTMSizeProportion: float, optional (default=0.4)
        Proportion of the overall instances that may be used for the LTM. This is only relevant when the maximum number(maxSize)
        of stored instances is reached.
    RecalculateSTMError : boolean, optional (default=False)
        "If true than the error rate of the STM for size adaption is always recalculated to be precise(Costly operation).
        Otherwise, an incremental approximation is used."
        If set to None, the STM is not adapted at all. When additionally useLTM=false, this algorithm is simply a kNN with fixed sliding window size.
    minSTMSize : int, optional (default=50)
        Minimum STM size which is evaluated during the STM size adaption.
    useLTM : boolean, optional (default=True)
        Specifies whether the LTM should be used at all.
    """
    def __init__(self, n_neighbors=5, knnWeights='distance', maxSize=5000, LTMSizeProportion = 0.4, minSTMSize=50, recalculateSTMError=False, useLTM=True, listener=[]):

        self.n_neighbors = n_neighbors
        self._STMSamples = None
        self._STMLabels = np.empty(shape=(0), dtype=np.int32)
        self._LTMSamples = None
        self._LTMLabels = np.empty(shape=(0), dtype=np.int32)
        self.maxLTMSize = LTMSizeProportion * maxSize
        self.maxSTMSize = maxSize - self.maxLTMSize
        self.minSTMSize = minSTMSize

        self.recalculateSTMError = recalculateSTMError
        if recalculateSTMError is not None:
            self.STMDistances = np.zeros(shape=(maxSize + 1, maxSize + 1))
        if knnWeights == 'distance':
            self.getLabelsFct = SAMKNN.getDistanceWeightedLabel
        elif knnWeights == 'uniform':
            self.getLabelsFct = SAMKNN.getMajLabel
        self.useLTM = useLTM
        if useLTM:
            self.predictFct = self.predictByAllMemories
            self.sizeCheckFct = self.sizeCheckSTMLTM
        else:
            self.predictFct = self.predictBySTM
            self.sizeCheckFct = self.sizeCheckFadeOut

        self.interLeavedPredHistories = {}
        self.LTMPredHistory = deque([])
        self.STMPredHistory = deque([])
        self.CMPredHistory = deque([])
        self.listener = listener


        self.trainStepCount = 0
        self.STMSizes = []
        self.LTMSizes = []
        self.numSTMCorrect = 0
        self.numLTMCorrect = 0
        self.numCMCorrect = 0
        self.numPossibleCorrectPredictions = 0
        self.numCorrectPredictions = 0
        self.classifierChoice = []
        self.predHistory = []

    def getClassifier(self):
        return None

    def getInfos(self):
        return ''

    def predict(self, samples):
        if self._STMSamples is None:
            return np.zeros(shape=samples.shape[0]).astype(np.int32)
        else:
            predictedLabels = []
            for sample in samples:
                distancesSTM = SAMKNN.getDistances(sample, self._STMSamples)
                predictedLabels.append(self.predictFct(sample, -1, distancesSTM))
            return np.array(predictedLabels)

    @staticmethod
    def getDistances(sample, samples):
        """Calculate distances from sample to all samples."""
        return np.sqrt(libNearestNeighbor.get1ToNDistances(sample, samples))

    def clusterDown(self, samples, labels):
        """Performs classwise kMeans++ clustering for given samples with corresponding labels. The number of samples is halved per class."""
        logging.debug('cluster Down %d' % self.trainStepCount)
        uniqueLabels = np.unique(labels)
        newSamples = np.empty(shape=(0, samples.shape[1]))
        newLabels = np.empty(shape=(0), dtype=np.int32)
        for label in uniqueLabels:
            tmpSamples = samples[labels == label]
            newLength = int(max(tmpSamples.shape[0]/2, 1))
            clustering = KMeans(n_clusters=newLength, n_init=1, random_state=0)
            clustering.fit(tmpSamples)
            newSamples = np.vstack([newSamples, clustering.cluster_centers_])
            newLabels = np.append(newLabels, label*np.ones(shape=newLength, dtype=np.int32))
        return newSamples, newLabels

    def sizeCheckFadeOut(self):
        """Makes sure that the STM does not surpass the maximum size, only used when useLTM=False."""
        STMShortened = False
        if len(self._STMLabels) > self.maxSTMSize + self.maxLTMSize:
            STMShortened = True
            self._STMSamples = np.delete(self._STMSamples, 0, 0)
            self._STMLabels = np.delete(self._STMLabels, 0, 0)
            if self.recalculateSTMError is not None:
                self.STMDistances[:len(self._STMLabels), :len(self._STMLabels)] = self.STMDistances[1:len(self._STMLabels) + 1, 1:len(self._STMLabels) + 1]

            if not self.recalculateSTMError:
                if 0 in self.interLeavedPredHistories:
                    self.interLeavedPredHistories[0].pop(0)
                for key in list(self.interLeavedPredHistories.keys()):
                    if key > 0:
                        if key == 1:
                            self.interLeavedPredHistories.pop(0, None)
                        tmp = self.interLeavedPredHistories[key]
                        self.interLeavedPredHistories.pop(key, None)
                        self.interLeavedPredHistories[key-1] = tmp
            else:
                self.interLeavedPredHistories = {}
        return STMShortened

    def sizeCheckSTMLTM(self):
        """Makes sure that the STM and LTM combined doe not surpass the maximum size, only used when useLTM=True."""
        STMShortened = False
        if len(self._STMLabels) + len(self._LTMLabels) > self.maxSTMSize + self.maxLTMSize:
            if len(self._LTMLabels) > self.maxLTMSize:
                self._LTMSamples, self._LTMLabels = self.clusterDown(self._LTMSamples, self._LTMLabels)
            else:
                if len(self._STMLabels) + len(self._LTMLabels) > self.maxSTMSize + self.maxLTMSize:
                    STMShortened = True
                    numShifts = int(self.maxLTMSize - len(self._LTMLabels) + 1)
                    shiftRange = range(numShifts)
                    self._LTMSamples = np.vstack([self._LTMSamples, self._STMSamples[:numShifts, :]])
                    self._LTMLabels = np.append(self._LTMLabels, self._STMLabels[:numShifts])
                    self._LTMSamples, self._LTMLabels = self.clusterDown(self._LTMSamples, self._LTMLabels)
                    self._STMSamples = np.delete(self._STMSamples, shiftRange, 0)
                    self._STMLabels = np.delete(self._STMLabels, shiftRange, 0)
                    self.STMDistances[:len(self._STMLabels),:len(self._STMLabels)] = self.STMDistances[numShifts:len(self._STMLabels)+numShifts, numShifts:len(self._STMLabels)+numShifts]
                    for i in shiftRange:
                        self.LTMPredHistory.popleft()
                        self.STMPredHistory.popleft()
                        self.CMPredHistory.popleft()
                    self.interLeavedPredHistories = {}
        return STMShortened

    def cleanSamples(self, samplesCl, labelsCl, onlyLast=False):
        """Removes distance-based all instances from the input samples that contradict those in the STM."""
        if self._STMLabels.shape[0] > self.n_neighbors and samplesCl.shape[0] > 0:
            if onlyLast:
                loopRange = [len(self._STMLabels)-1]
            else:
                loopRange = range(len(self._STMLabels))
            for i in loopRange:
                if len(labelsCl) == 0:
                    break
                samplesShortened = np.delete(self._STMSamples, i, 0)
                labelsShortened = np.delete(self._STMLabels, i, 0)
                distancesSTM = SAMKNN.getDistances(self._STMSamples[i,:], samplesShortened)
                nnIndicesSTM = libNearestNeighbor.nArgMin(self.n_neighbors, distancesSTM)[0]
                distancesLTM = SAMKNN.getDistances(self._STMSamples[i,:], samplesCl)
                nnIndicesLTM = libNearestNeighbor.nArgMin(min(len(distancesLTM), self.n_neighbors), distancesLTM)[0]
                correctIndicesSTM = nnIndicesSTM[labelsShortened[nnIndicesSTM] == self._STMLabels[i]]
                if len(correctIndicesSTM) > 0:
                    distThreshold = np.max(distancesSTM[correctIndicesSTM])
                    wrongIndicesLTM = nnIndicesLTM[labelsCl[nnIndicesLTM] != self._STMLabels[i]]
                    delIndices = np.where(distancesLTM[wrongIndicesLTM] <= distThreshold)[0]
                    samplesCl = np.delete(samplesCl, wrongIndicesLTM[delIndices], 0)
                    labelsCl = np.delete(labelsCl, wrongIndicesLTM[delIndices], 0)
        return samplesCl, labelsCl

    def singleFit(self, sample, sampleLabel, distancesSTM):
        if self._STMSamples is None:
            self._STMSamples = np.empty(shape=(0, sample.shape[0]))
            self._LTMSamples = np.empty(shape=(0, sample.shape[0]))

        self.trainStepCount += 1
        self._STMSamples = np.vstack([self._STMSamples, sample])
        self._STMLabels = np.append(self._STMLabels, sampleLabel)
        STMShortened = self.sizeCheckFct()


        self._LTMSamples, self._LTMLabels = self.cleanSamples(self._LTMSamples, self._LTMLabels, onlyLast=True)

        if self.recalculateSTMError is not None:
            if STMShortened:
                distancesSTM = SAMKNN.getDistances(sample, self._STMSamples[:-1,:])

            self.STMDistances[len(self._STMLabels)-1,:len(self._STMLabels)-1] = distancesSTM
            oldWindowSize = len(self._STMLabels)
            newWindowSize, self.interLeavedPredHistories = STMSizer.getNewSTMSize(self.recalculateSTMError, self._STMLabels, self.n_neighbors, self.getLabelsFct, self.interLeavedPredHistories, self.STMDistances, self.minSTMSize)

            if newWindowSize < oldWindowSize:
                delrange = range(oldWindowSize-newWindowSize)
                oldSTMSamples = self._STMSamples[delrange, :]
                oldSTMLabels = self._STMLabels[delrange]
                self._STMSamples = np.delete(self._STMSamples, delrange, 0)
                self._STMLabels = np.delete(self._STMLabels, delrange, 0)
                self.STMDistances[:len(self._STMLabels),:len(self._STMLabels)] = self.STMDistances[(oldWindowSize-newWindowSize):(oldWindowSize-newWindowSize)+len(self._STMLabels),(oldWindowSize-newWindowSize):(oldWindowSize-newWindowSize)+len(self._STMLabels)]

                if self.useLTM:
                    for i in delrange:
                        self.STMPredHistory.popleft()
                        self.LTMPredHistory.popleft()
                        self.CMPredHistory.popleft()

                    oldSTMSamples, oldSTMLabels = self.cleanSamples(oldSTMSamples, oldSTMLabels)
                    self._LTMSamples = np.vstack([self._LTMSamples, oldSTMSamples])
                    self._LTMLabels = np.append(self._LTMLabels, oldSTMLabels)
                    self.sizeCheckFct()
        self.STMSizes.append(len(self._STMLabels))
        self.LTMSizes.append(len(self._LTMLabels))
        for listener in self.listener:
            listener.onNewTrainStep(self, False, self.trainStepCount)

    def getSTMDistances(self, sample):
        if self._STMSamples is not None:
            return SAMKNN.getDistances(sample, self._STMSamples)
        else:
            return None


    def _partial_fit(self, sample, sampleLabel):
        """Processes a new sample."""
        distancesSTM = SAMKNN.getDistances(sample, self._STMSamples)
        predictedLabel = self.predictFct(sample, sampleLabel, distancesSTM)
        self.singleFit(sample, sampleLabel, distancesSTM)

        return predictedLabel

    def predictByAllMemories(self, sample, label, distancesSTM):
        """Predicts the label of a given sample by using the STM, LTM and the CM, only used when useLTM=True."""
        predictedLabelLTM = 0
        predictedLabelSTM = 0
        predictedLabelCM = 0
        classifierChoice = 0
        predictedLabel = 0
        if len(self._STMLabels) > 0:
            predictedLabelSTM = self.getLabelsFct(distancesSTM, self._STMLabels, min(len(self._STMLabels), self.n_neighbors))[0]
            distancesLTM = SAMKNN.getDistances(sample, self._LTMSamples)
            predictedLabelCM = self.getLabelsFct(np.append(distancesSTM, distancesLTM), np.append(self._STMLabels, self._LTMLabels), min(len(self._STMLabels) + len(self._LTMLabels), self.n_neighbors))[0]
            if len(self._LTMLabels) > 0:
                predictedLabelLTM = self.getLabelsFct(distancesLTM, self._LTMLabels, min(len(self._LTMLabels), self.n_neighbors))[0]

            labels = [predictedLabelSTM, predictedLabelLTM, predictedLabelCM]
            correctSTM = np.sum(self.STMPredHistory)
            correctLTM = np.sum(self.LTMPredHistory)
            correctCM = np.sum(self.CMPredHistory)
            classifierChoice = np.argmax([correctSTM, correctLTM, correctCM])
            predictedLabel = labels[classifierChoice]
        

        self.classifierChoice.append(classifierChoice)
        self.CMPredHistory.append(predictedLabelCM == label)
        self.numCMCorrect += predictedLabelCM == label
        self.STMPredHistory.append(predictedLabelSTM == label)
        self.numSTMCorrect += predictedLabelSTM == label
        self.LTMPredHistory.append(predictedLabelLTM == label)
        self.numLTMCorrect += predictedLabelLTM == label
        self.numPossibleCorrectPredictions += label in [predictedLabelSTM, predictedLabelCM, predictedLabelLTM]
        self.numCorrectPredictions += predictedLabel == label
        return predictedLabel

    def predictBySTM(self, sample, label, distancesSTM):
        """Predicts the label of a given sample by the STM, only used when useLTM=False."""
        predictedLabel = 0
        currLen = len(self._STMLabels)
        if currLen > 0:
            predictedLabel = self.getLabelsFct(distancesSTM, self._STMLabels, min(self.n_neighbors, currLen))[0]
        return predictedLabel

    def partial_fit(self, samples, labels, classes):
        _labels = labels.astype(np.int32)
        """Processes a new sample."""
        if self._STMSamples is None:
            self._STMSamples = np.empty(shape=(0, samples.shape[1]))
            self._LTMSamples = np.empty(shape=(0, samples.shape[1]))

        predictedLabels = []
        for i in range(samples.shape[0]):
            predictedLabels.append(self._partial_fit(samples[i, :], _labels[i]))
        return predictedLabels

    def alternateFitPredict(self, samples, labels, classes):
        """Processes all samples in the default online setting (first predict than use for training)."""
        if self._STMSamples is None:
            self._STMSamples = np.empty(shape=(0, samples.shape[1]))
            self._LTMSamples = np.empty(shape=(0, samples.shape[1]))
        predictedTrainLabels = []
        _labels = labels.astype(np.int32)
        for i in range(len(labels)):
            if (i+1)%(len(labels)/20) == 0:
                logging.info('%d%%' % int(np.round((i+1.)/len(labels)*100, 0)))
            predictedTrainLabels.append(self._partial_fit(samples[i, :], _labels[i]))
        return predictedTrainLabels

    @staticmethod
    def getMajLabel(distances, labels, numNeighbours):
        """Returns the majority label of the k nearest neighbors."""
        nnIndices = libNearestNeighbor.nArgMin(numNeighbours, distances)
        predLabels = libNearestNeighbor.mostCommon(labels[nnIndices])
        return predLabels

    @staticmethod
    def getDistanceWeightedLabel(distances, labels, numNeighbours):
        """Returns the the distance weighted label of the k nearest neighbors."""
        nnIndices = libNearestNeighbor.nArgMin(numNeighbours, distances)
        predLabels = libNearestNeighbor.getLinearWeightedLabels(labels[nnIndices], distances[nnIndices])
        return predLabels

    def getComplexity(self):
        return 0

    def getComplexityNumParameterMetric(self):
        return 0

    @property
    def STMSamples(self):
        return self._STMSamples

    @property
    def STMLabels(self):
        return self._STMLabels

    @property
    def LTMSamples(self):
        return self._LTMSamples

    @property
    def LTMLabels(self):
        return self._LTMLabels

    def getStatistics(self):
        result = ''
        result += 'avg. STMSize %f LTMSize %f' % (np.mean(self.STMSizes), np.mean(self.LTMSizes)) + '\n'
        result += 'num correct STM %d LTM %d CM %d ' % (self.numSTMCorrect, self.numLTMCorrect, self.numCMCorrect) + '\n'
        result += 'num correct %d/%d' % (self.numCorrectPredictions, self.numPossibleCorrectPredictions) + '\n'
        return result

class STMSizer(object):
    """Utility class to adapt the size of the sliding window of the STM."""
    @staticmethod
    def getNewSTMSize(recalculateSTMError, labels, nNeighbours, getLabelsFct, predictionHistories, distancesSTM, minSTMSize):
        """Returns the new STM size."""
        if recalculateSTMError is None:
            return len(labels), predictionHistories
        elif recalculateSTMError:
            return STMSizer.getMinErrorRateWindowSize(labels, nNeighbours, getLabelsFct, predictionHistories, distancesSTM, minSize=minSTMSize)
        else:
            return STMSizer.getMinErrorRateWindowSizeIncremental(labels, nNeighbours, getLabelsFct, predictionHistories, distancesSTM, minSize=minSTMSize)

    @staticmethod
    def errorRate(predLabels, labels):
        """Calculates the achieved error rate."""
        return 1 - np.sum(predLabels == labels)/float(len(predLabels))

    @staticmethod
    def getInterleavedTestTrainErrorRate(labels, nNeighbours, getLabelsFct, distancesSTM):
        """Calculates the interleaved test train error rate from the scratch."""
        predLabels = []
        for i in range(nNeighbours, len(labels)):
            distances = distancesSTM[i, :i]
            predLabels.append(getLabelsFct(distances, labels[:i], nNeighbours)[0])
        return STMSizer.errorRate(predLabels[:], labels[nNeighbours:]), (predLabels == labels[nNeighbours:]).tolist()

    @staticmethod
    def getIncrementalInterleavedTestTrainErrorRate(labels, nNeighbours, getLabelsFct, predictionHistory, distancesSTM):
        """Calculates the interleaved test train error rate incrementally by using the previous predictions."""
        for i in range(len(predictionHistory) + nNeighbours, len(labels)):
            distances = distancesSTM[i, :i]
            label = getLabelsFct(distances, labels[:i], nNeighbours)[0]
            predictionHistory.append(label == labels[i])
        return 1 - np.sum(predictionHistory)/float(len(predictionHistory)), predictionHistory

    @staticmethod
    def adaptHistories(numberOfDeletions, predictionHistories):
        """Removes predictions of the largest window size and shifts the remaining ones accordingly."""
        for i in range(numberOfDeletions):
            sortedKeys = np.sort(list(predictionHistories.keys()))
            predictionHistories.pop(sortedKeys[0], None)
            delta = sortedKeys[1]
            for j in range(1, len(sortedKeys)):
                predictionHistories[sortedKeys[j]- delta] = predictionHistories.pop(sortedKeys[j])
        return predictionHistories

    @staticmethod
    def getMinErrorRateWindowSize(labels, nNeighbours, getLabelsFct, predictionHistories, distancesSTM, minSize=50):
        """Returns the window size with the minimum Interleaved test-train error(exact calculation)."""
        numSamples = len(labels)
        if numSamples < 2 * minSize:
            return numSamples, predictionHistories
        else:
            numSamplesRange = [numSamples]
            while numSamplesRange[-1]/2 >= minSize:
                numSamplesRange.append(numSamplesRange[-1]/2)

            errorRates = []
            for key in list(predictionHistories.keys()):
                if key not in (numSamples - np.array(numSamplesRange)):
                    predictionHistories.pop(key, None)

            for numSamplesIt in numSamplesRange:
                idx = int(numSamples - numSamplesIt)
                if idx in predictionHistories:
                    errorRate, predHistory = STMSizer.getIncrementalInterleavedTestTrainErrorRate(labels[idx:], nNeighbours, getLabelsFct, predictionHistories[idx], distancesSTM[idx:, idx:])
                else:
                    errorRate, predHistory = STMSizer.getInterleavedTestTrainErrorRate(labels[idx:], nNeighbours, getLabelsFct, distancesSTM[idx:, idx:])
                predictionHistories[idx] = predHistory
                errorRates.append(errorRate)

            errorRates = np.round(errorRates, decimals=4)
            bestNumTrainIdx = np.argmin(errorRates)
            windowSize = numSamplesRange[bestNumTrainIdx]
            if windowSize < numSamples:
                predictionHistories = STMSizer.adaptHistories(bestNumTrainIdx, predictionHistories)
            return int(windowSize), predictionHistories

    @staticmethod
    def getMinErrorRateWindowSizeIncremental(labels, nNeighbours, getLabelsFct, predictionHistories, distancesSTM, minSize=50):
        """Returns the window size with the minimum Interleaved test-train error(using an incremental approximation)."""
        numSamples = len(labels)
        if numSamples < 2 * minSize:
            return numSamples, predictionHistories
        else:
            numSamplesRange = [numSamples]
            while numSamplesRange[-1]/2 >= minSize:
                numSamplesRange.append(numSamplesRange[-1]/2)
            errorRates = []
            for numSamplesIt in numSamplesRange:
                idx = int(numSamples - numSamplesIt)
                if idx in predictionHistories:
                    errorRate, predHistory = STMSizer.getIncrementalInterleavedTestTrainErrorRate(labels[idx:], nNeighbours, getLabelsFct, predictionHistories[idx], distancesSTM[idx:, idx:])
                elif idx-1 in predictionHistories:
                    predHistory = predictionHistories[idx-1]
                    predictionHistories.pop(idx-1, None)
                    predHistory.pop(0)
                    errorRate, predHistory = STMSizer.getIncrementalInterleavedTestTrainErrorRate(labels[idx:], nNeighbours, getLabelsFct, predHistory, distancesSTM[idx:, idx:])
                else:
                    errorRate, predHistory = STMSizer.getInterleavedTestTrainErrorRate(labels[idx:], nNeighbours, getLabelsFct, distancesSTM[idx:, idx:])
                predictionHistories[idx] = predHistory
                errorRates.append(errorRate)
            errorRates = np.round(errorRates, decimals=4)
            bestNumTrainIdx = np.argmin(errorRates)
            if bestNumTrainIdx > 0:
                moreAccurateIndices = np.where(errorRates < errorRates[0])[0]
                for i in moreAccurateIndices:
                    idx = int(numSamples - numSamplesRange[i])
                    errorRate, predHistory = STMSizer.getInterleavedTestTrainErrorRate(labels[idx:], nNeighbours, getLabelsFct, distancesSTM[idx:, idx:])
                    predictionHistories[idx] = predHistory
                    errorRates[i] = errorRate
                errorRates = np.round(errorRates, decimals=4)
                bestNumTrainIdx = np.argmin(errorRates)
            windowSize = numSamplesRange[bestNumTrainIdx]

            if windowSize < numSamples:
                predictionHistories = STMSizer.adaptHistories(bestNumTrainIdx, predictionHistories)
            return int(windowSize), predictionHistories