__author__ = 'vlosing'
from BaseClassifier import BaseClassifier
import numpy as np
import libNNPythonIntf
from sklearn.cluster import KMeans
from collections import deque
import logging

class SAMKNN(BaseClassifier):
    def __init__(self, n_neighbors=5, totalWindowSize=200, LTMSizeProportion = 0.4, minSTMSize=50, STMSizeAdaption=None, useLTM=True, knnWeights='distance', listener=[]):
        self.n_neighbors = n_neighbors
        self._STMSamples = None
        self._STMLabels = np.empty(shape=(0), dtype=np.int32)
        self._LTMSamples = None
        self._LTMLabels = np.empty(shape=(0), dtype=np.int32)
        self.maxLTMSize = LTMSizeProportion * totalWindowSize
        self.maxSTMSize = totalWindowSize - self.maxLTMSize
        self.minSTMSize = minSTMSize

        if STMSizeAdaption is not None:
            self.STMDistances = np.zeros(shape=(totalWindowSize+1,totalWindowSize+1))
        if knnWeights == 'distance':
            self.getLabelsFct = SAMKNN.getDistanceWeightedLabels
        elif knnWeights == 'uniform':
            self.getLabelsFct = SAMKNN.getMajLabels
        self.STMSizeAdaption = STMSizeAdaption
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

    @staticmethod
    def getDistances(sample, samples):
        return libNNPythonIntf.get1ToNDistances(sample, samples)

    def clusterDown(self, samples, labels):
        logging.info('cluster Down %d' % self.trainStepCount)
        uniqueLabels = np.unique(labels)
        newSamples = np.empty(shape=(0, samples.shape[1]))
        newLabels = np.empty(shape=(0), dtype=np.int32)
        for label in uniqueLabels:
            tmpSamples = samples[labels == label]
            newLength = max(tmpSamples.shape[0]/2, 1)
            clustering = KMeans(n_clusters=newLength, n_init=1, random_state=0)
            clustering.fit(tmpSamples)
            newSamples = np.vstack([newSamples, clustering.cluster_centers_])
            newLabels = np.append(newLabels, label*np.ones(shape=newLength, dtype=np.int32))
        return newSamples, newLabels

    def sizeCheckFadeOut(self):
        STMShortened = False
        if len(self._STMLabels) > self.maxSTMSize + self.maxLTMSize:
            STMShortened = True
            self._STMSamples = np.delete(self._STMSamples, 0, 0)
            self._STMLabels = np.delete(self._STMLabels, 0, 0)
            self.STMDistances[:len(self._STMLabels), :len(self._STMLabels)] = self.STMDistances[1:len(self._STMLabels) + 1, 1:len(self._STMLabels) + 1]

            if self.STMSizeAdaption == 'maxACCApprox':
                if self.interLeavedPredHistories.has_key(0):
                    self.interLeavedPredHistories[0].pop(0)
                for key in self.interLeavedPredHistories.keys():
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
        STMShortened = False
        if len(self._STMLabels) + len(self._LTMLabels) > self.maxSTMSize + self.maxLTMSize:
            if len(self._LTMLabels) > self.maxLTMSize:
                self._LTMSamples, self._LTMLabels = self.clusterDown(self._LTMSamples, self._LTMLabels)
            else:
                if len(self._STMLabels) + len(self._LTMLabels) > self.maxSTMSize + self.maxLTMSize:
                    STMShortened = True
                    numShifts = int(self.maxLTMSize - len(self._LTMLabels) + 1)
                    shiftRange = xrange(numShifts)
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

    def validateSamples(self, samplesCl, labelsCl, onlyLast=False):
        if len(self._STMLabels) > self.n_neighbors and samplesCl.shape[0] > 0:
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
                nnIndicesSTM = libNNPythonIntf.nArgMin(self.n_neighbors, distancesSTM)[0]
                distancesLTM = SAMKNN.getDistances(self._STMSamples[i,:], samplesCl)
                nnIndicesLTM = libNNPythonIntf.nArgMin(min(len(distancesLTM), self.n_neighbors), distancesLTM)[0]
                correctIndicesSTM = nnIndicesSTM[labelsShortened[nnIndicesSTM] == self._STMLabels[i]]
                if len(correctIndicesSTM) > 0:
                    distThreshold = np.max(distancesSTM[correctIndicesSTM])
                    wrongIndicesLTM = nnIndicesLTM[labelsCl[nnIndicesLTM] != self._STMLabels[i]]
                    delIndices = np.where(distancesLTM[wrongIndicesLTM] <= distThreshold)[0]
                    samplesCl = np.delete(samplesCl, wrongIndicesLTM[delIndices], 0)
                    labelsCl = np.delete(labelsCl, wrongIndicesLTM[delIndices], 0)
        return samplesCl, labelsCl

    def _partial_fit(self, sample, sampleLabel):
        distancesSTM = SAMKNN.getDistances(sample, self._STMSamples)
        predictedLabel = self.predictFct(sample, sampleLabel, distancesSTM)

        self.trainStepCount += 1
        self._STMSamples = np.vstack([self._STMSamples, sample])
        self._STMLabels = np.append(self._STMLabels, sampleLabel)
        STMShortened = self.sizeCheckFct()


        self._LTMSamples, self._LTMLabels = self.validateSamples(self._LTMSamples, self._LTMLabels, onlyLast=True)

        if self.STMSizeAdaption is not None:
            if STMShortened:
                distancesSTM = SAMKNN.getDistances(sample, self._STMSamples[:-1,:])

            self.STMDistances[len(self._STMLabels)-1,:len(self._STMLabels)-1] = distancesSTM
            oldWindowSize = len(self._STMLabels)
            newWindowSize, self.interLeavedPredHistories = STMSizer.getWindowSize(self.STMSizeAdaption,  self._STMLabels, self.n_neighbors, self.getLabelsFct, self.interLeavedPredHistories, self.STMDistances, self.minSTMSize)

            if newWindowSize < oldWindowSize:
                delrange = xrange(oldWindowSize-newWindowSize)
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

                    oldSTMSamples, oldSTMLabels = self.validateSamples(oldSTMSamples, oldSTMLabels)
                    self._LTMSamples = np.vstack([self._LTMSamples, oldSTMSamples])
                    self._LTMLabels = np.append(self._LTMLabels, oldSTMLabels)
                    self.sizeCheckFct()
        self.STMSizes.append(len(self._STMLabels))
        self.LTMSizes.append(len(self._LTMLabels))
        for listener in self.listener:
            listener.onNewTrainStep(self, False, self.trainStepCount)
        return predictedLabel

    def predictByAllMemories(self, sample, label, distancesSTM):
        predictedLabelLTM = 0
        predictedLabelSTM = 0
        predictedLabelBoth = 0
        classifierChoice = 0
        if len(self._STMLabels) == 0:
            predictedLabel = predictedLabelSTM
        else:
            if len(self._STMLabels) < self.n_neighbors:
                predictedLabelSTM = self.getLabelsFct(distancesSTM, self._STMLabels, len(self._STMLabels))[0]
                predictedLabel = predictedLabelSTM
            else:
                distancesLTM = SAMKNN.getDistances(sample, self._LTMSamples)
                predictedLabelSTM = self.getLabelsFct(distancesSTM, self._STMLabels, self.n_neighbors)[0]
                predictedLabelBoth = self.getLabelsFct(np.append(distancesSTM, distancesLTM), np.append(self._STMLabels, self._LTMLabels), self.n_neighbors)[0]

                if len(self._LTMLabels) >= self.n_neighbors:
                    predictedLabelLTM = self.getLabelsFct(distancesLTM, self._LTMLabels, self.n_neighbors)[0]
                    correctLTM = np.sum(self.LTMPredHistory)
                    correctSTM = np.sum(self.STMPredHistory)
                    correctBoth = np.sum(self.CMPredHistory)
                    labels = [predictedLabelSTM, predictedLabelLTM, predictedLabelBoth]
                    classifierChoice = np.argmax([correctSTM, correctLTM, correctBoth])
                    predictedLabel = labels[classifierChoice]
                else:
                    predictedLabel = predictedLabelSTM

        self.classifierChoice.append(classifierChoice)
        self.CMPredHistory.append(predictedLabelBoth == label)
        self.numCMCorrect += predictedLabelBoth == label
        self.STMPredHistory.append(predictedLabelSTM == label)
        self.numSTMCorrect += predictedLabelSTM == label
        self.LTMPredHistory.append(predictedLabelLTM == label)
        self.numLTMCorrect += predictedLabelLTM == label
        self.numPossibleCorrectPredictions += label in [predictedLabelSTM, predictedLabelBoth, predictedLabelLTM]
        self.numCorrectPredictions += predictedLabel == label
        return predictedLabel

    def predictBySTM(self, sample, label, distancesSTM):
        predictedLabel = 0
        currLen = len(self._STMLabels)
        if currLen > 0:
            predictedLabel = self.getLabelsFct(distancesSTM, self._STMLabels, min(self.n_neighbors, currLen))[0]
        return predictedLabel

    def partial_fit(self, samples, labels, classes):
        if self._STMSamples is None:
            self._STMSamples = np.empty(shape=(0, samples.shape[1]))
            self._LTMSamples = np.empty(shape=(0, samples.shape[1]))

        predictedLabels = []
        for i in range(len(samples)):
            predictedLabels.append(self._partial_fit(samples[i, :], labels[i]))
        return predictedLabels

    def alternateFitPredict(self, samples, labels, classes):
        if self._STMSamples is None:
            self._STMSamples = np.empty(shape=(0, samples.shape[1]))
            self._LTMSamples = np.empty(shape=(0, samples.shape[1]))
        predictedTrainLabels = []
        for i in range(len(labels)):
            if (i+1)%(len(labels)/20) == 0:
                print '%d%%' % int(np.round((i+1.)/len(labels)*100, 0))
            predictedTrainLabels.append(self._partial_fit(samples[i, :], labels[i]))
        return predictedTrainLabels

    @staticmethod
    def getMajLabels(distances, labels, numNeighbours):
        nnIndices = libNNPythonIntf.nArgMin(numNeighbours, distances)
        predLabels = libNNPythonIntf.mostCommon(labels[nnIndices])
        return predLabels

    @staticmethod
    def getDistanceWeightedLabels(distances, labels, numNeighbours):
        nnIndices = libNNPythonIntf.nArgMin(numNeighbours, distances)
        sqrtDistances = np.sqrt(distances[nnIndices])
        predLabels = libNNPythonIntf.getLinearWeightedLabels(labels[nnIndices], sqrtDistances)
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
    accCalcCount = 0
    nnCalcCount = 0
    notShrinkedCount = 0
    shrinkedCount = 0
    currentSizeAccs = []
    smallestSizeAccs = []
    largestSizeAccs = []
    @staticmethod
    def getWindowSize(adaptionStrategy, labels, nNeighbours, getLabelsFct, predictionHistories, distancesSTM, minSTMSize):
        if adaptionStrategy is None:
            return len(labels), predictionHistories
        elif adaptionStrategy == 'maxACC':
            return STMSizer.getMaxAccWindowSize(labels, nNeighbours, getLabelsFct, predictionHistories, distancesSTM, minSize=minSTMSize)
        elif adaptionStrategy == 'maxACCApprox':
            return STMSizer.getMaxAccApproxWindowSize(labels, nNeighbours, getLabelsFct, predictionHistories, distancesSTM, minSize=minSTMSize)
        else:
            raise Exception('unknown driftStrategy')

    @staticmethod
    def accScore(predLabels, labels):
        return np.sum(predLabels == labels)/float(len(predLabels))

    @staticmethod
    def getInterleavedTestTrainAcc(labels, nNeighbours, getLabelsFct, distancesSTM):
        predLabels = []
        for i in range(nNeighbours, len(labels)):
            distances = distancesSTM[i, :i]
            predLabels.append(getLabelsFct(distances, labels[:i], nNeighbours)[0])
            STMSizer.nnCalcCount += len(labels[:i])
        return STMSizer.accScore(predLabels[:], labels[nNeighbours:]), (predLabels == labels[nNeighbours:]).tolist()

    @staticmethod
    def getInterleavedTestTrainAccPredHistory(labels, nNeighbours, getLabelsFct, predictionHistory, distancesSTM):
        for i in range(len(predictionHistory) + nNeighbours, len(labels)):
            distances = distancesSTM[i, :i]
            label = getLabelsFct(distances, labels[:i], nNeighbours)[0]
            predictionHistory.append(label == labels[i])
            STMSizer.nnCalcCount += len(labels[:i])
        return np.sum(predictionHistory)/float(len(predictionHistory)), predictionHistory

    @staticmethod
    def adaptHistories(numberOfDeletions, predictionHistories):
        for i in range(numberOfDeletions):
            sortedKeys = np.sort(predictionHistories.keys())
            predictionHistories.pop(sortedKeys[0], None)
            delta = sortedKeys[1]
            for j in range(1, len(sortedKeys)):
                predictionHistories[sortedKeys[j]- delta] = predictionHistories.pop(sortedKeys[j])
        return predictionHistories

    @staticmethod
    def getMaxAccWindowSize(labels, nNeighbours, getLabelsFct, predictionHistories, distancesSTM, minSize=50):
        numSamples = len(labels)
        if numSamples < 2 * minSize:
            return numSamples, predictionHistories
        else:
            numSamplesRange = [numSamples]
            while numSamplesRange[-1]/2 >= minSize:
                numSamplesRange.append(numSamplesRange[-1]/2)

            accuracies = []
            for key in predictionHistories.keys():
                STMSizer.accCalcCount += 1
                if key not in (numSamples - np.array(numSamplesRange)):
                    predictionHistories.pop(key, None)

            for numSamplesIt in numSamplesRange:
                idx = numSamples - numSamplesIt
                if predictionHistories.has_key(idx):
                    accuracy, predHistory = STMSizer.getInterleavedTestTrainAccPredHistory(labels[idx:], nNeighbours, getLabelsFct, predictionHistories[idx], distancesSTM[idx:, idx:])
                else:
                    accuracy, predHistory = STMSizer.getInterleavedTestTrainAcc(labels[idx:], nNeighbours, getLabelsFct, distancesSTM[idx:, idx:])
                predictionHistories[idx] = predHistory
                accuracies.append(accuracy)
            accuracies = np.round(accuracies, decimals=4)
            bestNumTrainIdx = np.argmax(accuracies)
            windowSize = numSamplesRange[bestNumTrainIdx]

            if windowSize < numSamples:
                predictionHistories = STMSizer.adaptHistories(bestNumTrainIdx, predictionHistories)
            return int(windowSize), predictionHistories

    @staticmethod
    def getMaxAccApproxWindowSize(labels, nNeighbours, getLabelsFct, predictionHistories, distancesSTM, minSize=50):
        numSamples = len(labels)
        if numSamples < 2 * minSize:
            return numSamples, predictionHistories
        else:
            numSamplesRange = [numSamples]
            while numSamplesRange[-1]/2 >= minSize:
                numSamplesRange.append(numSamplesRange[-1]/2)
            accuracies = []
            for numSamplesIt in numSamplesRange:
                STMSizer.accCalcCount += 1
                idx = numSamples - numSamplesIt
                if predictionHistories.has_key(idx):
                    accuracy, predHistory = STMSizer.getInterleavedTestTrainAccPredHistory(labels[idx:], nNeighbours, getLabelsFct, predictionHistories[idx], distancesSTM[idx:, idx:])
                elif predictionHistories.has_key(idx-1):
                    predHistory = predictionHistories[idx-1]
                    predictionHistories.pop(idx-1, None)
                    predHistory.pop(0)
                    accuracy, predHistory = STMSizer.getInterleavedTestTrainAccPredHistory(labels[idx:], nNeighbours, getLabelsFct, predHistory, distancesSTM[idx:, idx:])
                else:
                    accuracy, predHistory = STMSizer.getInterleavedTestTrainAcc(labels[idx:], nNeighbours, getLabelsFct, distancesSTM[idx:, idx:])
                predictionHistories[idx] = predHistory
                accuracies.append(accuracy)
            accuracies = np.round(accuracies, decimals=4)
            bestNumTrainIdx = np.argmax(accuracies)
            if bestNumTrainIdx > 0:
                moreAccurateIndices = np.where(accuracies > accuracies[0])[0]
                for i in moreAccurateIndices:
                    idx = numSamples - numSamplesRange[i]
                    accuracy, predHistory = STMSizer.getInterleavedTestTrainAcc(labels[idx:], nNeighbours, getLabelsFct, distancesSTM[idx:, idx:])
                    predictionHistories[idx] = predHistory
                    accuracies[i] = accuracy
                accuracies = np.round(accuracies, decimals=4)
                bestNumTrainIdx = np.argmax(accuracies)
            windowSize = numSamplesRange[bestNumTrainIdx]

            if windowSize < numSamples:
                predictionHistories = STMSizer.adaptHistories(bestNumTrainIdx, predictionHistories)
            return int(windowSize), predictionHistories