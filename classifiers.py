#!/usr/bin/env python

import cv2
import numpy as np

from abc import ABCMeta, abstractmethod
from matplotlib import pyplot as plt



class Classifier:
    
    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, X_train, y_train):
        pass

    @abstractmethod
    def evaluate(self, X_test, y_test, visualize=False):
        pass

    def _accuracy(self, y_test, Y_vote):

        # predicted classes
        y_hat = np.argmax(Y_vote, axis=1)

        # all cases where predicted class was correct
        mask = y_hat == y_test
        return np.count_nonzero(mask)*1. / len(y_test)

    def _precision(self, y_test, Y_vote):
 
        # predicted classes
        y_hat = np.argmax(Y_vote, axis=1)

        if self.mode == "one-vs-one":
            # need confusion matrix
            conf = self._confusion(y_test, Y_vote)

            # consider each class separately
            prec = np.zeros(self.num_classes)
            for c in xrange(self.num_classes):
                # true positives: label is c, classifier predicted c
                tp = conf[c, c]

                # false positives: label is c, classifier predicted not c
                fp = np.sum(conf[:, c]) - conf[c, c]

                # precision
                if tp + fp != 0:
                    prec[c] = tp * 1. / (tp + fp)
        elif self.mode == "one-vs-all":
            # consider each class separately
            prec = np.zeros(self.num_classes)
            for c in xrange(self.num_classes):
                # true positives: label is c, classifier predicted c
                tp = np.count_nonzero((y_test == c) * (y_hat == c))

                # false positives: label is c, classifier predicted not c
                fp = np.count_nonzero((y_test == c) * (y_hat != c))

                if tp + fp != 0:
                    prec[c] = tp * 1. / (tp + fp)
        return prec

    def _recall(self, y_test, Y_vote):
        # predicted classes
        y_hat = np.argmax(Y_vote, axis=1)

        if self.mode == "one-vs-one":
            # need confusion matrix
            conf = self._confusion(y_test, Y_vote)

            # consider each class separately
            recall = np.zeros(self.num_classes)
            for c in xrange(self.num_classes):
                # true positives: label is c, classifier predicted c
                tp = conf[c, c]

                # false negatives: label is not c, classifier predicted c
                fn = np.sum(conf[c, :]) - conf[c, c]
                if tp + fn != 0:
                    recall[c] = tp * 1. / (tp + fn)
        elif self.mode == "one-vs-all":
            # consider each class separately
            recall = np.zeros(self.num_classes)
            for c in xrange(self.num_classes):
                # true positives: label is c, classifier predicted c
                tp = np.count_nonzero((y_test == c) * (y_hat == c))

                # false negatives: label is not c, classifier predicted c
                fn = np.count_nonzero((y_test != c) * (y_hat == c))

                if tp + fn != 0:
                    recall[c] = tp * 1. / (tp + fn)
        return recall

    def _confusion(self, y_test, Y_vote):

        y_hat = np.argmax(Y_vote, axis=1)
        conf = np.zeros((self.num_classes, self.num_classes)).astype(np.int32)
        for c_true in xrange(self.num_classes):
            # looking at all samples of a given class, c_true
            # how many were classified as c_true? how many as others?
            for c_pred in xrange(self.num_classes):
                y_this = np.where((y_test == c_true) * (y_hat == c_pred))
                conf[c_pred, c_true] = np.count_nonzero(y_this)
        return conf


class MultiLayerPerceptron(Classifier):

    def __init__(self, layer_sizes, class_labels, params=None, 
                 class_mode="one-vs-all"):
        self.num_features = layer_sizes[0]
        self.num_classes = layer_sizes[-1]
        self.class_labels = class_labels
        self.params = params or dict()
        self.mode = class_mode

        # initialize MLP
        self.model = cv2.ANN_MLP()
        self.model.create(layer_sizes)

    def load(self, file):
        """Loads a pre-trained MLP from file"""
        self.model.load(file)

    def save(self, file):
        """Saves a trained MLP to file"""
        self.model.save(file)

    def fit(self, X_train, y_train, params=None):
        if params is None:
            params = self.params

        # need int labels as 1-hot code
        y_train = self._labels_str_to_num(y_train)
        y_train = self._one_hot(y_train).reshape(-1, self.num_classes)

        # train model
        self.model.train(X_train, y_train, None, params=params)

    def predict(self, X_test):
        ret, y_hat = self.model.predict(X_test)
        # find the most active cell in the output layer
        y_hat = np.argmax(y_hat, 1)
        # return string labels
        return self.__labels_num_to_str(y_hat)

    def evaluate(self, X_test, y_test):
        # need int labels
        y_test = self._labels_str_to_num(y_test)
        # predict labels
        ret, Y_vote = self.model.predict(X_test)

        accuracy = self._accuracy(y_test, Y_vote)
        precision = self._precision(y_test, Y_vote)
        recall = self._recall(y_test, Y_vote)

        return (accuracy, precision, recall)

    def _one_hot(self, y_train):
        """Converts a list of labels into a 1-hot code"""
        numSamples = len(y_train)
        new_responses = np.zeros(numSamples*self.num_classes, np.float32)
        resp_idx = np.int32(y_train + np.arange(numSamples)*self.num_classes)
        new_responses[resp_idx] = 1
        return new_responses

    def _labels_str_to_num(self, labels):
        """Converts a list of string labels to their corresponding ints"""
        return np.array([int(np.where(self.class_labels == l)[0])
                         for l in labels])

    def __labels_num_to_str(self, labels):
        """Converts a list of int labels to their corresponding strings"""
        return self.class_labels[labels]
