"""
Ryan Farr
rlf238

CS 5785 - Applied Machine Learning
Homework 1
"""
import numpy as np
from heapq import heappush, heappop

class KNN:
    """ class KNN

    An implementation of K-Nearest Neighbors
    """
    _k = 1
    _tdata = None
    _tlabels = None

    def __init__(self, k, training_data, training_labels):
        self._k = k
        self._tdata = training_data
        self._tlabels = training_labels

    def classify(self, test_point):
        """ classify function

        Classify a given test point.

        Args
        ----
        test_point : np.array
            data point to be tested

        Returns
        -------
        Number
        """
        # compare to each point in training set
        heap = []
        for i in range(len(self._tdata)):
            data = self._tdata[i]
            label = self._tlabels[i]
            heappush(heap, (np.linalg.norm(data-test_point), label))

        # grab the k closest points from our heap
        k_votes = {}
        for i in range(self._k):
            label = heappop(heap)[1]
            if label not in k_votes:
                k_votes[label] = 0
            k_votes[label] += 1
        
        # return the highest voted item
        highest_vote = -1
        vote_label = None
        for v,c in k_votes.items():
            if c > highest_vote:
                highest_vote = c
                vote_label = v
        return vote_label