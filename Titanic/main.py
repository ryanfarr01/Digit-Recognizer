"""
Ryan Farr
rlf238

CS 5785 - Applied Machine Learning
Homework 1
"""
import csv
import sys
import math
import numpy as np
from sklearn import linear_model

def read_data(filename, include_labels=True):
    """ read_data function

    Reads the CSV data stored in the given filename and returns it as an Nxp
    matrix, where N is the number of samples and p is the number of features
    as well as a list of labels and ids.

    Args
    ----
    filename : string
        filename of file to load
    include_labels : bool (default True)
        If True, extracts labels from the CSV. Set to False if you're loading a testing dataset

    Returns
    -------
    Tuple (Nxp Matrix, array, array)
    """
    features = []
    ids = []
    labels = []
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            ids.append(int(row['PassengerId']))
            if include_labels:
                labels.append(int(row['Survived']))
            features.append(extract_features(row))
    
    return np.array(features), np.array(labels), np.array(ids)

def extract_features(row):
    """ extract_features function

    Taking one row, extracts all features deemed useful for logistic regression
    and returns them as a list.

    Args
    ----
    row : dict
        A single data point

    Returns
    -------
    list
    """
    pclass = float(row['Pclass']) if row['Pclass'] != '' else 0
    sex = 1 if row['Sex'] == 'male' else 0
    age = float(row['Age']) if row['Age'] != '' else 0
    sibsp = float(row['SibSp']) if row['SibSp'] != '' else 0
    parch = float(row['Parch']) if row['Parch'] != '' else 0

    return [pclass, sex, age, sibsp, parch]

def train_linear_regression(features, labels):
    """ train_linear_regression function

    Trains a logistic regression classifier given the features and labels.

    Args
    ----
    features : np.array
        matrix of size N x p where N is the number of data points and p
        is the number of features
    labels : np.array
        list of size N storing the correct labels for the feature set

    Returns
    -------
    LogisticRegression
    """
    logreg = linear_model.LogisticRegression()
    logreg.fit(features, labels)

    return logreg

def test_classifier(classifier, test_data, test_labels=None):
    """ test_classifier function

    Tests a given classifier on the given test data and outputs the results.
    If test_labels is given, will calculate and print accuracy.

    Args
    ----
    classifier : LogisticRegression
        classifier to be tested
    test_data : np.array
        matrix of size N x p where N is the number of data points and p
        is the number of features
    test_labels : np.array (default None)
        the list of labels for the associated data set

    Returns
    -------
    LogisticRegression
    """
    guesses = classifier.predict(test_data)

    if test_labels is not None:
        right = 0.0
        total = 0.0
        for i in range(len(guesses)):
            guess = guesses[i]
            actual = test_labels[i]
            if guess == actual:
                right += 1.0
            total += 1.0
        print("Accuracy: " + str(right/total))
    return guesses

def store_guesses(guesses, ids):
    """ store_guesses function

    Saves the guesses array to a 'predictions_titanic.csv' file.

    Args
    ----
    guesses : list
        list of guesses from classifier
    ids : np.array
        array of ids for associated guesses
    """
    with open('predictions_titanic.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['PassengerId', 'Survived'])
        for i in range(len(guesses)):
            writer.writerow([str(ids[i]), str(int(guesses[i]))])

if __name__ == '__main__':
    print("Loading training data...")
    features, labels, ids = read_data('train.csv')

    print("Training classifier...")
    logreg = train_linear_regression(features, labels)

    print("Loading test data...")
    t_features, _, t_ids = read_data("test.csv", False)

    print("Testing classifier...")
    guesses = test_classifier(logreg, t_features)
    store_guesses(guesses, t_ids)
