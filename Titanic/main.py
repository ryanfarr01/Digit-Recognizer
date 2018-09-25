import csv
import numpy as np

def read_data(filename, include_labels=True):
    """ read_data function

    Reads the CSV data stored in the given filename and returns it as an Nxp
    matrix, where N is the number of samples and p is the number of features
    as well as a list of labels.

    Args
    ----
    filename : string
        filename of file to load

    Returns
    -------
    Tuple (Nxp Matrix, array, array)
    """
    features = []
    ids = []
    labels = []
    with open(filename) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            ids.append(row['PassengerId'])
            if include_labels:
                labels.append(row['Survived'])
            features.append([row['Pclass'], row['Sex'], row['Age'], row['SibSp'], row['Parch']])

    return features, labels, ids

def clean_data(features):
    # remove name, ticket, fare, embarked
    features = np.delete(features, 'name', axis=1)
    features = np.delete(features, 'ticket', axis=1)
    features = np.delete(features, 'fare', axis=1)
    features = np.delete(features, 'embarked', axis=1)

    #remove top line
    features = np.delete(features, 0, axis=0)

    # map gender to 0, 1
    # for i in len(features):
    #     gender = features[1]
    #     if 

    return features

if __name__ == '__main__':
    features, labels, ids = read_data('./train.csv')