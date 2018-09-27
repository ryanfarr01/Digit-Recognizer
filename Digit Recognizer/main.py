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
from knn import KNN
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize


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
    features = np.genfromtxt(filename, delimiter=',')
    features = np.delete(features, 0, axis=0)
    if not include_labels:
        return features, None

    labels = [row[0] for row in features]
    features = np.delete(features, 0, axis=1)
    return (features, labels)


def display_digit(data, labels, digit):
    """ display_digit function

    Finds an example of the given digit within the lables and plots
    the associated data as an image.

    Args
    ----
    data : np.array
        the set of pixel data points
    labels : np.array
        the associated list of labels
    digit : Number
        digit to display
    """
    plt.clf()
    digit_row = None
    for i in range(data.shape[0]):
        if labels[i] == digit:
            digit_row = data[i]
            break

    get_matrix_plot(digit_row, str(int(digit)))
    plt.savefig('digit_' + str(int(digit)) + '.png')


def get_matrix_plot(data, title):
    """ get_matrix_plot function

    Plots the given data with pyplot.matshow

    Args
    ----
    data : np.array
        the pixel data to be displayed
    title : string
        title for the matrix
    """
    matrix = data.reshape(28, 28)
    plt.matshow(matrix, cmap='gray')
    plt.title(title)


def calculate_prior_probs(labels):
    """ calculate_prior_probs function

    Calculate the posterior probabilities of encountering each digit.
    Saved to 'prior_probabilities.png'

    Args
    ----
    labels : np.array
        the set of labels for the dataset
    """
    plt.clf()
    bins = np.arange(11) - 0.5
    plt.hist(labels, bins, density=True)
    plt.xlabel('Digit')
    plt.ylabel('Prior Probability in Training Data')
    plt.savefig('prior_probabilities.png')


def find_instance_of_digit(train_labels, digit):
    """ find_instance_of_digit function

    Find the first instance of a digit within the given labels and returns
    its index.

    Args
    ----
    train_labels : np.array
        list of labels
    digit : Number
        digit to search for

    Returns
    -------
    integer index
    """
    for i in range(len(train_labels)):
        if train_labels[i] == digit:
            return i

    return -1


def find_nearest_neighbor_random(train_data, train_labels, digit):
    """ find_nearest_neighbor_random function

    Find a given digit within the data then find its nearest neighbor.
    Results stored in 'NN_Digit<digit>.png'

    Args
    ----
    train_data : np.array
        training data
    train_labels : Number
        associated labels
    digit : Number
        digit to search for
    """
    plt.clf()
    digit_index = find_instance_of_digit(train_labels, digit)
    d_i = find_nearest_neighbor(train_data, train_labels, digit_index)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    m1 = train_data[digit_index].reshape(28, 28)
    ax1.matshow(m1, cmap='gray')
    m2 = train_data[d_i].reshape(28, 28)
    ax2.matshow(m2, cmap='gray')
    fig.suptitle("Nearest Neighbor for Digit " + str(int(digit)))
    plt.savefig("NN_Digit" + str(int(digit)) + ".png")


def find_nearest_neighbor(train_data, train_labels, index):
    """ find_nearest_neighbor function

    Find the nearest neighbor to a data point in the train_data set.

    Args
    ----
    train_data : np.array
        training data
    train_labels : Number
        associated labels
    index : Number
        index of the training data point you want to find the neighbor of

    Returns
    -------
    integer index
    """
    smallest_val = sys.float_info.max
    smallest_index = -1

    for i in range(len(train_data)):
        if i == index:
            continue
        dist = l2_distance(train_data[index], train_data[i])
        if(dist < smallest_val):
            smallest_val = dist
            smallest_index = i

    return smallest_index


def l2_distance(data1, data2):
    """ l2_distance function

    Find the L2 distance between two data points.

    Args
    ----
    data1 : np.array
        pixel array
    data2 : np.array
        pixel array

    Returns
    -------
    Number
    """
    sum = 0

    for i in range(len(data1)):
        val1 = data1[i]
        val2 = data2[i]
        sum += ((val1-val2) ** 2)

    return math.sqrt(sum)


def extract_0_1_data(data_set, label_set):
    """ extract_0_1_data function

    Extract all 0 and 1 data in the given data set

    Args
    ----
    data_set : np.array
        dataset
    label_set : np.array
        associated labels

    Returns
    -------
    Tuple (np.array, np.array)
    """
    zeros = [data_set[i] for i in range(len(data_set)) if label_set[i] == 0.0]
    ones = [data_set[i] for i in range(len(data_set)) if label_set[i] == 1.0]

    return (np.array(zeros), np.array(ones))


def get_genuine_imposter_distances(zeros, ones):
    """ get_genuine_imposter_distances function

    Calculates the genuine and imposter distances for the given data sets.

    Args
    ----
    zeros : np.array
        dataset of 0s
    ones : np.array
        dataset of 1s

    Returns
    -------
    Tuple (np.array, np.array)
    """
    genuine_distances = []
    imposter_distances = []
    independent_genuine_distances = [[], []]

    # calculate genuines
    dist_ind = 0
    for data in (zeros, ones):
        for i in range(len(data)):
            for j in range(i+1, len(data)):
                dist = np.linalg.norm(data[i] - data[j])
                genuine_distances.append(dist)
                independent_genuine_distances[dist_ind].append(dist)
        dist_ind += 1

    # calculate imposter distances
    for i in range(len(zeros)):
        for o in ones:
            dist = np.linalg.norm(z-o)
            imposter_distances.append(dist)

    # create the figure for combined genuine/imposter histogram
    plt.clf()
    plt.hist((genuine_distances, imposter_distances), 40, stacked=True)
    plt.xlabel('L2 Distance')
    plt.title('Genuine/Imposter Histogram for 0s and 1s')
    plt.legend(labels=['Genuine', 'Imposter'])
    plt.savefig('Genuine_Imposter_Hist.png')

    # create histogram for 0s
    plt.clf()
    plt.hist(
        (independent_genuine_distances[0], imposter_distances), 40, stacked=True)
    plt.xlabel('L2 Distance')
    plt.title('Genuine/Imposter Histogram for 0s')
    plt.legend(labels=['Genuine', 'Imposter'])
    plt.savefig('Genuine_Imposter_Hist_0.png')

    # create histogram for 1s
    plt.clf()
    plt.hist(
        (independent_genuine_distances[1], imposter_distances), 40, stacked=True)
    plt.xlabel('L2 Distance')
    plt.title('Genuine/Imposter Histogram for 1s')
    plt.legend(labels=['Genuine', 'Imposter'])
    plt.savefig('Genuine_Imposter_Hist_1.png')

    return genuine_distances, imposter_distances


def generate_ROC(genuine, imposter):
    """ generate_ROC function

    Generates an ROC curve from the genuine and imposter distances.

    Args
    ----
    genuine : list
        list of distances for genuine matches
    imposter : list
        list of distances for imposters
    """
    combined = [(g, 'genuine') for g in genuine]
    combined += [(i, 'imposter') for i in imposter]
    combined.sort(key=lambda t: t[0])

    tot_genuine = float(len(genuine))
    tot_imposter = float(len(imposter))
    tp_count = 0
    fp_count = 0
    tp = []
    fp = []
    rand = []
    eer = []
    for c in combined:
        if c[1] == 'genuine':
            tp_count += 1
        else:
            fp_count += 1
        tp.append(tp_count / tot_genuine)
        fp.append(fp_count / tot_imposter)
        rand.append(fp_count / tot_imposter)
        eer.append(1-rand[-1])

        if tp_count == fp_count:
            print("tp == fp at: " + str(float(tp_count)/(tp_count + fp_count)))
    plt.clf()
    plt.plot(fp, tp, fp, rand, fp, eer)
    plt.xlabel('FP')
    plt.ylabel('TP')
    plt.title('ROC Curve')
    plt.savefig('ROC_Curve.png')


def k_fold_cross_validation(training_data, training_labels):
    """ k_fold_cross_validation function

    Performs 3-fold cross validation on the training data to determine
    the best k-value for k-NN. Values tested are [1,5]

    Args
    ----
    training_data : np.array
        training data
    training_labels : np.array
        Associated training labels

    Returns
    -------
    integer
    """
    data = np.array_split(training_data, 3)
    labels = np.array_split(np.array(training_labels), 3)
    best_accuracy = -1.0
    best_k = -1
    best_confusion_matrix = None

    for k in range(1, 6):
        right = 0
        wrong = 0
        confusion_matrix = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in range(10)]
        for n in range(3):  # 3-fold cross validation
            # split up data
            test_data = data[n]
            test_label = labels[n]
            if n == 0:
                train_data = np.concatenate((data[1], data[2]))
                train_labels = np.concatenate((labels[1], labels[2]))
            elif n == 1:
                train_data = np.concatenate((data[0], data[2]))
                train_labels = np.concatenate((labels[0], labels[2]))
            elif n == 2:
                train_data = np.concatenate((data[0], data[1]))
                train_labels = np.concatenate((labels[0], labels[1]))

            # train classifier
            knn = KNN(k, train_data, train_labels)

            # test classifier
            for d_index in range(len(test_data)):
                true_label = test_label[d_index]
                guess = knn.classify(test_data[d_index])
                confusion_matrix[int(true_label)][int(guess)] += 1
                if guess == true_label:
                    right += 1.0
                else:
                    wrong += 1.0

        # determine accuracy
        accuracy = right / (right + wrong)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k
            best_confusion_matrix = confusion_matrix
        print("Accuracy for k=" + str(k) + ": " + str(accuracy))

    return best_k, best_confusion_matrix


def test_knn(k, train_data, train_labels, test_data):
    """ test_knn function

    Trains a KNN classifier with the given testing set then tests it
    on the testing data. Outputs as a CSV file.

    Args
    ----
    k : integer
        number of neighbors to use for KNN
    train_data : np.array
        training dataset
    train_labels : np.array
        training dataset labels
    test_data : np.array
        testing dataset

    Returns
    -------
    Tuple (np.array, np.array)
    """
    print("Final k:" + str(k))
    knn = KNN(k, train_data, train_labels)

    # print to CSV
    with open('predictions_digit_recognizer.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['ImageId', 'Label'])
        for i in range(len(test_data)):
            data = test_data[i]
            guess = knn.classify(data)
            writer.writerow([str(i+1), str(int(guess))])


if __name__ == '__main__':
    digits = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]

    print('Loading training data...')
    train_data, train_labels = read_data('train.csv')

    print('Calculating posterior probabilities')
    calculate_prior_probs(train_labels)

    print('Displaying Digits...')
    for digit in digits:
        display_digit(train_data, train_labels, digit)

    print('Finding nearest neighbor for instance of each digit')
    for digit in digits:
        find_nearest_neighbor_random(train_data, train_labels, digit)

    print('Extracting 0 and 1 data...')
    z, o = extract_0_1_data(train_data, train_labels)

    print('Calculating genuine and imposter distance...')
    g_d, i_d = get_genuine_imposter_distances(z, o)

    print('Generating ROC...')
    generate_ROC(g_d, i_d)

    print('Performing cross validation...')
    k, matrix = k_fold_cross_validation(train_data, train_labels)

    print('Loading test data')
    test_data, _ = read_data('test.csv', False)

    print('Testing classifier...')
    test_knn(3, train_data, train_labels, test_data)
