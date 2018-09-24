import csv
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize


def read_data(filename):
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
    Tuple (Nxp Matrix, list)
    """
    features = np.genfromtxt(filename, delimiter=',')
    features = np.delete(features, 0, axis=0)
    labels = [row[0] for row in features]
    features = np.delete(features, 0, axis=1)
    return (features, labels)


def display_digit(data, labels, digit):
    plt.clf()
    digit_row = None
    for i in range(data.shape[0]):
        if labels[i] == digit:
            digit_row = data[i]
            break

    get_matrix_plot(digit_row, str(int(digit)))
    plt.savefig('digit_' + str(int(digit)) + '.png')


def get_matrix_plot(data, title):
    matrix = data.reshape(28, 28)
    plt.matshow(matrix, cmap='gray')
    plt.title(title)


def calculate_prior_probs(labels):
    plt.clf()
    plt.hist(labels, 10, density=True)
    plt.xlabel('Digit')
    plt.ylabel('Prior Probability in Training Data')
    plt.savefig('prior_probabilities.png')


def find_instance_of_digit(train_labels, digit):
    for i in range(len(train_labels)):
        if train_labels[i] == digit:
            return i
    return -1


def find_nearest_neighbor_random(train_data, train_labels, digit):
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
    sum = 0
    for i in range(len(data1)):
        val1 = data1[i]
        val2 = data2[i]
        sum += ((val1-val2) ** 2)
    return math.sqrt(sum)


def extract_0_1_data(data_set, label_set):
    zeros = [data_set[i] for i in range(len(data_set)) if label_set[i] == 0.0]
    ones = [data_set[i] for i in range(len(data_set)) if label_set[i] == 1.0]
    return (np.array(zeros), np.array(ones))


if __name__ == '__main__':
    digits = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    train_data, train_labels = read_data('train_min.csv')
    # for digit in digits:
    #     display_digit(train_data, train_labels, digit)
    # for digit in digits:
    #     find_nearest_neighbor_random(train_data, train_labels, digit)
    z,o = extract_0_1_data(train_data, train_labels)
    
