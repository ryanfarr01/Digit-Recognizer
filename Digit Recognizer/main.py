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
    if not include_labels:
        return features, None

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


def get_genuine_imposter_distances(zeros, ones):
    plt.clf()
    genuine_distances = []
    imposter_distances = []

    for data in (zeros, ones):
        for i in range(len(data)):
            for j in range(i+1, len(data)):
                genuine_distances.append(np.linalg.norm(data[i] - data[j]))
    for i in range(len(zeros)):
        z = zeros[i]
        for o in ones:
            imposter_distances.append(np.linalg.norm(z-o))

    plt.hist((genuine_distances, imposter_distances), 40, stacked=True)
    plt.xlabel('L2 Distance')
    plt.legend(labels=['Genuine', 'Imposter'])
    plt.savefig('Genuine_Imposter_Hist.png')

    return genuine_distances, imposter_distances


def generate_ROC(genuine, imposter):
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
    for c in combined:
        if c[1] == 'genuine':
            tp_count += 1
        else:
            fp_count += 1
        tp.append(tp_count / tot_genuine)
        fp.append(fp_count / tot_imposter)
        rand.append(fp_count / tot_imposter)

    plt.clf()
    plt.plot(fp, tp, fp, rand)
    plt.xlabel('FP')
    plt.ylabel('TP')
    plt.title('ROC Curve')
    plt.savefig('ROC_Curve.png')


def k_fold_cross_validation(training_data, training_labels):
    data = np.array_split(training_data, 3)
    labels = np.array_split(np.array(training_labels), 3)
    best_accuracy = -1.0
    best_k = -1
    best_confusion_matrix = None

    for k in range(1, 6):
        right = 0
        wrong = 0
        confusion_matrix = [[0,0,0,0,0,0,0,0,0,0] for _ in range(10)]
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


def print_confusion_matrix(matrix):
    print('\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in matrix]))

def test_knn(k, train_data, train_labels, test_data):
    print("Final k:" + str(k))
    knn = KNN(k, train_data, train_labels)

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

    print('Displaying Digits...')
    for digit in digits:
        display_digit(train_data, train_labels, digit)

    print('Finding nearest neighbor for instance of each digit')
    for digit in digits:
        find_nearest_neighbor_random(train_data, train_labels, digit)

    print('Extracting 0 and 1 data...')
    z,o = extract_0_1_data(train_data, train_labels)

    print('Calculating genuine and imposter distance...')
    g_d, i_d = get_genuine_imposter_distances(z,o)

    print('Generating ROC...')
    generate_ROC(g_d, i_d)

    print('Performing cross validation...')
    k, matrix = k_fold_cross_validation(train_data, train_labels)
    print_confusion_matrix(matrix)

    print('Loading test data')
    test_data, _ = read_data('test.csv', False)

    print('Testing classifier...')
    test_knn(3, train_data, train_labels, test_data)
