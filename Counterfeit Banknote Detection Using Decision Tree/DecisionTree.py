__author__ = 'AAK'

"""
Author: Ankita Anilkumar Khadsare (ak8932)

This project implements the decision tree classifier.
This classifier is used to classify the banknote authentication dataset to identify counterfeit currency.

Total number of records in the data set are 1372.
"""

import numpy as np
import math
import random


class Nodes:
    """
    Class for implementing node structure required for decision tree.
    """

    __slots__ = 'left', 'right', 'attribute', 'value', 'prediction'

    def __init__(self, val, attr):
        """
        Initialize all class level variables
        :param val: split point
        :param attr: attribute to split dataset on
        """

        self.left = None
        self.right = None
        self.attribute = attr
        self.value = val
        self.prediction = None

    def __str__(self):
        """
        To get the string representation of the class.
        :return:
        """

        return "Attribute: " + str(self.attribute) + "| value: " + str(self.value) + "| Prediction: " + str(self.prediction)


def get_pos_neg_count(data):
    """
    Returns the frequency count of 0's and 1's from the input data.
    :param data: input data
    :return: 0 and 1 counts
    """
    neg_count = 0   # count of class 0
    pos_count = 0   # count of class 1

    # iterate over each record of the data
    for record in data:
        # if record belongs to class 0 increment negative count
        if record[-1] == 0:
            neg_count += 1
        # if record belongs to class 1 increment positive count
        elif record[-1] == 1:
            pos_count += 1
    return neg_count, pos_count


def load_dataset(filename):
    """
    Read the input data file and load the dataset in a np array.
    :param filename: input data filename
    :return: np array of input data
    """
    # read the data from the file and store it in an np array.
    data = np.array([[float(x) for x in line.strip().split(',')] for line in open(filename).readlines()])
    print('Loaded %d observations.' % len(data))
    return data


class DecisionTree:
    """"
    This class implements the Decision Tree functionality. It considers the information gain as a
    measure of the purity.
    """

    # Declare the variables used in the class.
    __slots__ = 'all_data', 'sample_size', 'test_data', 'train_data', 'pos', 'neg', 'root'

    def __init__(self, complete_data, train, test):
        """
        Initialize the parameters with default values passed in the constructor.

        :param complete_data: input dataset
        :param train: training data
        :param test: testing data
        """
        self.all_data = complete_data   # Save the complete data to reference later
        # print("size:", len(self.all_data))
        self.train_data = train
        self.test_data = test
        # Compute and save the frequency count of class 0 and class 1 in the training data.
        self.neg, self.pos = get_pos_neg_count(self.train_data)
        self.root = None

    def train(self, data, considered_list):
        """
        Trains the decision tree model using training data. Also, gives a call to another function
        to create the decision tree using the trained model.

        :param data: input data for training
        :param considered_list: list of considered attributes
        :return: Root
        """
        # Initialize the dictionary to store the information gains.
        gain = {}

        # Iterate over each attribute one by one to find the best attribute to split
        for attr_id in range(len(data[0])-1):
            # print("attr-->", attr_i)

            # Here, we have considered each attribute only once. This is one of our design decisions.
            if attr_id not in considered_list:

                # Initialize the gain value for a considered attribute
                gain['atr_'+str(attr_id)] = [-99999, None, None]  # [info gain, splitting reference, attribute_id]

                # Initialize the list for attribute data
                attr_data = []

                # Iterate over each record and store the values for the considered attribute
                for record_id in range(len(data)):
                    # if data[record_id][attr_id] not in attr_data:
                    attr_data.append(data[record_id][attr_id])

                # Sort the attribute data.
                attr_data.sort()

                # Iterate over each record to find the best splitting point
                for sorted_record_id in range(len(attr_data)-1):
                    # take the average of two consecutive elements and consider it as splitting point
                    split_ref = (attr_data[sorted_record_id] + attr_data[sorted_record_id+1])/2

                    # Initialize dictionary to store splitting information
                    split = dict()
                    split['left'] = {}
                    split['right'] = {}

                    # Initialize the class 0 and class 1 counts on both left and right side
                    # in the splitting dictionary to 0.
                    for side in split.keys():
                        split[side]['pos'] = 0
                        split[side]['neg'] = 0

                    # Iterate over each record in the data to split the data and
                    # find the counts of class 0 and 1 on both sides
                    for record in data:
                        # Update class frequency counts for left half
                        if record[attr_id] <= split_ref:
                            if record[-1] == 0:
                                split['left']['neg'] += 1
                            elif record[-1] == 1:
                                split['left']['pos'] += 1
                        # Update the class frequency counts for right half
                        elif record[attr_id] > split_ref:
                            if record[-1] == 0:
                                split['right']['neg'] += 1
                            elif record[-1] == 1:
                                split['right']['pos'] += 1

                    # Compute the information gain for the current split
                    temp_gain = self.gain(split)

                    # Store the best gain information, split point and attribute id for the considered attribute.
                    if gain['atr_'+str(attr_id)][0] < temp_gain:
                        gain['atr_'+str(attr_id)] = [temp_gain, split_ref, attr_id]
            else:
                continue
        # Initialize the max gain to a very low value.
        max_gain = [-999, None, None]

        # Find the attribute with the best information gain.
        for attr in gain.keys():
            if gain[attr][0] > max_gain[0]:
                max_gain = gain[attr]

        # initialize the parent as none.
        parent = None
        # if the attribute id is not none consider the attribute from the best gain
        if max_gain[2] is not None:
            considered_list.append(max_gain[2])  # store the attribute which gives the best split
            parent = self.generate_tree(max_gain, data, considered_list)  # create a tree and store the parent
            # print("Cosidered-->", considered_list)
        # save the parent as a root
        self.root = parent
        return parent

    def generate_tree(self, details, data, considered):
        """
        Generates the decision tree on the basis of split point and attributes.

        :param details: details of the attribute [info gain, splitting reference, attribute_id]
        :param data: input data
        :param considered: considered attributes
        :return: root
        """
        root = Nodes(details[1], details[2])  # create a root node
        # split the data based on a given attribute and splitting point
        left, right = self.split_data(data, details[1], details[2])
        zeros = 0   # initialize the variable for storing the count of class 0
        ones = 0   # initialize the variable for storing the count of class 1

        if len(left) > 0:
            # count the frequency of classes from the left side
            for record in left:
                # print(cat)
                if record[-1] == 0:
                    zeros += 1
                if record[-1] == 1:
                    ones += 1
            # if all records belong to class zero predict class 0
            if zeros == len(left):
                root.left = Nodes("class", -1)
                root.left.prediction = 0
            # if all records belong to class one predict class 1
            elif ones == len(left):
                root.left = Nodes("class", -1)
                root.left.prediction = 1
            # if all attributes are considered
            elif len(considered) == len(data[0])-1:
                # create a leaf node for prediction
                root.left = Nodes("class", -1)
                # predict the majority class
                if zeros > ones:
                    root.left.prediction = 0
                else:
                    root.left.prediction = 1
            else:
                # recursive call to generate tree with remaining attributes
                root.left = self.train(left, considered.copy())

        zeros = 0
        ones = 0
        if len(right) > 0:
            # count the frequency of classes from the right side
            for record in right:
                if record[-1] == 0:
                    zeros += 1
                if record[-1] == 1:
                    ones += 1
            # if all records belong to class zero predict class 0
            if zeros == len(right):
                root.right = Nodes("class", -1)
                root.right.prediction = 0
            # if all records belong to class one predict class 1
            elif ones == len(right):
                root.right = Nodes("class", -1)
                root.right.prediction = 1
            # if all attributes are considered
            elif len(considered) == len(data[0])-1:
                # create leaf node with prediction
                root.right = Nodes("class", -1)
                # predict the majority class
                if zeros > ones:
                    root.right.prediction = 0
                else:
                    root.right.prediction = 1
            else:
                # recursive call to consider next attributes and generate the tree
                root.right = self.train(right, considered.copy())
        return root

    def split_data(self, data, split_ref, attr):
        """
        Splits data into two parts based on the split reference.

        :param data: input data
        :param split_ref: split point
        :param attr: attribute
        :return: left half and right half
        """
        left = []
        right = []
        for record in data:
            # print(record[attr])
            if record[attr] < split_ref:
                left.append(list(record))
            elif record[attr] > split_ref:
                right.append(list(record))
        return left, right

    def gain(self, split):
        """
        Returns the information gain
        :param split: split details
        :return: gain
        """
        return (self.entropy(self.pos/(self.pos+self.neg))) - self.rem(split)

    def rem(self, split):
        """
        Returns the remainder or weighted entropy.
        :param split: split details
        :return: remainder
        """
        result = 0.0
        # iterate over each side of the split
        for side in split.keys():
            # Calculate the splitting ratio
            temp = (split[side]['pos'] + split[side]['neg']) / (self.pos + self.neg)
            try:
                result += (temp * self.entropy(split[side]['pos']/(split[side]['pos']+split[side]['neg'])))
            except ZeroDivisionError:
                return 0
        return result

    def entropy(self, q):
        """
        Returns the entropy

        :param q: parameter to calculate entropy of.
        :return: entropy
        """
        if q == 0 or q == 1:
            return 0
        return -1*((q*math.log2(q))+((1-q)*math.log2(1-q)))

    def print_tree(self, root, t=''):
        """
        Prints the tree
        :param root: root
        :param t:
        :return: None
        """
        if root is not None:
            print(t, "Left of ", root.attribute)
            self.print_tree(root.left, t+'\t')
            print(t, root)
            print(t, "Right of ", root.attribute)
            self.print_tree(root.right, t+'\t')

    def test(self, data, ip_type):
        """
        Tests the performance of decision tree using testing data. It makes use of 3 methods.
        1. random : generate a random class
        2. majority: always returns the majority class.
        3. Regular: decision tree classifier.

        :param data: testing data
        :param ip_type: random/regular/majority.
        :return: prediction(0/1)
        """
        correct = 0
        total = 0
        value = None
        # iterate over each record to predict the class.
        for record in data:
            if ip_type == "Random":
                value = random.randint(0, 1)  # generate class 0 or 1 randomly.
            elif ip_type == "Majority":
                # Returns the class with maximum frequency
                if self.pos <= self.neg:
                    value = 0
                else:
                    value = 1
            elif ip_type == "Regular":
                # Predict using decision tree
                value = self.predict(record, self.root)
            if value == record[-1]:
                correct += 1
            total += 1
        # print("\t\t", correct, " / ", total)
        accuracy = int(correct*100/total)
        # print("\t\tAccuracy : ", str(accuracy), "%")
        return accuracy

    def predict(self, data_record, node):
        """
        Returns the prediction for input record.

        :param data_record: input
        :param node:
        :return:
        """
        prediction = None

        # if node is none return
        if node is None:
            return

        # if node is leaf node, predict leaf node class
        if node.left is None and node.right is None:
            prediction = node.prediction
        # recurse until the prediction can be made
        else:
            if data_record[node.attribute] < node.value:
                prediction = self.predict(data_record, node.left)
            else:
                prediction = self.predict(data_record, node.right)
        # print(prediction, data_record)
        return prediction


def main():
    """
    Driver function for decision tree.
    :return:
    """
    # load data from the input data file
    input_data = load_dataset("data_banknote_authentication.txt")

    # initialize values for accuracy using regular function, majority function and random function to be 0 respectively.
    re_run_acc = [0, 0, 0]

    # Set the maximum limit for the re-runs
    max_rerun = 5

    # re-run the classifier for 5 times.
    for run_id in range(max_rerun):
        print("\nRun ", str(run_id+1), " : ")

        # randomly shuffle the input data
        np.random.shuffle(input_data)

        # generating the number of records in the input dataset
        size = len(input_data)

        # initialize N for N-fold cross validation
        n_for_cv = 5

        # initialize the size of each chunk to be used for cross validation
        chunk = size // n_for_cv
        accuracy = [0, 0, 0]

        # iterate for 5 times to perform 5 fold cross validation
        for chunk_id in range(n_for_cv):
            train = []

            # Extract the chunk of a testing data from original data.
            test = input_data[chunk_id*chunk:(chunk_id*chunk)+chunk]

            # Save remaining data as a training data.
            train.extend(input_data[:chunk_id*chunk])
            train.extend(input_data[(chunk_id*chunk)+chunk:])

            # create an object of Decision tree.
            d_object = DecisionTree(input_data, train, test)

            # train the decision tree using the input data
            root = d_object.train(d_object.train_data, [])

            # printing the generated tree
            # d_object.print_tree(root)

            # check the result of the generated decision tree on a testing data using regular function.
            accuracy_d_tree = d_object.test(d_object.test_data, "Regular")
            accuracy[0] += accuracy_d_tree

            # check the result of the generated decision tree on a testing data using majority function.
            accuracy_majority = d_object.test(d_object.test_data, "Majority")
            accuracy[1] += accuracy_majority

            # check the result of the generated decision tree on a testing data using random function.
            accuracy_rand = d_object.test(d_object.test_data, "Random")
            accuracy[2] += accuracy_rand

            print("Accuracy during Cross Validation Iteration {} : Decision Tree = {:2.3f}%,Majority = {:2.3f}%,"
                  " Random = {:2.3f}% ".format(chunk_id+1, accuracy_d_tree, accuracy_majority, accuracy_majority))

        print("Average Accuracy of cross validations : Decision Tree = {:2.3f}%, Majority = {:2.3f}%, "
              "Random = {:2.3f}% \n".format(accuracy[0]/n_for_cv, accuracy[1]/n_for_cv, accuracy[2]/n_for_cv))

        re_run_acc[0] += accuracy[0]/n_for_cv
        re_run_acc[1] += accuracy[1]/n_for_cv
        re_run_acc[2] += accuracy[2]/n_for_cv

    print("Average Accuracy after multiple re-runs : Decision Tree = {:2.3f}%, Majority = {:2.3f}%, "
          "Random = {:2.3f}%\n".format(re_run_acc[0]/max_rerun, re_run_acc[1]/max_rerun, re_run_acc[2]/max_rerun))


if __name__ == '__main__':
    main()
