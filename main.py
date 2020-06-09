# program prepared for PSZT course
# project 2 - machine learning
# id3 alghoritm + k-fold cross validation
# author: Emilia Gosk

import pandas as pd
import numpy as np
from numpy import log
from sklearn.model_selection import KFold


# small value used to prevent division by 0 and log(0)
epsilon = np.finfo(float).eps


# function used to import the data set and
# process it into data frame
def data_import():
    raw_data = []
    with open('divorce.csv') as file:
        for line in file:
            temp = line.strip().split(";")
            raw_data.append(temp)

    temp1 = raw_data[1:] # we take only attribute values, without column names, one row has data about one person
    attributes = zip(*temp1) # trasposition --> one row has data about one attribute
    temp_data = zip(raw_data[0], attributes) # creates tuple (column_names, attributes)
    dictionary_dataset = dict(temp_data) # creates dictionary: Atr1: val_1, val_2.. , Atr2: val_1, val_2 ....
    divorces_df = pd.DataFrame(dictionary_dataset, columns=raw_data[0]) # creates table

    return divorces_df


# function returns 2 sets of data
# training set and test set
# splitting it into set in ratio (k-1):1
def k_cross_valid_data_sets(my_data_frame, k):
    kfold = KFold(k, True, 1)
    kf = kfold.split(my_data_frame)

    training_set_indexes = []
    test_set_indexes = []

    for train, test in kf:
        training_set_indexes.append(train)
        test_set_indexes.append(test)

    train_dataframes =[]
    test_dataframes =[]
    for i in range(0, k):
        test_dataframe = 0
        test_dataframe = pd.DataFrame(my_data_frame.loc[test_set_indexes[i]])
        test_dataframes.append(test_dataframe)

        train_dataframe = 0
        train_dataframe = pd.DataFrame(my_data_frame.loc[training_set_indexes[i]])
        train_dataframes.append(train_dataframe)

    return train_dataframes, test_dataframes


# function computes entropy of data set splited into subsets
def entropy(my_data_frame):
    entropy = 0

    classes = my_data_frame.keys()[-1] # last column in data set contains classes
    possible_values = my_data_frame[classes].unique() # here: 1 - divorced, 0 - married

    # coursebook p. 76:
    #I(S) = - sum(fc(S) * ln fc(S))
    #I(S) - entropy of whole data set
    #fc(S) = frequency of appearing class c in set S
    for value in possible_values:
        frequency = my_data_frame[classes].value_counts()[value] / len (my_data_frame[classes])
        entropy += - frequency * np.log(frequency)

    return entropy


# function computes entropy for provided attribute
def attribute_entropy(my_data_frame, attr):
    attr_entropy = 0

    classes = my_data_frame.keys()[-1]   # last column in data set contains classes
    class_variables = my_data_frame[classes].unique()  # here: 1 - divorced, 0 - married
    variables = my_data_frame[attr].unique()    # unique different values in that attribute

    # coursebook p. 76:
    # equation for entropy of subset:
    # Inf(D,S) = sum { (|Sj|/|S|) * I(Sj) }
    # |S| - number of elements in set S
    # Sj for j = 1,2... - subsets created by different values of attribute D
    for variable in variables:
        entropy = 0

        for class_variable in class_variables:
            num_of_elem_in_set_attr = len(my_data_frame[attr][my_data_frame[attr]==variable][my_data_frame[classes] ==class_variable]) # |Sj|
            num_of_elem_in_set = len(my_data_frame[attr][my_data_frame[attr]==variable]) # |S|
            frequency = num_of_elem_in_set_attr/(num_of_elem_in_set + epsilon) #  |Sj| / |S| ; adding epsilon to prevent division by zero
            entropy += -frequency*log(frequency + epsilon)  # calculates entropy for one value of attribute

        fraction2 = num_of_elem_in_set/len(my_data_frame)
        attr_entropy += -fraction2*entropy # calculates entropy of data set after it was divided for subsets
        abs_attr_entropy = abs(attr_entropy)

    return abs_attr_entropy


# function computes information gain and finds max value
# coursebook p. 77: InfGain(D,S) = I(S) - Inf(D,S)
def max_inf_gain(my_data_frame):
    inf_gain = []

    for key in my_data_frame.keys()[:-1]:
        inf_gain.append(entropy(my_data_frame)-attribute_entropy(my_data_frame,key))
    return my_data_frame.keys()[:-1][np.argmax(inf_gain)]


# function returns subtable
# where attribute attr has value attr_value
def get_subtable(my_data_frame, attr, attr_value):
    return my_data_frame[my_data_frame[attr] == attr_value].reset_index(drop=True)


# function where we build tree
def id3_tree(my_data_frame, tree=None):
    classes = my_data_frame.keys()[-1]  # last column in data set contains classes

    # Find attribute with max inf_gain
    # and put it in tree node
    tree_node = max_inf_gain(my_data_frame)

    possible_attValues = np.unique(my_data_frame[tree_node])  # unique values for attribute in node

    # to build tree we use dictionaries
    # create empty tree
    if tree is None:
        tree = {}
        tree[tree_node] = {}


    # here we check all possible values for attribute and use recursion to develop the tree
    for value in possible_attValues:

        subtable = get_subtable(my_data_frame, tree_node, value)

        # np.unique returns (unique_values, number_of_times_each_of_the_unique_values_comes_up_in_the_original_array)
        class_values, counter = np.unique(subtable[classes], return_counts=True)

        if len(counter) == 1:
            tree[tree_node][value] = class_values[0] # if class value comes up only once it is tree leaf
        else:
            tree[tree_node][value] = id3_tree(subtable)  # recursion here

    return tree


# function uses tree to predict
# which class the data_set belongs to
# returns 1 - divorce
# returns 0 - married
def predict(tree, testing_series):
    c = tree
    for k, v in testing_series.iteritems():
        try:
            c = c[k][v]
        except KeyError:
            continue
        if isinstance(c, dict):  # is dictionary
            pass
        else:    # is leaf
            return c


# function calculates  loss for data set
# coursebook p.62:
# q(d) = 0 when d=y
# q(d) = 1 when d!=y
# q = sum q(d)
def calculate_loss(tr, tst_set):
    q = 0
    for i in range (0, len(tst_set)):
        if predict(tr, tst_set.iloc[i]) != tst_set.iloc[i][-1]:
            q += 1

    q = q / len(tst_set)
    return q


# function calulcates average loss for model
# coursebook p.81
# q_mod = (1/k)* sum q(d)
def calculate_model_loss(train, test, k):
    q_mod = 0
    for j in range (0, k):
        q_mod += calculate_loss(id3_tree(train[j]), test[j])
    q_mod = q_mod / k
    return q_mod


def main():
    # model built on whole dataset
    print('Model built on whole dataset: ')
    dane_test = data_import()
    tree_full_dataset = id3_tree(dane_test)
    print(tree_full_dataset)

    # models built on data splited in 3:1 ratio
    # for ttraining and testing data set
    # value of loss per model printed
    print('Models and loss per model for k=4: ')
    train, test = k_cross_valid_data_sets(dane_test, 4)
    for i in range(0, 4):
        tree_4 = id3_tree(train[i])
        print(tree_4)
        print('loss:', calculate_loss(tree_4, test[i]))

    # sorted list of pairs
    # (attribute entropy, attribute name)
    print('Sorted list of entropy per attribute: ')
    list_key_entr = []
    for key in dane_test.keys()[:-1]:
        key_entr = [attribute_entropy(dane_test, key), key]
        list_key_entr.append(key_entr)
    list_key_entr.sort()
    print(list_key_entr)


main()



