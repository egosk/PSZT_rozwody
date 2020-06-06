import pandas as pd
import numpy as np
from numpy import log

#small value used to prevent division by 0 and log(0)
epsilon = np.finfo(float).eps

dane = []

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
    # print(divorces_df)
    # print(klasy)

    return divorces_df

dane_test = data_import()
#print(dane_test)

# function used to compute entropy of whole data set
# w podreczniku I(S) str 76

def entropy(my_data_frame):
    entropy = 0

    classes = my_data_frame.keys()[-1] # last column in data set contains classes
    possible_values = my_data_frame[classes].unique() # here: 1 - divorced, 0 - married
    #print(possible_values)

    #podrecznik str 76:
    #I(S) = - sum(fc(S) * ln fc(S))
    #I(S) - entropy of whole data set
    #fc(S) = frequency of appearing class c in set S
    for value in possible_values:
        frequency = my_data_frame[classes].value_counts()[value] / len (my_data_frame[classes])
        entropy += - frequency * np.log(frequency)

    return entropy


print(entropy(dane_test))

# function computes entropy for provided attribute

def attribue_entropy (my_data_frame, attr):
    attr_entropy = 0

    classes = my_data_frame.keys()[-1]   # last column in data set contains classes
    class_variables = my_data_frame[classes].unique()  # here: 1 - divorced, 0 - married
    variables = my_data_frame[attr].unique()    # unique different values in that attribute

    # podr str 76:
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

# for key in dane_test.keys():
#     print(key, attribue_entropy(dane_test, key))

#print(attribue_entropy(dane_test, 'Atr18'))


# function compytes information gain and finds max value
# podr str 77: InfGain(D,S) = I(S) - Inf(D,S)
def max_inf_gain(my_data_frame):
    inf_gain = []

    for key in my_data_frame.keys()[:-1]:
        inf_gain.append(entropy(my_data_frame)-attribue_entropy(my_data_frame,key))
    return my_data_frame.keys()[:-1][np.argmax(inf_gain)]

print(entropy(dane_test)-attribue_entropy(dane_test,'Atr18'))
print(max_inf_gain(dane_test))