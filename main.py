import pandas as pd

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
print(dane_test)