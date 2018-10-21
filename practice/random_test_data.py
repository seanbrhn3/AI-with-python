#Creating a csv file with random data to then use for testing purposes
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize

def make_Data():
    with open("test_data.csv",'w') as data:
        for i in range(1,1001):
            data.write(str(random.randint(0,i))+"\n")

def plot_data(data_file):
    store_data = []
    twod_data = []
    with open(data_file, 'r') as data:
        for line in data:
            store_data.append(int(line))
    twod_data.append(store_data)
    #turn array into numpy array
    num_stored_data = np.array(twod_data)
    print(num_stored_data)
    #normalize data
    normalize_data(num_stored_data)
    
    plt.plot(store_data)
    plt.ylabel("Regular data")
    plt.show()

    plt.plot(twod_data)
    plt.ylabel("Normalized data")
    plt.show()

def normalize_data(array):
    normed = normalize(array)
    print(normed)
plot_data("test_data.csv")
