#Creating a csv file with random data to then use for testing purposes
import random

with open("test_data.csv",'w') as data:
    for i in range(1,1001):
        data.write(str(random.randint(0,i))+"\n")
