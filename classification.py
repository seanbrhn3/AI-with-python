#Classification determines what catagory data points belong
import numpy as np
from sklearn import preprocessing

input_data = np.array([[5.1,-2.9,3.3],[-1.2,7.8,-6.1],[3.9,0.4,2.1],[7.3,-9.9,-4.5]])

#Preprocessing techniques: 
#Technique 1 Binarization - When you want to convert your numerical values into boolean values

#Binarization
data_binarized = preprocessing.Binarizer(threshold=2.1).transform(input_data)
print("\nBinarized data:\n", data_binarized)
#once run you can see that the values above 2.1 become 1 and the lower ones
#become 0

#Mean removal: You want to remove the mean so that you can center the data at 0
#removing the mean get rid of bias in the data

#Print mean and standard deviation
print("\nBEFORE")
print("Mean =", input_data.mean(axis=0))
print("standard deviation=", input_data.std(axis=0))

print("\AFTER")
data_scaled = preprocessing.scale(input_data)
print("Mean =", data_scaled.mean(axis=0))
print("Standard Deviation = ", data_scaled.std(axis=0))

#Scaling
#Feature vector- A individual measure of a observed phenomina
data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0,1))
data_scaled_minmax = data_scaler_minmax.fit_transform(input_data)
print("\nMin max scaled data:\n", data_scaled_minmax)
#NORMALIZATION

#There are two forms of normalization. We use normalization to modify features
#in out vector so that they can be measured on a common scale

#l1 normalization - makes sure that the sum of absolute values is 1 in each row
#l2 normalization - make sure that the sum of squares is at least 1
data_normalized_l1 = preprocessing.normalize(input_data,norm='l1')
data_normalized_l2 = preprocessing.normalize(input_data,norm='l2')
print('\nL1 normalized data:\n',data_normalized_l1)
print('\nL2 normalized data:\n',data_normalized_l2)

#Label encoding: the process of turning human readable labels into the numerical form
