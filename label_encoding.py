#label encoding is the process of converting human readable labels into numerical form
import numpy as np
from sklearn import preprocessing

input_lables = ['red','black','red','green','black','yellow','white']

#creating label encoder to train
encoder = preprocessing.LabelEncoder()
encoder.fit(input_lables)

#print mapping
print("\n Label mapping:")
for  i, item in enumerate(encoder.classes_):
	print(item,"-->",i)

#Encode a set of lavels using the encoder:
test_labels =['green','black','red']
encoded_values = encoder.transform(test_labels)
print("\nLables =", test_labels)
print("Encoded values =", list(encoded_values))

#Decode values
encoded_values = [3,0,4,1]
decoded_list = encoder.inverse_transform(encoded_values)
print("\nENcoded values =",encoded_values)
print("\nDecoded labels =", decoded_list)
