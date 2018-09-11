import numpy as np
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn import cross_validation

#this will be where we load the data
X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000
#https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
#adult_data = requests.get("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data")
#we need to preprocess the data to read it
#Read data
with open("adult.txt","r") as data:
    for line in data.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if "?" in line:
            continue
        else:
            data = line[:-1].split(", ")
            splitter = line.split(", ")[-1]
            print("his: ", data[-1])
            print("mine: ",splitter) #Mine had a space in it so it was wrong :()
            if data[-1] == '<=50K' and count_class1 < max_datapoints:
                X.append(data)
                count_class1 += 1
            if data[-1] == '>50K' and count_class2 < max_datapoints:
                X.append(data)
                count_class2 +=1
X = np.array(X)
#Convert string data to numerical data
label_encoder = []
X_encoded = np.empty(X.shape)
for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:,i] = X[:,i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:,i] = label_encoder[-1].fit_transform(X[:,i])
X = X_encoded[:,:-1].astype(int)
y = X_encoded[:,-1].astype(int)
#Create SVM classifier
classifier = OneVsOneClassifier(LinearSVC(random_state=0))
#Train Classifier
classifier.fit(X,y)


        #datem = line[:-1] == "<=50k" and count_class2 < max_datapoints:
