#Logistic Regression - is a technique that is used to explain the relationship btw input variables and output variables
#input varaibles are independent while the output values are dependent
#this classifer helps idenify the relationship btw the independant and dependant variables

import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

from utilities import visualize_classifier

x = np.array([[3.1,7.2],[4,6.7],[2.9,8],[5.1,4.5],[6,5],[5.6,5],[3.3,0.4],[3.9,0.9],[2.8,1],[0.5,3.4],[1,4],[0.6,4.9]])

y = np.array([0,0,0,1,1,2,2,2,3,3,3])

#creating the logistic regression classifier
classifier = liner_model.logisticRegression(solerver='liblinear',C=1)

#train the classifier
classifier.fit(x,y)

visualize_classifier(classifier,x,y)

