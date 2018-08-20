import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#A confusion matrix is a figure or table which determines the accuracy of your classification by using a data set that has the ground truth to compare

true_labels = [2,0,0,2,4,4,1,0,3,3,3]
pred_labels = [2,1,0,2,4,3,1,0,1,3,3]

confusion_mat = confusion_matrix(true_labels,pred_labels)

plt.imshow(confusion_mat, interpolation='nearest',cmap=plt.cm.gray)
plt.title('Confusion matrix')
plt.colorbar()
ticks = np.arange(5)
plt.xticks(ticks,ticks)
plt.yticks(ticks,ticks)
plt.ylabel("True Lables")
plt.xlabel("Predicted lables")
plt.show()

targets = ['Class-0','Class-1','Class-2','Class-3','Class-4']
print('\n', classification_report(true_labels,pred_labels,target_names=targets))
