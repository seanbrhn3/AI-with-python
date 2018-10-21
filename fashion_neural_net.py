#Using tensor flows keras library to build a neural netowrk

import tensorflow as tf
from tensorflow import keras

#Helper libraries
import numpy as np
import matplotlib.pyplot as plt

#download dataset
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels),(test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#this give you info on the data showing you its 60,000 images and the images are 28x28
print(train_images.shape)

#Example of how the shoe images look. The images should also be preprocessed
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)


#Before we feed our data to a neural network we need to scale our data to be 0 and 1
#To do this we convert our data to decimal values
train_images = train_images / 255.0
test_images = test_images / 255.0

#To test whether the data we have is in the format, images and labels both. We will
#graph some of the images with their labels to make sure they work
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i],cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
#plt.show()

#LAYERS
#this is the first part of a neural network. We build layers to etract information from out dataset
#hopefully the information extracted is useful to solve the problem at hand

#most deep learning cinsists of chaining layers together

model = keras.Sequential([
keras.layers.Flatten(input_shape=(28,28)),
keras.layers.Dense(128,activation=tf.nn.relu),
keras.layers.Dense(10,activation=tf.nn.softmax)
])

#The Flatten portion of the model changes the 2d array of 28x28 pixel images to one row
#of 28 x 28 = 784 pixels, this basically lines them up in a rown to easily go through them

#The first layer consists of 128 nodes that will take input. The last layer
#Has 10 which will output a 0 or 1 if what ever is fed in fits into on of the 10 categories

#Before we run the neural network we need check certain things



model.compile(optimizer=tf.train.AdamOptimizer(),
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])

model.fit(train_images,train_labels,epochs=5)

test_loss, test_acc = model.evaluate(test_images,test_labels)
print(test_acc)

#Considering the training data was more accuracte than the test dataset is an
#example of overfitting - this occurs when we do better on our training data
#then we do on our test data

#PREDICTIONS
#predictions are used to check the confidence of the neural network
predictions = model.predict(test_images)
print(predictions[0])

#You use this to see what the predictions gives You
print(np.argmax(predictions[0]))
