import numpy as np
from numpy import asarray
import pandas as pd
import time
import os
from PIL import Image
from cv2 import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
# from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from keras.datasets import cifar10
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import v_measure_score

path = 'train/'
fruits = []
for x in os.listdir(path):
    fruits.append(x)

#fruits = fruits[:2] #FOR TESTING PURPOSES
print(len(fruits))
data=[] #array which stores the image data in array format
labels=[] #Array which stores corresponding label for each image
test_data = []
test_labels = []
im_w = 64
im_h = 64


fig = plt.figure(figsize=(3, 3), dpi=300)
"""
Appends all the images and their corresponding labels to an array. I save the array data to a pickle file to avoid re-computation
"""
def addImageData():
    for x in range(len(fruits)):
        sub_path = path+fruits[x]+'/'
        for y in os.listdir(sub_path):
            img_path = sub_path+y
            last = img_path[-12:]
            imag=cv2.imread(img_path)
            #imag = cv2.imread(img_path, 0) #Load image in grayscale
            if(imag is None):
                print(img_path, last)
                continue
            imag = imag[...,::-1]
            #plt.imshow(imag)
            #plt.show()
            img_from_ar = Image.fromarray(imag, 'RGB')
            resized_image = img_from_ar.resize((im_w, im_h))
            data.append(np.array(resized_image))
            #data.append(asarray(resized_image))
            labels.append(x)

            # plt.imshow(resized_image)
            # plt.show()
            # print(data)
            # print(labels)
    """
    Normalizing the pixels to be values between 0 - 1
    """
    #data = data.astype('float32')
    for i in range(len(data)):
        data[i] = data[i]/255
    #Save image data
    with open('images_data.pickle', 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    #Save labels
    with open('labels_data.pickle', 'wb') as f:
        pickle.dump(labels, f, protocol=pickle.HIGHEST_PROTOCOL)

#Populates the data array with all the images stored as arrays, and creates the labels corresponding to each image
#addImageData() #UNCOMMENT THIS TO TEST CODE FOR SAVING PICKLE FILE

def addTestData():
    test_path = 'test/'
    for x in range(len(fruits)):
        sub_path = test_path+fruits[x]+'/'
        for y in os.listdir(sub_path):
            img_path = sub_path+y
            last = img_path[-12:]
            imag=cv2.imread(img_path)
            if(imag is None):
                print(img_path, last)
                continue
            imag = imag[...,::-1] #Fix conversion from BGR to RGB
            # plt.imshow(imag)
            # plt.show()
            img_from_ar = Image.fromarray(imag, 'RGB')
            resized_image = img_from_ar.resize((im_w, im_h))
            test_data.append(np.array(resized_image))
            test_labels.append(x)

    #data = data.astype('float32')
    for i in range(len(test_data)):
        test_data[i] = test_data[i]/255
    #Save image data
    with open('test_data.pickle', 'wb') as f:
        pickle.dump(test_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    #Save labels
    with open('test_labels_data.pickle', 'wb') as f:
        pickle.dump(test_labels, f, protocol=pickle.HIGHEST_PROTOCOL)

addTestData() #UNCOMMENT THIS TO TEST CODE FOR SAVING PICKLE FILE

def loadPickleFile(filename):
    with open(filename, 'rb') as f:
        file = pickle.load(f)
    return file

#x_train = np.array(data)
x_train = loadPickleFile('images_data.pickle')
x_train = np.array(x_train)
x_train_knn = x_train.reshape(3579, 64*64*3)
y_train = keras.utils.to_categorical(loadPickleFile('labels_data.pickle'))
train_labels = loadPickleFile('labels_data.pickle')
x_test = loadPickleFile('test_data.pickle')
x_test = np.array(x_test)
x_test_knn = x_test.reshape(359, 64*64*3)
y_test = keras.utils.to_categorical(loadPickleFile('test_labels_data.pickle'))
test_labels = loadPickleFile('test_labels_data.pickle')
#(x_train1, y_train1), (x_test1, y_test1) = cifar10.load_data()

"""
Neural Networks
"""

def cnn():
    net = Sequential()
    es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, verbose = 1)
    #EarlyStopping(monitor='val_loss', mode='min', verbose=1)
    #Convolution layer to extract features from input image
    net.add(    Conv2D(64, (5,5), input_shape=(64,64,3),activation='relu')  ) #Try sigmoid later

    #Pooling layer which creates 2x2 pixel filter to get max element from feature maps
    net.add(    MaxPool2D(pool_size=(2,2)) )

    #2nd convolution layer
    net.add(    Conv2D(64, (5,5),activation='relu')  ) #Try sigmoid later

    #Another pooling layer
    net.add(    MaxPool2D(pool_size=(2,2)) )

    #Flattening layer. This reduces the dimensionality to a linear array.
    net.add(Flatten())

    #Create layer with 1000 neurons
    net.add(Dense(500, activation='relu'))

    #Dropout layer with 50% dropout rate
    net.add(Dropout(0.5))

    #Create layer with 500 neurons
    net.add(Dense(250, activation='relu'))

    #Dropout layer with 50% dropout rate
    net.add(Dropout(0.5))

    #Create layer with 250 neurons
    net.add(Dense(100, activation='relu'))

    #Create a layer with x neurons, where x is the number of classifications
    net.add(Dense(36, activation='softmax'))

    """
    Compiling Model
    """
    net.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    """
    Training model
    """
    #Splitting dataset into validation set
    model_train = net.fit(x_train, y_train, batch_size=256, epochs=50, validation_split=0.2, use_multiprocessing=True, callbacks=[es], shuffle=True)
    #model_train = net.fit(x_train, y_train, batch_size=256, epochs=50, validation_split=0.2, shuffle=True)
    """
    Visualization of model performance
    """
    plt.plot(model_train.history['accuracy'])
    plt.title('Model Performance')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()

    """
    Evaluate model using test data
    """
    numPredictions = 0
    numCorrectPredictions = 0
    dictionary_fruits = {}
    for i in range(len(fruits)):
        dictionary_fruits[fruits[i]] = []

    evaluation = net.evaluate(x_test, y_test)
    print(evaluation)
    preds = net.predict(x_test)
    #print(preds)
    for i in range(len(preds)):
        max_index = np.argmax(preds[i])
        #print(max_index)
        numPredictions += 1
        if(max_index == test_labels[i]):
            numCorrectPredictions+= 1
        else: #If prediction was not correct
            dictionary_fruits[fruits[test_labels[i]]].append(fruits[max_index])
    print("--------CNN----------\n")
    print("Fruits that were incorrectly classified: ", dictionary_fruits)
    print("Total predictions: ", numPredictions)
    print("Total correct predictions: ", numCorrectPredictions)
    print("Accuracy: ", numCorrectPredictions/numPredictions)
    print("-----------------------\n")
    #model_accuracy_test = net.evaluate(x_test, y_test)[1]
    #OR WE CAN SHOW PREDICTIONS 1 BY 1 AS FOLLOWS
    #predict = net.predict(np.array([resized_image]))
    #predictions will have a list of all indices and their respective probabilites. The one with highest number is the classification

cnn()

def ann():
    """"""
    """
    Artificial Neural Network without convolutional layers
    """
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(36, input_shape=(64,64,3), activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model_fitted = model.fit(x_train, y_train, validation_split=0.20, epochs= 50)
    plt.plot(model_fitted.history['accuracy'])
    plt.title('Artificial Neural Network Model Performance')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()
    """
    Evaluate model using test data
    """
    numPredictions = 0
    numCorrectPredictions = 0
    dictionary_fruits = {}
    for i in range(len(fruits)):
        dictionary_fruits[fruits[i]] = []

    evaluation = model.evaluate(x_test, y_test)
    #print(evaluation)
    preds = model.predict(x_test)
    #print(preds)
    for i in range(len(preds)):
        max_index = np.argmax(preds[i])
        #print(max_index)
        numPredictions += 1
        if(max_index == test_labels[i]):
            numCorrectPredictions+= 1
        else: #If prediction was not correct
            dictionary_fruits[fruits[test_labels[i]]].append(fruits[max_index])
    print("--------ANN----------\n")
    print("Fruits that were incorrectly classified: ", dictionary_fruits)
    print("Total predictions: ", numPredictions)
    print("Total correct predictions: ", numCorrectPredictions)
    print("Accuracy: ", numCorrectPredictions/numPredictions)
    print("-----------------------\n")

ann()

def knn():
    k = 10
    accuracyList = []
    for i in range(1, k+1):
        model = KNeighborsClassifier(n_neighbors=i, n_jobs=-1)
        model.fit(x_train_knn, train_labels)
        predicted = model.predict(x_test_knn)
        numPredictions = 0
        numCorrect = 0
        for i in range(len(predicted)):
            if(predicted[i] == test_labels[i]):
                numCorrect += 1
            numPredictions += 1
        print("---------KNN -", i,"-----------\n")
        print("Total Predictions: ", numPredictions)
        print("Total correct predictions: ", numCorrect)
        print("Accuracy: ", numCorrect/numPredictions)
        print("--------------------------")
        accuracyList.append(numCorrect/numPredictions)

    print(accuracyList)
    plt.plot([1,2,3,4,5,6,7,8,9,10], accuracyList, color='red')
    plt.xlabel("K-value for K-Nearest Neighbors")
    plt.ylabel("Accuracy")
    plt.title("Accuracy in Image Classifications for Different values of K in K-NN")
    plt.show()

knn()

def kMeans():
    kmeans = KMeans(init="k-means++", n_clusters=36)
    kmeans.fit(x_train_knn)
    preds = kmeans.predict(x_test_knn)
    #print(kmeans.labels_)
    print("Accuracy Test: ", v_measure_score(test_labels, preds))
    print("Accuracy Train: ", v_measure_score(train_labels, kmeans.labels_))

kMeans()
#nnet.evaluate(x_test_flattened, y_test)

#predicted = nnet.predict(x_test_flattened) #Will return the probabilities for all labels
#predictedLabel = [np.argmax(i) for i in predicted] #Will return index of predicted label






