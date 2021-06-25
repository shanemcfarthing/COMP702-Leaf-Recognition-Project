#this program processes the images and creates a K Nearest Neighbour Classifier
from sklearn.neighbors import *
from sklearn import *
from sklearn.model_selection import *
from ImageReading import *
import numpy as np
import cv2 as cv

#create the feature lists that will be used by the KNN classifier
averageGreenFeature = []
areaFeature = []
shapeFeature = []
erosionFeature = []
doubleErosionFeature = []

#create the list to store the labels of the images
labels = []

#this is the name of the file from which the image paths will be read
fileName = "leafsnap-dataset-images.txt"

#call the method to handle image reading and processing
readImages(fileName, averageGreenFeature, areaFeature, shapeFeature, labels)

#normalise the values in the feature lists so that both are in range [0,1] to 
#ensure that distance function is not skewed towards a single feature
greenFeatureNormalisingFactor = np.sum(averageGreenFeature)
areaNormalisingFactor = np.sum(areaFeature)
shapeFeatureNormalisingFactor = np.sum(shapeFeature)

for i in range(0, len(averageGreenFeature)):
    averageGreenFeature[i] = averageGreenFeature[i]/greenFeatureNormalisingFactor

for j in range(0, len(areaFeature)):
    areaFeature[j] = areaFeature[j]/areaNormalisingFactor

for k in range(0, len(shapeFeature)):
    shapeFeature[k] = shapeFeature[k]/shapeFeatureNormalisingFactor

#combine the features into one feature list
features = list(zip(averageGreenFeature, areaFeature, shapeFeature))


#create a label encoder to convert the class labels to numerics
le = preprocessing.LabelEncoder()

#convert the string labels into numbers 
labelsEncoded = le.fit_transform(labels)

accuracy = 0

for i in range(0, 500):

    #split the data into training set and testing set, with 70% for training and
    #30% for testing
    feature_train, feature_test, label_train, label_test = train_test_split(features, labelsEncoded, test_size = 0.3)

    n = 6

    #create the KNN classifier
    knn = KNeighborsClassifier(n)

    #train the model using the training sets
    knn.fit(feature_train, label_train)

    #predict the classes of the test dataset
    labels_prediction = knn.predict(feature_test)
    print("\nUsing "+str(n)+ " Nearest Neighbor Classifier:")
    print("\nTraining set size:",len(feature_train),"data samples")
    print("Testing set size:",len(feature_test), "data samples")
    print("\nAccuracy on predicting for testing data: "+ str((metrics.accuracy_score(label_test, labels_prediction))*100)+"%")
    accuracy += 100*(metrics.accuracy_score(label_test, labels_prediction))
    
print("Average accuracy over 500 runs is :"+str(accuracy/500)+"%")
    
