#feature extraction methods
import cv2
import numpy as np

#returns the mean value of the green channel stored in the segmented gray image
def extractGreenFeature(image):
    
    #get the number of the non-zero levels in the image
    total = cv2.countNonZero(image)
    
    #get the sum of all non-zero levels in the image (these correspond with the
    #green levels of the leaf as a result of the segmentation process
    sumLevels = np.sum(image)
    
    #return the average green value of the leaf
    return sumLevels/total


#returns the number of non-zero pixels in the image, thereby giving a measure of
#the area of the leaf
def extractAreaFeature(image):
    
    return cv2.countNonZero(image)

#extracts information about the shape of the leaves
def extractShapeFeature(image):
    
    #use a matrix of size 5 for the picture element
    picElement = np.ones((5,5), np.uint8)
    eroded_image = cv2.erode(image, picElement, iterations=1)
    
    #return the number of non-zero pixels in the image after erosion
    return cv2.countNonZero(eroded_image)
   




