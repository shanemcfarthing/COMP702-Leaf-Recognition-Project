#segmentation methods
import cv2
import numpy as np
from FeatureExtraction import *

#function to perform preprocessing, segmentation, and normalisation of image 
def segmentation(image):
    
    #get the image in the Hue Saturation Value scale, as this will be used to
    #get good thresholding results
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    #obtain a greyscale copy of the colour image to store the saturation channel in
    saturationImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #obtain a greyscale copy of the colour image to store the green channel in
    greenChannelImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #store the saturation channel in the grayscale image created for this purpose
    cv2.mixChannels([hsv_image], [saturationImage], (1, 0))
    
    #store the green channel in the grayscale image created for this purpose
    cv2.mixChannels([image], [greenChannelImage], (1, 0))
    
    #perform the image thresholding using the hsv image for good results
    #automatic thresholding is used to ensure that a good threshold value is used
    #intensity levels are set to 0 or 1 depending on their relation to the threshold
    thresholdResult = cv2.threshold(saturationImage, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    #use a median blur filter to remove any excess noise that arose from the thresholding step
    filteredImage = cv2.medianBlur(thresholdResult[1], 3)
    
    #find the points in the image belonging to the leaf (these are all non-zero)
    #and use them to determine the size of the smallest bounding rectangle
    leafPoints = cv2.findNonZero(filteredImage)
    rectangleInfo = cv2.boundingRect(leafPoints)
    
    #black out the background, resulting in a grey leaf on a black background
    #this is done to better isolate and segment the region of interest
    segmentedImage = np.multiply(greenChannelImage, filteredImage)
    
    #crop the image using the bounding rectangle information to remove excess information
    #which does not contribute towards the classification (i.e. the background)
    croppedImage = segmentedImage[rectangleInfo[1]:rectangleInfo[1]+rectangleInfo[3], rectangleInfo[0]:rectangleInfo[0]+rectangleInfo[2]]
    
    #resize the images so that they are all a standard size
    return cv2.resize(croppedImage, (200, 200))

