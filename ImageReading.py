#method for reading filepath names and calling segmentation and feature extraction methods
import cv2
from Segmentation import *
from FeatureExtraction import *
import numpy as np

def readImages(indexFilePath, greenFeatureList, pixelFeatureList, shapeFeatureList, labelList):
    
    plantCategories = ["abies_concolor", "amelanchier_arborea",
                       "carya_cordiformis", "chamaecyparis_pisifera",
                       "cornus_florida", "juniperus_virginiana", 
                       "liriodendron_tulipifera", "malus_pumila",
                       "metasequoia_glyptostroboides", "pinus_bungeana",
                       "ptelea_trifoliata", "quercus_falcata", 
                       "quercus_velutina", "aesculus_pavi", 
                       "acer_palmatum", "catalpa_speciosa",
                       "tsuga_canadensis", "zelkova_serrata",
                       "koelreuteria_paniculata"]
    
    #use this to track process in shell
    s = 0
    counter = 1200
    
    fp = open(indexFilePath)
    for i, line in enumerate(fp):
        
            #split the line by tab characters due to the format of the file
            lineContents = line.split("\t")
            
            if any(plant in lineContents[1] for plant in plantCategories):
            
                if not counter > 0:
                    break
                if lineContents[4] == "field\n":
                    counter -= 1   
                    s+=1
                    print("Sample",s,"processed")
                
                    #the file path that we want is the second word on the line
                    filePath = lineContents[1]
                    
                    #the plant that the leaf belongs to is the fourth word on the line
                    plantLabel = lineContents[3]
                
                    #load the image
                    image = cv2.imread(filePath, cv2.IMREAD_COLOR)
                    
                    #get the segmented image
                    segmentedImage = segmentation(image)
                    
                    #get the green value feature
                    greenFeature = extractGreenFeature(segmentedImage)
                    
                    #get the leaf area feature
                    areaFeature = extractAreaFeature(segmentedImage)
                    
                    #get the leaf shape feature
                    shapeFeature = extractShapeFeature(segmentedImage)
                    
                    #add the images features and label to the relevant lists
                    greenFeatureList.append(greenFeature)
                    pixelFeatureList.append(areaFeature)
                    shapeFeatureList.append(shapeFeature)
                    labelList.append(plantLabel)
                    
                    #at this point, image processing has completed for this image
                    #and next image must be obtained and processed
               