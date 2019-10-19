import cv2
import numpy as np 
import matplotlib.pyplot as plt
%matplotlib inline

def objectiveFunction(w,image):
    #compute the histogram
    TotalNumberOfPixelsInImage = (image.shape[0])*(image.shape[1]) 
    NumberofPixelsPerIntensity = np.zeros(256)
    for i in range(0,256):
        NumberofPixelsPerIntensity[i] = np.sum(image==i)
    probDensity = NumberofPixelsPerIntensity/TotalNumberOfPixelsInImage
    #from w compute the lower and upper limits
    lowerLimit = 0
    upperLimit = 255
    tempSum = 0
    for i in range(0,256):
        tempSum += probDensity[i]
        if(tempSum>w):
            lowerLimit = i
            break
    tempSum = 0
    for i in range(0,256):
        tempSum += probDensity[int(255-i)]
        if(tempSum>w):
            upperLimit = int(255-i)
            break
    #perform contrast stretching to map lower and upper limits to a and b
    a = 0
    b = 255
    factor = ((b-a)/(upperLimit-lowerLimit))
    transformedImage = np.round_(a+((im-lowerLimit)*factor))    
    #compute the histogram of transformed image
    NumberofPixelsPerIntensity = np.zeros(256)
    for i in range(0,256):
        NumberofPixelsPerIntensity[i] = np.sum(transformedImage==i)
    NewprobDensity = NumberofPixelsPerIntensity/TotalNumberOfPixelsInImage
    #compute the entropy
    entropy = -1*np.sum(NewprobDensity*np.log2(NewprobDensity))
    #compute the average gradient
    dy,dx = np.gradient(transformedImage)
    avg_gradient = np.sum(np.sqrt((dy*dy)+(dx*dx)))/TotalNumberOfPixelsInImage
    #return sum of these
    return entropy+avg_gradient

def contrastStretching(image,w):
    #compute the histogram
    TotalNumberOfPixelsInImage = (image.shape[0])*(image.shape[1]) 
    NumberofPixelsPerIntensity = np.zeros(256)
    for i in range(0,256):
        NumberofPixelsPerIntensity[i] = np.sum(image==i)
    probDensity = NumberofPixelsPerIntensity/TotalNumberOfPixelsInImage
    #from w compute the lower and upper limits
    lowerLimit = 0
    upperLimit = 255
    tempSum = 0
    for i in range(0,256):
        tempSum += probDensity[i]
        if(tempSum>w):
            lowerLimit = i
            break
    tempSum = 0
    for i in range(0,256):
        tempSum += probDensity[int(255-i)]
        if(tempSum>w):
            upperLimit = int(255-i)
            break
    #perform contrast stretching to map lower and upper limits to a and b
    a = 0
    b = 255
    factor = ((b-a)/(upperLimit-lowerLimit))
    transformedImage = np.round_(a+((im-lowerLimit)*factor))
    return transformedImage

def EnhanceTheImage(GivenImage):
    BestWforEachChannel = np.zeros(3)
    for i in range(0,3):
        BestWforEachChannel[i] = DEAlgo
    EnhancedImage = np.zeros(GivenImage)
    for i in range(0,3):
        EnhancedImage[:,:,i] = contrastStretching(GivenImage[:,:,i],BestWforEachChannel[i])
    return EnhancedImage        
