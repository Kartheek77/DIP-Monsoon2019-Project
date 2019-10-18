import cv2
import numpy as np
import matplotlib.pyplot as plt

GivenImage = cv2.imread("SampleImage1.jpg")
GivenImage = cv2.cvtColor(GivenImage, cv2.COLOR_BGR2RGB)
plt.imshow(GivenImage)
plt.show()

RedChannel = GivenImage[:,:,0]
GreenChannel = GivenImage[:,:,1]
BlueChannel = GivenImage[:,:,2] 

fobj(w,im):
    #compute the histogram
    TotalNumberOfPixelsInImage = (image.shape[0])*(image.shape[1]) 
    NumberofPixelsPerIntensity = np.zeros(256)
    for i in range(0,256):
        NumberofPixelsPerIntensity[i] = np.sum(image==i)
    probDensity = NumberofPixelsPerIntensity/TotalNumberOfPixelsInImage
