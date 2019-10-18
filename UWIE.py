import cv2
import numpy as np
import matplotlib.pyplot as plt

GivenImage = cv2.imread("SampleImage1.jpg")
GivenImage = cv2.cvtColor(GivenImage, cv2.COLOR_BGR2RGB)
plt.imshow(GivenImage)
plt.show()
