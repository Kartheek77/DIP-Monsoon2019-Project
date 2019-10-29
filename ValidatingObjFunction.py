import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
# %matplotlib inline
from tqdm import tqdm_notebook as tqdm
import time
from scipy.optimize import differential_evolution
import copy

#iterate over the image 
#assuming the clarity in the image is decreasing
#read every image and then calculate the objective function value
#append that objective function value to a list
#plot the objective function value vs range(0,len())


def objFunctionValue(GivenImage):
	tempObjValue = 0
	tempObjEntropyValue = 0
	tempObjAvg_GradValue = 0
	TotalNumberOfPixelsInImage = (GivenImage.shape[0])*(GivenImage.shape[1])
	for i in range(0,3):
		image = GivenImage[:,:,i]
		probDensity, _ = np.histogram(image, 256, [0, 256])
		probDensity = probDensity/TotalNumberOfPixelsInImage		
		NonZeroNewProbDensity = probDensity[probDensity > 0]
		entropy = -1*np.sum(NonZeroNewProbDensity*np.log2(NonZeroNewProbDensity))
		dy, dx = np.gradient(image)
		avg_gradient = np.sum(np.sqrt((dy**2)+(dx**2)))/TotalNumberOfPixelsInImage
		end = time.time()
		tempObjEntropyValue += entropy
		tempObjAvg_GradValue += avg_gradient
		tempObjValue += (entropy+avg_gradient)
	return tempObjValue,tempObjEntropyValue,tempObjAvg_GradValue
		

ObjFuncValueForEachImage = []
EntropyObjFuncValueForEachImage = []
AvgGradObjFuncValueForEachImage = []

for i in range(1,10):
	print(i)
	tempImageName = str(i)+".jpg"
	#read the image 
	tempGivenImage = cv2.imread(tempImageName)
	tempGivenImage = cv2.cvtColor(tempGivenImage, cv2.COLOR_BGR2RGB)
	#find the objective function value and append to the list
	temp = objFunctionValue(tempGivenImage)
	ObjFuncValueForEachImage.append(temp[0])
	EntropyObjFuncValueForEachImage.append(temp[1])
	AvgGradObjFuncValueForEachImage.append(temp[2])

for i in range(10,20):
	print(i)
	tempImageName = "a"+str(i)+".jpg"
	#read the image 
	tempGivenImage = cv2.imread(tempImageName)
	tempGivenImage = cv2.cvtColor(tempGivenImage, cv2.COLOR_BGR2RGB)
	#find the objective function value and append to the list
	temp = objFunctionValue(tempGivenImage)
	ObjFuncValueForEachImage.append(temp[0])
	EntropyObjFuncValueForEachImage.append(temp[1])
	AvgGradObjFuncValueForEachImage.append(temp[2])

#plot the graph
plt.figure(figsize=(30, 10))
plt.tight_layout()
plt.subplot(131)
plt.plot(range(0,len(ObjFuncValueForEachImage)), ObjFuncValueForEachImage, 'ro')
plt.title('Milk Data Set Clarity Vs (Entropy + Avg Gradient)')
plt.xlabel('clarity (decreasing order)')
plt.ylabel('objective function value')
#plt.show()

plt.subplot(132)
plt.plot(range(0,len(EntropyObjFuncValueForEachImage)), EntropyObjFuncValueForEachImage, 'ro')
plt.title('Milk Data Set Clarity Vs Entropy')
plt.xlabel('clarity (decreasing order)')
plt.ylabel('Entropy value')
#plt.show()

plt.subplot(133)
plt.plot(range(0,len(AvgGradObjFuncValueForEachImage)), AvgGradObjFuncValueForEachImage, 'ro')
plt.title('Milk Data Set Clarity Vs Avg Gradient')
plt.xlabel('clarity (decreasing order)')
plt.ylabel('Avg_Gradient value')
#plt.tight_layout()
plt.show() 


if(0):
	plt.plot(range(0,len(ObjFuncValueForEachImage)), ObjFuncValueForEachImage, 'ro')
	#plt.axis([0, , 0, 20])
	plt.title('Milk Data Set Clarity Vs (Entropy + Avg Gradient)')
	plt.xlabel('clarity (decreasing order)')
	plt.ylabel('objective function value')
	plt.show()