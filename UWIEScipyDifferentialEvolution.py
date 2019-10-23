#!/usr/bin/env python
# coding: utf-8

# In[4]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
# %matplotlib inline
from tqdm import tqdm_notebook as tqdm
import time
from scipy.optimize import differential_evolution
import copy

# In[5]:


# GivenImage = cv2.imread("SampleImage1.jpg")
# GivenImage = cv2.imread("flag.png")
# GivenImage = cv2.imread("a12.jpg")
# RocksUnderWater
GivenImage = cv2.imread("RocksUnderWater.png")
#GivenImage = cv2.imread("unDRocks1.png")
GivenImage = cv2.cvtColor(GivenImage, cv2.COLOR_BGR2RGB)
# GivenImage = GivenImage[275:300,225:250]
# GivenImage = GivenImage[600:1000,2000:2500]
GivenImage = GivenImage/255.0
GivenImage = GivenImage*255
# plt.imshow(GivenImage/255)
# plt.show()
# GivenImage = GivenImage[:,:,0:1]
print(GivenImage.dtype)


# In[10]:
if(0):
	for i in range(0, GivenImage.shape[-1]):
		image = GivenImage[:, :, i]
		# image = GivenImage
		TotalNumberOfPixelsInImage = (image.shape[0])*(image.shape[1])
		NumberofPixelsPerIntensity = np.zeros(256)
		for j in range(0, 256):
			NumberofPixelsPerIntensity[j] = np.sum(image == j)
		probDensity = NumberofPixelsPerIntensity/TotalNumberOfPixelsInImage
		plt.plot(np.arange(0, 256, 1), probDensity, color='red')
		plt.fill_between(np.arange(0, 256, 1), probDensity, color='red')
		plt.show()
		plt.imshow(GivenImage[:, :, i], cmap='gray')
		plt.show()

# print(probDensity)
# plt.hist(image, bins=[0,256])
# plt.show()
#cumulImageProbDensity1 = np.cumsum(probDensity)
# print(cumulImageProbDensity1)
global itera
itera = 0


# In[7]:


def objectiveFunction(w, image, cumulImageProbDensity):
	# compute the histogram
	# print('w in the objectiveFunction is ')
	# print(w)
	image = image.astype(np.float64)
	global itera
	itera += 1
	# print('current iteration is ',itera)
	w = w[0]
	TotalNumberOfPixelsInImage = (image.shape[0])*(image.shape[1])
	# probDensity = imageProbDensity
	# from w compute the lower and upper limits
	lowerLimit = (w*255)
	upperLimit = ((1-w)*255)
	if(0):
		lowerLimit = 0.0
		upperLimit = 255.0

		# print('1measuring times in this function call')
		start = time.time()
		tempSum = 0
		for i in range(0, 256):
			if(cumulImageProbDensity[i] >= w):
				lowerLimit = i
				break
		for i in range(0, 256):
			if(cumulImageProbDensity[255-i] <= (1-w)):
				upperLimit = int(255-i)
				break
		end = time.time()
		# print('time taken is ',end-start)
		tempSum = 0
		# print(upperLimit,lowerLimit)
		if(lowerLimit < 0 or upperLimit < 0):
			print('lowerLimit,upperLimit')
			print(lowerLimit, upperLimit)
		if(lowerLimit == 0 and upperLimit == 0):
			lowerLimit = 0.0
			upperLimit = 255.0
		if(lowerLimit == upperLimit):
			print(w)

		# print('w,lowerLimit,upperLimit')
		# print(w,lowerLimit,upperLimit)
		# perform contrast stretching to map lower and upper limits to a and b
		# print('1perform contrast stretching to map')
	start = time.time()
	a = 0.0
	b = 255.0
	factor = (b-a)/(upperLimit-lowerLimit)
	# plt.imshow(image)
	# plt.show()
	if(0):
		print('factor is')
		print(factor)
		print('lowerLimit,upperLimit')
		print(lowerLimit, upperLimit)
		print('the data type of the image before intensity transformation')
		print(image.dtype)

	transformedImage = image[:]
	tempMask1 = (image <= lowerLimit)
	tempMask2 = (image >= upperLimit)
	transformedImage = np.around(a+((image-lowerLimit)*factor))
	transformedImage[tempMask1] = image[tempMask1]
	transformedImage[tempMask2] = image[tempMask2]

	if(0):
		print('data type of the transformedImage is ')
		print(transformedImage.dtype)
		print('min and max in the transformedImage')
		print(np.min(transformedImage), np.max(transformedImage))
		print('current w used ')
		print(w)
	end = time.time()
	# print('time taken is ',end-start)
	# compute the histogram of transformed image
	# print('1compute the histogram of transformed image')

	start = time.time()
	NewprobDensity, _ = np.histogram(transformedImage, 256, [0, 256])
	NewprobDensity = NewprobDensity/TotalNumberOfPixelsInImage
	end = time.time()
	# print('time taken is ',end-start)
	# compute the entropy
	# print('1compute the entropy')
	start = time.time()
	NonZeroNewProbDensity = NewprobDensity[NewprobDensity > 0]
	entropy = -1*np.sum(NonZeroNewProbDensity*np.log2(NonZeroNewProbDensity))
	end = time.time()
	# print('time taken is ',end-start)
	# compute the average gradient
	# print('1compute the average gradient')
	start = time.time()
	dy, dx = np.gradient(transformedImage)
	end = time.time()
	# print('time taken is ',end-start)
	start = time.time()
	# avg_gradient = np.sum(np.absolute(dy+(1j*dx)))/TotalNumberOfPixelsInImage
	avg_gradient = np.sum(np.sqrt((dy**2)+(dx**2)))/TotalNumberOfPixelsInImage
	end = time.time()
	# print('time taken is ',end-start)
	# return sum of these

	tempObjValue = entropy+avg_gradient
	if(0):
		print('current objectiveFunction value is')
		print(tempObjValue)
		plt.imshow(transformedImage)
		plt.show()
	return np.array([-tempObjValue])


# In[8]:


def contrastStretching(image, w):
	# compute the histogram
	print('w in contrastStretching is ',w)
	TotalNumberOfPixelsInImage = (image.shape[0])*(image.shape[1])
	NumberofPixelsPerIntensity = np.zeros(256)
	for i in range(0, 256):
		NumberofPixelsPerIntensity[i] = np.sum(image == i)
	probDensity = NumberofPixelsPerIntensity/TotalNumberOfPixelsInImage
	plt.plot(np.arange(0, 256, 1), probDensity, color='red')
	plt.fill_between(np.arange(0,256,1),probDensity, color='red')
	plt.show()
	# from w compute the lower and upper limits
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
	# perform contrast stretching to map lower and upper limits to a and b
	if(1):
		lowerLimit = (w*255)
		upperLimit = ((1-w)*255)
	a = 0
	b = 255
	factor = (b-a)/(upperLimit-lowerLimit)

	transformedImage = image[:]
	tempMask1 = (image<=lowerLimit)
	tempMask2 = (image>=upperLimit)
	transformedImage = np.around(a+((image-lowerLimit)*factor))
	transformedImage[tempMask1] = image[tempMask1]
	transformedImage[tempMask2] = image[tempMask2]
	image = transformedImage
	TotalNumberOfPixelsInImage = (image.shape[0])*(image.shape[1])
	NumberofPixelsPerIntensity = np.zeros(256)
	for j in range(0, 256):
		NumberofPixelsPerIntensity[j] = np.sum(image == j)
	probDensity = NumberofPixelsPerIntensity/TotalNumberOfPixelsInImage
	plt.plot(np.arange(0,256,1),probDensity, color='red')
	plt.fill_between(np.arange(0,256,1),probDensity, color='red')
	plt.show()
	return transformedImage


# In[14]:


def EnhanceTheImage(GivenImage):
	BestWforEachChannel = np.zeros(3)
	for i in range(0,GivenImage.shape[-1]):
		image = GivenImage[:,:,i]
		TotalNumberOfPixelsInImage = (image.shape[0])*(image.shape[1]) 
		NumberofPixelsPerIntensity = np.zeros(256)
		for j in range(0,256):
			NumberofPixelsPerIntensity[j] = np.sum(image==j)
		probDensity = NumberofPixelsPerIntensity/TotalNumberOfPixelsInImage
		cumulImageProbDensity1 = np.cumsum(probDensity)
		bounds = [(0,1)]
		result = differential_evolution(objectiveFunction, bounds,args = (GivenImage[:,:,i],cumulImageProbDensity1),disp = True,strategy = 'best1exp',tol=1e-6,popsize=40,mutation=0.8,recombination = 0.7,maxiter = 1000,workers = -1)
		BestWforEachChannel[i] = result.x[0]
	print('BestWforEachChannel')
	print(BestWforEachChannel)
	EnhancedImage = np.zeros(GivenImage.shape)
	for i in range(0,GivenImage.shape[-1]):
		EnhancedImage[:,:,i] = contrastStretching(GivenImage[:,:,i],BestWforEachChannel[i])
	return EnhancedImage        


# In[ ]:


temp = EnhanceTheImage(GivenImage)
print('min and max in the EnhancedImages are ',np.min(temp),np.max(temp))

plt.imshow(temp.astype(np.uint8))
plt.savefig('EnhancedImage.png', bbox_inches='tight')
plt.show()


plt.imshow(GivenImage.astype(np.uint8))
plt.savefig('GivenImage.png', bbox_inches='tight')
plt.show()

print('norm difference ')
print(np.linalg.norm(temp-GivenImage))
