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
def pixelVal(pix, r1, s1, r2, s2): 
    if (0 <= pix and pix <= r1): 
        return (s1 / r1)*pix 
    elif (r1 < pix and pix <= r2): 
        return ((s2 - s1)/(r2 - r1)) * (pix - r1) + s1 
    else: 
        return ((255 - s2)/(255 - r2)) * (pix - r2) + s2


def objectiveFunction(wfull, image1, cumulImageProbDensity):
	# compute the histogram
	#print('wfull in the objectiveFunction is ')
	#print(wfull)
	#print('the w supplied to the function call is         ',wfull)
	image1 = image1.astype(np.float64)
	global itera
	itera += 1
	#print('current iteration is ',itera)
	 #w = w[0]
	TotalNumberOfPixelsInImage = (image1.shape[0])*(image1.shape[1])
	# probDensity = imageProbDensity
	# from w compute the lower and upper limits
	tempObjValue = 0
	for i in range(0,3):
		w = wfull[i]
		image = GivenImage[:,:,i]
		lowerLimit = (w*255)
		upperLimit = ((1-w)*255)

		if(1):
			lowerLimit = 0.0
			upperLimit = 255.0

			# print('1measuring times in this function call')
			start = time.time()
			tempSum = 0
			for j in range(0, 256):
				if(cumulImageProbDensity[i][j] >= w):
					lowerLimit = j
					break
			for j in range(0, 256):
				if(cumulImageProbDensity[i][255-j] <= (1-w)):
					upperLimit = 255.0-j
					break

		
		if(0):
			a = 0.0
			b = 255.0
			#print('lowerLimit and upperLimit in this fun call is  ',w,lowerLimit,upperLimit)
			if(lowerLimit == upperLimit):
				transformedImage = image[:]
				tempMask1 = (image <= lowerLimit)
				tempMask2 = (image >= upperLimit)				
				transformedImage[tempMask1] = 0.0#image[tempMask1]#lowerLimit#
				transformedImage[tempMask2] = 255.0#image[tempMask2]#upperLimit#

			else:
				factor = (b-a)/(upperLimit-lowerLimit)
				transformedImage = image[:]
				if(lowerLimit< upperLimit):
					tempMask1 = (image <= lowerLimit)
					tempMask2 = (image >= upperLimit)				
				else:
					tempMask1 = (image >= lowerLimit)
					tempMask2 = (image <= upperLimit)				
				transformedImage = np.around(a+((image-lowerLimit)*factor))
				transformedImage[tempMask1] = 0.0#image[tempMask1]#lowerLimit#
				transformedImage[tempMask2] = 255.0#image[tempMask2]#upperLimit#
				#transformedImage[tempMask1] = image[tempMask1]#lowerLimit#
				#transformedImage[tempMask2] = image[tempMask2]#upperLimit#

		if(1):
			# Vectorize the function to apply it to each value in the Numpy array. 
			pixelVal_vec = np.vectorize(pixelVal) 
	  
			# Apply contrast stretching.
			transformedImage = pixelVal_vec(image, lowerLimit, wfull[i+3], upperLimit, wfull[i+6]) 



		NewprobDensity, _ = np.histogram(transformedImage, 256, [0, 256])
		NewprobDensity = NewprobDensity/TotalNumberOfPixelsInImage
		#print('number of elements in NewprobDensity is ',NewprobDensity.shape[0])
		
		NonZeroNewProbDensity = NewprobDensity[NewprobDensity > 0]
		entropy = -1*np.sum(NonZeroNewProbDensity*np.log2(NonZeroNewProbDensity))

		dy, dx = np.gradient(transformedImage)
		end = time.time()

		start = time.time()
		# avg_gradient = np.sum(np.absolute(dy+(1j*dx)))/TotalNumberOfPixelsInImage
		avg_gradient = np.sum(np.sqrt((dy**2)+(dx**2)))/TotalNumberOfPixelsInImage
		end = time.time()
		# print('time taken is ',end-start)
		# return sum of these
		#avg_gradient = 0.0
		#avg_gradient = 0.0

		tempObjValue += (entropy+avg_gradient)
		#print('cumulImageProbDensity is ')
		#print(cumulImageProbDensity[i])
		#print('End of cumulImageProbDensity')
		

	
	#print('tempObjValue in this objectiveFunction call is ',tempObjValue)

	return np.array([-tempObjValue])


# In[8]:


def contrastStretching(image, w, s1, s2):
	# compute the histogram
	print('w in contrastStretching is ',w*255)
	TotalNumberOfPixelsInImage = (image.shape[0])*(image.shape[1])
	NumberofPixelsPerIntensity = np.zeros(256)
	for i in range(0, 256):
		NumberofPixelsPerIntensity[i] = np.sum(image == i)
	probDensity = NumberofPixelsPerIntensity/TotalNumberOfPixelsInImage
	#plt.plot(np.arange(0, 256, 1), probDensity, color='red')
	#plt.fill_between(np.arange(0,256,1),probDensity, color='red')
	#plt.show()
	# from w compute the lower and upper limits
	if(1):
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
	if(0):
		lowerLimit = (w*255)
		upperLimit = ((1-w)*255)
	
	if(0):
		a = 0.0
		b = 255.0
		if(lowerLimit == upperLimit):
				transformedImage = image[:]
				tempMask1 = (image <= lowerLimit)
				tempMask2 = (image >= upperLimit)				
				transformedImage[tempMask1] = 0.0#image[tempMask1]#lowerLimit#
				transformedImage[tempMask2] = 255.0#image[tempMask2]#upperLimit#
		else:
			factor = (b-a)/(upperLimit-lowerLimit)
			transformedImage = image[:]
			if(lowerLimit<upperLimit):
				tempMask1 = (image <= lowerLimit)
				tempMask2 = (image >= upperLimit)				
			else:
				tempMask1 = (image >= lowerLimit)
				tempMask2 = (image <= upperLimit)				
			transformedImage = np.around(a+((image-lowerLimit)*factor))
			transformedImage[tempMask1] = 0.0#image[tempMask1]#lowerLimit#
			transformedImage[tempMask2] = 255.0#image[tempMask2]#upperLimit#
			#transformedImage[tempMask1] = image[tempMask1]#lowerLimit#
			#transformedImage[tempMask2] = image[tempMask2]#upperLimit#

	if(1):
		pixelVal_vec = np.vectorize(pixelVal) 
		# Apply contrast stretching.
		transformedImage = pixelVal_vec(image, lowerLimit, s1, upperLimit, s2) 


	if(0):
		image = transformedImage
		TotalNumberOfPixelsInImage = (image.shape[0])*(image.shape[1])
		NumberofPixelsPerIntensity = np.zeros(256)
		for j in range(0, 256):
			NumberofPixelsPerIntensity[j] = np.sum(image == j)
		probDensity = NumberofPixelsPerIntensity/TotalNumberOfPixelsInImage
		#plt.plot(np.arange(0,256,1),probDensity, color='red')
		#plt.fill_between(np.arange(0,256,1),probDensity, color='red')
		#plt.show()
	return transformedImage


# In[14]:


def EnhanceTheImage(GivenImage):
	if(1):
		BestWforEachChannel = np.zeros(3)
		cumulImageProbDensity1 = []
		for i in range(0,GivenImage.shape[-1]):
			image = GivenImage[:,:,i]
			TotalNumberOfPixelsInImage = (image.shape[0])*(image.shape[1]) 
			NumberofPixelsPerIntensity = np.zeros(256)
			for j in range(0,256):
				NumberofPixelsPerIntensity[j] = np.sum(image==j)
			probDensity = NumberofPixelsPerIntensity/TotalNumberOfPixelsInImage
			cumulImageProbDensity1.append(np.cumsum(probDensity))

	#if(0):
	#cumulImageProbDensity1 = 5
	lowerBound = 0.0
	upperBound = 0.1
	s1LowerBound = 0.0
	s1UpperBound = 127.0
	s2LowerBound = 128.0
	s2UpperBound = 255.0
	bounds1 = [(lowerBound,upperBound)]*3
	bounds2 = [(s1LowerBound,s1UpperBound)]*3
	bounds3 = [(s2LowerBound,s2UpperBound)]*3
	bounds = bounds1+bounds2+bounds3
	print('the bounds are ')
	print(bounds)
	print('starting the DEAlgo')
	result = differential_evolution(objectiveFunction, bounds,args = (GivenImage,cumulImageProbDensity1),disp = True,strategy = 'best1bin',tol=1e-4,popsize=40,mutation=0.8,recombination = 0.7,maxiter = 100,workers = 10) 
	BestWforEachChannel = result.x

	print('BestWforEachChannel')
	print(BestWforEachChannel)

	EnhancedImage = np.zeros(GivenImage.shape)
	for i in range(0,GivenImage.shape[-1]):
		EnhancedImage[:,:,i] = contrastStretching(GivenImage[:,:,i],BestWforEachChannel[i],BestWforEachChannel[i+3],BestWforEachChannel[i+6])
	return EnhancedImage        


# In[ ]:
# GivenImage = cv2.imread("SampleImage1.jpg")
#GivenImage = cv2.imread("1flag.png")
#GivenImage = cv2.imread("2flag.JPG")

GivenImage = cv2.imread("a12.jpg")
#GivenImage = cv2.imread("UW3.png")
GivenImage = cv2.cvtColor(GivenImage, cv2.COLOR_BGR2RGB)
GivenImage = GivenImage[950:1050,1600:1900,:]
print('min and max in the griven image are ',np.min(GivenImage),np.max(GivenImage))
print(GivenImage.dtype)
plt.imshow(GivenImage)
plt.show()
GivenImage = GivenImage.astype(np.float64)
print('min and max in the griven image are ',np.min(GivenImage),np.max(GivenImage))
print(GivenImage.dtype)
plt.imshow(GivenImage.astype(np.uint8))
plt.show()
# RocksUnderWater
#GivenImage = cv2.imread("RocksUnderWater.png")
#GivenImage = cv2.imread("unDRocks1.png")

# GivenImage = GivenImage[275:300,225:250]
# GivenImage = GivenImage[600:1000,2000:2500]
#GivenImage = GivenImage/255.0
#GivenImage = GivenImage*255.0
# plt.imshow(GivenImage/255)
# plt.show()
# GivenImage = GivenImage[:,:,0:1]
print(GivenImage.dtype)

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




temp = EnhanceTheImage(GivenImage)
print('min and max in the EnhancedImages are ',np.min(temp),np.max(temp))

plt.imshow(temp.astype(np.uint8))
plt.savefig('EnhancedImage.png', bbox_inches='tight')
plt.show()


plt.imshow(GivenImage.astype(np.uint8))
plt.savefig('GivenImage.png', bbox_inches='tight')
plt.show()

image = temp
gaussian_3 = cv2.GaussianBlur(image, (9,9), 10.0) 
unsharp_image = cv2.addWeighted(image, 1.5, gaussian_3, -0.5, 0, image)
#plt.imshow(unsharp_image.astype(np.uint8))
#plt.show()


print('norm difference ')
print(np.linalg.norm(temp-GivenImage))
