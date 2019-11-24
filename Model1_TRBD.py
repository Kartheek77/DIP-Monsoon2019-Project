import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
# %matplotlib inline
from tqdm import tqdm_notebook as tqdm
import time
from scipy.optimize import differential_evolution
from skimage.measure import compare_ssim as ssim
import copy

#iterate over the image 
#assuming the clarity in the image is decreasing
#read every image and then calculate the objective function value
#append that objective function value to a list
#plot the objective function value vs range(0,len())
Resources = "TurbidityDataSetInputImages/"
def pixelVal(pix, r1, s1, r2, s2): 
    if (0 <= pix and pix <= r1):
    	return 0.0 
        #return (s1 / r1)*pix 
    elif (r1 < pix and pix <= r2): 
        return ((s2 - s1)/(r2 - r1)) * (pix - r1) + s1 
    else:
    	return 255.0

global histFigNum
histFigNum = 1

global itera
itera = 0

def mse(x, y):
    return np.linalg.norm(x - y)

#def objectiveFunction(wfull, cumulImageProbDensity,OriginalProbDensity):
def objectiveFunction(wfull, image1, cumulImageProbDensity):
	#global itera
	#itera += 1
	#print('current iteration is ',itera)
	TotalNumberOfPixelsInImage = (image1.shape[0])*(image1.shape[1])
	tempObjValue = 0
	for i in range(0,3):
		w = wfull[i]
		lowerLimit = 0.0
		upperLimit = 255.0
		tempSum = 0
		for j in range(0, 256):
			if(cumulImageProbDensity[i][j] >= w):
				lowerLimit = j
				break
		for j in range(0, 256):
			if(cumulImageProbDensity[i][255-j] <= (1-w)):
				upperLimit = 255.0-j
				break


		if(1):
			a = 0.0
			b = 255.0
			#a = wfull[i+3]
			#b = wfull[i+6]
			#print('lowerLimit and upperLimit in this fun call is  ',w,lowerLimit,upperLimit)
			if(lowerLimit == upperLimit):
				#transformedImage = image[:]
				transformedImage = np.empty((image1.shape[0],image1.shape[1]))
				tempMask1 = (image1[:,:,i] <= lowerLimit)
				tempMask2 = (image1[:,:,i] >= upperLimit)				
				transformedImage[tempMask1] = 0.0#image[tempMask1]#lowerLimit#
				transformedImage[tempMask2] = 255.0#image[tempMask2]#upperLimit#

			else:
				factor = (b-a)/(upperLimit-lowerLimit)
				#transformedImage = image[:]
				transformedImage = np.empty((image1.shape[0],image1.shape[1]))
				if(lowerLimit< upperLimit):
					tempMask1 = (image1[:,:,i] <= lowerLimit)
					tempMask2 = (image1[:,:,i] >= upperLimit)				
				else:
					tempMask1 = (image1[:,:,i] >= lowerLimit)
					tempMask2 = (image1[:,:,i] <= upperLimit)				
				transformedImage = np.around(a+((image1[:,:,i]-lowerLimit)*factor))
				transformedImage[tempMask1] = 0.0#image[tempMask1]#lowerLimit#
				transformedImage[tempMask2] = 255.0#image[tempMask2]#upperLimit#
				#transformedImage[tempMask1] = np.around((wfull[i+3] / lowerLimit)*image1[:,:,i][tempMask1]) 
				#transformedImage[tempMask2] = np.around(((255 - wfull[i+6])/(255 - upperLimit)) * (image1[:,:,i][tempMask2] - upperLimit) + wfull[i+6])
		

		if(0):
			pixelVal_vec = np.vectorize(pixelVal) 
			transformedIntensities = pixelVal_vec(range(0,256), lowerLimit, 0, upperLimit, 255)
			NewprobDensity = np.zeros(256)
			for j in range(0,256):
				tempTI = []
				for k in range(0,256):
					if(int(transformedIntensities[k])==j):
						tempTI.append(k)
				if(len(tempTI)!=0):
					for p in tempTI:
						NewprobDensity[j] +=  OriginalProbDensity[i][p]
			#for j in range(0,256):
				#NewprobDensity[int(transformedIntensities[j])] += OriginalProbDensity[i][j]



			#print('NewprobDensity total is ',np.sum(NewprobDensity))
		if(0):
			NewprobDensity = np.empty(256) 
			for j in range(0,256):
				NewprobDensity[j] = OriginalProbDensity[i][int(transformedIntensities[j])]
				NewprobDensity = NewprobDensity/TotalNumberOfPixelsInImage
		if(1):
			NewprobDensity, _ = np.histogram(transformedImage, 256, [0, 256])
		NewprobDensity = NewprobDensity/TotalNumberOfPixelsInImage

		NonZeroNewProbDensity = NewprobDensity[NewprobDensity > 0]
		entropy = -1*np.sum(NonZeroNewProbDensity*np.log2(NonZeroNewProbDensity))
		dy, dx = np.gradient(transformedImage)
		avg_gradient = np.sum(np.sqrt((dy**2)+(dx**2)))/TotalNumberOfPixelsInImage
		#tempObjValue += (entropy+avg_gradient)
		#tempMean = np.sum(range(0,256)*NewprobDensity)
		#tempVariance = np.sum(((range(0,256)-tempMean)**2)*NewprobDensity)
		#avg_gradient = tempVariance
		tempObjValue += (entropy+avg_gradient)
	return np.array([-tempObjValue])


def contrastStretching(image, w, s1, s2):
	# compute the histogram
	#print('w in contrastStretching is ',w)
	TotalNumberOfPixelsInImage = (image.shape[0])*(image.shape[1])
	NumberofPixelsPerIntensity = np.zeros(256)
	for i in range(0, 256):
		NumberofPixelsPerIntensity[i] = np.sum(image == i)
	probDensity = NumberofPixelsPerIntensity/TotalNumberOfPixelsInImage
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
	pixelVal_vec = np.vectorize(pixelVal) 
	# Apply contrast stretching.
	print('w,lowerLimit,upperLimit are',w,lowerLimit,upperLimit)
	#transformedImage = pixelVal_vec(image, lowerLimit, s1, upperLimit, s2)
	transformedImage = pixelVal_vec(image, lowerLimit, 0, upperLimit, 255)
	transformedImage = np.clip(np.round(transformedImage),0,255)
	return transformedImage

def EnhanceTheImage(GivenImage):
	#BestWforEachChannel = np.zeros(3)
	cumulImageProbDensity1 = []
	probDensity1 = []
	for i in range(0,GivenImage.shape[-1]):
		image = GivenImage[:,:,i]
		TotalNumberOfPixelsInImage = (image.shape[0])*(image.shape[1]) 
		NumberofPixelsPerIntensity = np.zeros(256)
		for j in range(0,256):
			NumberofPixelsPerIntensity[j] = np.sum(image==j)
		probDensity = NumberofPixelsPerIntensity/TotalNumberOfPixelsInImage
		probDensity1.append(probDensity)
		cumulImageProbDensity1.append(np.cumsum(probDensity))

	#if(0):
	#cumulImageProbDensity1 = 5
	lowerBound = 0.0
	upperBound = 0.45
	s1LowerBound = 0.0
	s1UpperBound = 127.0
	s2LowerBound = 128.0
	s2UpperBound = 255.0
	bounds1 = [(lowerBound,upperBound)]*3
	#bounds2 = [(s1LowerBound,s1UpperBound)]*3
	#bounds3 = [(s2LowerBound,s2UpperBound)]*3
	bounds = bounds1#+bounds2+bounds3
	#print('the bounds are ')
	#print(bounds)
	print('starting the DEAlgo')
	#result = differential_evolution(objectiveFunction, bounds,args = (cumulImageProbDensity1,probDensity1),disp = True,strategy = 'best1bin',tol=1e-4,popsize=40,mutation=0.8,recombination = 0.7,maxiter = 200,workers = 5) 
	result = differential_evolution(objectiveFunction, bounds,args = (GivenImage,cumulImageProbDensity1),disp = True,strategy = 'best1bin',tol=0.1,popsize=40,mutation=0.8,recombination = 0.7,maxiter = 100,workers = 30,updating= 'immediate') 
	BestWforEachChannel = result.x

	#print('BestWforEachChannel')
	#print(BestWforEachChannel)

	EnhancedImage = np.zeros(GivenImage.shape)
	for i in range(0,GivenImage.shape[-1]):
		EnhancedImage[:,:,i] = contrastStretching(GivenImage[:,:,i],BestWforEachChannel[i],0,255)
		#EnhancedImage[:,:,i] = contrastStretching(GivenImage[:,:,i],BestWforEachChannel[i],BestWforEachChannel[i+3],BestWforEachChannel[i+6])
	return EnhancedImage        





def objFunctionValue(GivenImage):
	global histFigNum
	tempObjValue = 0
	tempObjEntropyValue = 0
	tempObjAvg_GradValue = 0
	TotalNumberOfPixelsInImage = (GivenImage.shape[0])*(GivenImage.shape[1])
	for i in range(0,GivenImage.shape[-1]):
		image = GivenImage[:,:,i]
		probDensity, _ = np.histogram(image, 256, [0, 256])
		probDensity = probDensity/TotalNumberOfPixelsInImage
		if(0):
			plt.plot(np.arange(0, 256, 1), probDensity, color='red')
			plt.fill_between(np.arange(0, 256, 1), probDensity, color='red')
			plt.savefig(str(histFigNum)+'thHistogram.jpg')
		histFigNum += 1		
		NonZeroNewProbDensity = probDensity[probDensity > 0]
		entropy = -1*np.sum(NonZeroNewProbDensity*np.log2(NonZeroNewProbDensity))
		#dy, dx = np.gradient(image)
		#avg_gradient = np.sum(np.sqrt((dy**2)+(dx**2)))/TotalNumberOfPixelsInImage
		tempMean = np.sum(range(0,256)*probDensity)
		tempVariance = np.sum(((range(0,256)-tempMean)**2)*probDensity)
		avg_gradient = tempVariance
		tempObjEntropyValue += entropy
		tempObjAvg_GradValue += avg_gradient
		tempObjValue += (entropy+avg_gradient)
	return tempObjValue,tempObjEntropyValue,tempObjAvg_GradValue
		

ObjFuncValueForEachImage = []
EntropyObjFuncValueForEachImage = []
AvgGradObjFuncValueForEachImage = []
SDIValueForEachImage  = []
#MSEValueForEachImage = []
TurbidMSEValueForEachImage = []
EnhancedMSEValueForEachImage = []


OriginalImage = cv2.imread(Resources+str(1)+".jpg")
OriginalImage = cv2.cvtColor(OriginalImage, cv2.COLOR_BGR2RGB)
plt.axis('off')
plt.imshow(OriginalImage.astype(np.uint8))
plt.savefig('RealImage.png', bbox_inches='tight',transparent=True,pad_inches=0)

for i in range(2,20):
	print('Image Number: ',i)
	if(i<10):
		tempImageName = str(i)#+".jpg"
	else:
		tempImageName = "a"+str(i)#+".jpg"
	#read the image 
	tempGivenImage = cv2.imread(Resources+tempImageName+".jpg")
	tempGivenImage = cv2.cvtColor(tempGivenImage, cv2.COLOR_BGR2RGB)
	temp1 =0.0
	temp2 =0.0
	temp3 =0.0
	for j in range(0,3):
		tempGivenImagetemp = tempGivenImage[:,:,j:j+1]
		#find the objective function value and append to the list
		temp = objFunctionValue(tempGivenImagetemp)
		temp1 += temp[0]
		temp2 += temp[1]
		temp3 += temp[2]
	ssimtemp = ssim(OriginalImage, tempGivenImage,data_range=tempGivenImage.max() - tempGivenImage.min(),multichannel=True)
	print('SDI Value for this Image')
	print((1.0-ssimtemp)/0.639512208985103)
	ObjFuncValueForEachImage.append(temp1)
	EntropyObjFuncValueForEachImage.append(temp2)
	AvgGradObjFuncValueForEachImage.append(temp3)
	SDIValueForEachImage.append((1.0-ssimtemp)/0.639512208985103)
	tempGivenImage = tempGivenImage.astype(np.float64)
	temp = EnhanceTheImage(tempGivenImage)
	plt.axis('off')
	plt.imshow(tempGivenImage.astype(np.uint8))
	plt.savefig(tempImageName+'GivenImage.png', bbox_inches='tight',transparent=True,pad_inches=0)
	plt.axis('off')
	plt.imshow(temp.astype(np.uint8))
	plt.savefig(tempImageName+'EnhancedImage.png', bbox_inches='tight',transparent=True,pad_inches=0)
	#MSEValueForEachImage.append(mse(OriginalImage,temp))
	TurbidMSEValueForEachImage.append(mse(OriginalImage,tempGivenImage))
	EnhancedMSEValueForEachImage.append(mse(OriginalImage,temp))

plt.figure(figsize=(10, 10))
plt.plot(SDIValueForEachImage, TurbidMSEValueForEachImage, linestyle='--', marker='o', color='b',label = 'TubidImageMSE')
plt.plot(SDIValueForEachImage, EnhancedMSEValueForEachImage, linestyle='-', marker='x', color='r',label = 'EnhancedImageMSE')
#plt.plot(SDIValueForEachImage, MSEValueForEachImage, 'ro')
plt.title('SDI Vs MSE with True Image')
plt.xlabel('Structural Degradation Index')
plt.ylabel('MSE')
plt.legend(loc='best')
plt.show()

#plot the graph
plt.figure(figsize=(30, 10))
plt.tight_layout()
plt.subplot(131)
plt.plot(SDIValueForEachImage, ObjFuncValueForEachImage, 'ro')
#plt.plot(range(0,len(ObjFuncValueForEachImage)), ObjFuncValueForEachImage, 'ro')
plt.title('SDI Vs (Entropy + Variance)')
plt.xlabel('Structural Degradation Index')
plt.ylabel('objective function value')
#plt.show()

plt.subplot(132)
plt.plot(SDIValueForEachImage, EntropyObjFuncValueForEachImage, 'ro')
#plt.plot(range(0,len(EntropyObjFuncValueForEachImage)), EntropyObjFuncValueForEachImage, 'ro')
plt.title(' SDI Vs Entropy')
plt.xlabel('Structural Degradation Index')
plt.ylabel('Entropy value')
#plt.show()

plt.subplot(133)
plt.plot(SDIValueForEachImage, AvgGradObjFuncValueForEachImage, 'ro')
#plt.plot(range(0,len(AvgGradObjFuncValueForEachImage)), AvgGradObjFuncValueForEachImage, 'ro')
plt.title(' SDI Vs Variance')
plt.xlabel('Structural Degradation Index')
plt.ylabel('Variance value')
#plt.tight_layout()
plt.show() 


if(0):
	plt.plot(range(0,len(ObjFuncValueForEachImage)), ObjFuncValueForEachImage, 'ro')
	#plt.axis([0, , 0, 20])
	plt.title('Milk Data Set Clarity Vs (Entropy + Avg Gradient)')
	plt.xlabel('clarity (decreasing order)')
	plt.ylabel('objective function value')
	plt.show()
