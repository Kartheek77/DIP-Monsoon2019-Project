# DIP Monsoon2019 Project  

## Underwater Image Enhancement Based On Contrast Adjustment via Differential Evolution Algorithm   
- Team Name     : kv1   
- Project ID    : 39
## Team Members  : 

  1. 2018801010 - Karnati Venkata Kartheek    
  2. 2018900014 - Arun Kumar Subramaniam

## Install Dependencies

- Install virtualenv, virtualenvwrapper   
  `sudo pip3 install virtualenv virtualenvwrapper`  
  
- Create a UWIE Virtual environment and activate it   
  `mkvirtualenv UWIE`   
  `workon UWIE`  

- Install required Python Packages   
  `pip3 install -r requiremnts.txt`
 
 ## Usage   
 - Contrast Stretching Generally Require (r1,s1) and (r2,s2) as the input
 - Here r1, r2 are determined by the Differential Evolution algorithm as specified in the research paper. s1 and s2 are our choice.
 - We tried Three models,
 - In Model1, we set s1 = 0 and s2 = 255
 - In Model2, s1 and s2 are also considered as variables and will be determined by DE Algo
 - In Model3, Objective Function is changed to Entropy + Variance (within Lower and Upper Limits) 
 - `UWIEf_MODEL1.ipynb` implements Model1 on the Images Given in the associated research paper. It takes input from the `PaperImages` Folder and displays the given and enhanced Images in the Notebook Itself.
 - `UWIEf_MODEL2.py` implements Models2. As more number of variable are involved in optimization problem. This need multiprocessing. Thus this file is ran on ADA. This file takes input from `PaperImages` and outputs the enhanced images. The Folder `Model2_UWIE_OutputImages` contain the images enhanced by this method. `Model2_UWIE_Output.txt` file contains the shell output while running on ADA
 
 - `Model1_TRBD.py` apply's Model1 on [TurbidityDataSetImages](http://amandaduarte.com.br/turbid/Turbid_Dataset.pdf). These images are present in `TurbidityDataSetInputImages` Folder. This file takes input from here and outputs Enhanced Images. Since images are of large size multiprocessing is needed here. This code is also ran on ADA. The output images are in the Folder   `EnhancedAndGivenImagesTurbidDataSetUsingOriginalMethod`. `Model1_TRBD_Output.txt` contains the shell output while running on ADA. The notebook `Model1_TRBD_RESULTS.ipynb` displays both given and enhanced images for comparision.
 
 - `Model3_TRBD.py` is similar to `Model1_TRBD.py`. But here we use Model3 instead. It takes input from `TurbidityDataSetInputImages` Folder.Since images are of large size multiprocessing is needed here. This code is also ran on ADA. The output images are in the Folder   `EnhancedAndGivenImagesTurbidDataSetUsingVarianceApproach`  .  `Model3_TRBD_Output.txt` contains the shell output while running on ADA. The notebook `Model3_TRBD_RESULTS.ipynb` displays both given and enhanced images for comparision.
 - The file `gridSearch.py` contains the code for grid search. Here mutation and cross overconstants are the hyperparameters. Since generally its better to have highter population size. We fix population size as 40. Thus remaining only two hyperparameters in range 0 to 1. We choose values in steps of 0.1. We computer the objective value for (9*9 = 81) combinations. And select a combination of mutation and crossover coefficients which maximize the objective value. The ouput is `gridSearch.csv`. The file `gridSearchOutput.txt` contains shell output. Here in this file we take a sub image to make the calculation faster. Input is taken from the `PaperImages' Folder. 
 - The notebook `PSNRvsSubjectiveRatings.ipynb` is where we claculate PSNR(Peak Signal to Noise Ratio) between Given and EnhancedImages. This file takes the input images from `OtherImages` Folder. Here We calculate the correlation between PSNR ratio and Hardcoded Subjective ratings collected from 5 people. 
 - 

 
 
 ## References
 
 - [Underwater Image Enhancement Based On Contrast Adjustment via Differential Evolution Algorithm](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7571849&tag=1)
 - [A Dataset to Evaluate Underwater Image
Restoration Methods](http://amandaduarte.com.br/turbid/Turbid_Dataset.pdf)   
 
    
 
