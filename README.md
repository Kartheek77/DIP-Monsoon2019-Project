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
 - In UWIEf_MODEL1.ipynb the algorithm specified in the paper has been Implemented. It takes input from the PaperImages Folder and displays the given and enhanced Images in the Notebook Itself.
 - The s1s2underWaterIEfinalVersion.py code is a different method where optimal values (Lower_Limit(r1), Upper_Limit(r2), s1, s2) for contrast stretching are determined by Differential Evolution Algo unlike in the first case where we fix s1 as 0 and s2 as 255. This takes input from PaperImages and outputs the enhanced images. We placed the resulting images in s1s2methodOutputPaperImages folder.
 -
 - Open UWIEf.ipynb for UWIE algorithm.
 - To create SMCSVM object use:   
   `clf = SMCSVM()`
 
 - Pass training data and trainng labels to fit() func'tion to train the classifier.   
   `clf.fit(train_X_data, train_y_label)`   
 
 - You can also pass folowing parameters:
    
    1. C, default_value, C=10 - Penalizing factor for Slack  
    2. kernel, default value, kernel='rbf', can also take - 'linear', 'polynomial'  for utilizing kernel tricks on Non-Linear data.   
    3. sigma, default_value=1.0, required for 'RBF' kernel.
    4. degree, default_value=1, degree of polynomial function used in 'polynomial' kernel.
 - To predict on testing data use:    
   `clf.predict(test_X_data)`
 - The algorithm uses K-fold cross validation as a performance metric.
 
 
 ## Run tests
 
 - To run tests, run the run_tests.ipynb file in tests directory
 
 ## References
 
 - [Underwater Image Enhancement Based On Contrast Adjustment via Differential Evolution Algorithm](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7571849&tag=1)
 - [A Dataset to Evaluate Underwater Image
Restoration Methods](http://amandaduarte.com.br/turbid/Turbid_Dataset.pdf)   
 
    
 
