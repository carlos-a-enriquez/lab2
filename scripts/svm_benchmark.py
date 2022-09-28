#! /usr/bin/env python
"""
Note: In comparison to the vH benchmark module, this module cannot
execute the training as a subroutine. Instead, it expects to receive the 
SVM model as input (provide the filehandler). 
"""
import sys
import time
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix, matthews_corrcoef, precision_score, \
recall_score, f1_score, accuracy_score
import environmental_variables as env
import svm_encode as enco
import svm_train as svt
from cross_validation import graphics_confusion_matrix

start_time = time.time()





if __name__ == "__main__":
	try:
		svm_model_fh = sys.argv[0]
		image_folder_path = sys.argv[2]
		bench_fh = sys.argv[3]
		cross_validate= sys.argv[4] #Write yes to specifiy that the cross_validation_init code must be executed again
		
	except IndexError:
		svm_model_fh = input("Insert the svm model path   ")
		image_folder_path = input("insert the output image folder path  ")
		bench_fh = input("insert the benchmark data paths  ")
		cross_validate= input("Should the cross-validation procedure be repeated for the training data? (Y/N)  ")
		
	if image_folder_path[-1] != "/":
		image_folder_path += "/"		
		
	
	
	
	
	
	print("--- %0.2f seconds ---" % (time.time() - start_time))
