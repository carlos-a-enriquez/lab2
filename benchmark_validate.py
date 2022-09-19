#! /usr/bin/env python

import pandas as pd
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
#from sklearn.preprocessing import OneHotEncoder
import environmental_variables as env
import vH_train as tra
import vH_predict as pre
import cross_validation as cr

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from confusion_matrix.cf_matrix import make_confusion_matrix











if __name__ == "__main__":
	#Opening the input examples file and defining the output image folder path
	try:
		train_fh = sys.argv[1]
		image_folder_path = sys.argv[2]
		bench_fh = sys.argv[3]
		cross_validate= sys.argv[4] #Write yes to specifiy that the cross_validation_init code must be executed again
		
	except:
		train_fh = input("insert the training data path   ")
		image_folder_path = input("insert the output image folder path  ")
		bench_fh = input("insert the benchmark data paths  ")
		cross_validate= input("Should the cross-validation procedure be repeated for the training data? (Y/N)  ")
	
	if image_folder_path[-1] != "/":
		image_folder_path += "/"		
		
	#Loading the dataframes
	train = pd.read_csv(train_fh, sep='\t')
	bench = pd.read_csv(bench_fh, sep='\t')
	
	#Generating the scores for cross_validation
	if cross_validate.lower()[0] == "y":
		print("Repeating cross-validation procedure")
		cr.cross_validation_init(train, env.alphabet, env.aa_ratios_alphabet)
		
		
	#Finding the best threshold from the cross-validation score results
	n_folds = len(train.loc[:,'Cross-validation fold'].unique().tolist())
	best_thresholds = cr.threshold_optimization(n_folds, image_folder_path)
	
