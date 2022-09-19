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
import cross_validation as cross

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
		
	except:
		train_fh = input("insert the training data path   ")
		image_folder_path = input("insert the output image folder path  ")
		bench_fh = input("insert the benchmark data paths  ")
	
	if image_folder_path[-1] != "/":
		image_folder_path += "/"		
		
	#Loading the dataframes
	train = pd.read_csv(train_fh, sep='\t')
	bench = pd.read_csv(bench_fh, sep='\t')
		
	
