#! /usr/bin/env python

import sys, os
import pandas as pd
import numpy as np
from sklearn import svm
#import matplotlib.pyplot as plt
#import seaborn as sn
import environmental_variables as env
import svm_encode as enco


def extract_true_classes(training_fh):
	"""
	This function extracts the true sequence classifications
	 from the dataframe.
	
	training_data = Assumes a filehandler that points to a .tsv
	 file with the 'Class' column. 
	 
	 Returns an array of values. 
	"""
	train = pd.read_csv(training_fh, sep='\t')
	return np.array(list(train['Class']))
	
	
def grid_search_validate():
	"""
	"""
	return None



if __name__ == "__main__":
	#Opening the input examples file and defining the output image folder path
	try:
		train_fh = sys.argv[1]		
	except IndexError:
		train_fh = input("insert the training data path   ")
		
	#Encoding workflow
	sequences = extract_sequences(train_fh)
	train_x = encode_sequences(sequences, 24, env.alphabet)
	train_y = extract_true_classes(train_fh)
	
	
	
