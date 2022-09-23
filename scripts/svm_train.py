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
	
	
def cross_validation_init_grid(x,y)
	
	
def grid_search_validate(x, y, k_list, c_list, gamma_list):
	"""
	This function is meant to implement the grid search for the
	tuning of different SVM hyperparameters. The input should be the following:
	
	k_list = An array type object that contains the list of k values to be evaluated
	
	c_list =  An array type object that contains the list of C (soft margin) values 
	to be evaluated
	
	gamma_list = An array type object that contains the list of gamma values to 
	be evaluated (for the RBF kernel)
	
	X= Numpy array containing the 20-dimensional vectors corresponding to each
	training example. 
	
	Y = Numpy array containing the true classes corresponding to each training 
	example.
	"""
	#Creating the combination of hyperparameters
	hyper_param = [(k,c,g) for k in k_list for c in c_list for g in gamma_list]
	
	#iterating over all hyperparameter combinations
	for comb in hyper_param:
		
		
	



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
	
	
	
