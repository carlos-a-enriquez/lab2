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
	
	training_fh = Assumes a filehandler that points to a .tsv
	 file with the 'Class' column. 
	 
	 Returns an array of values. 
	"""
	train = pd.read_csv(training_fh, sep='\t')
	return np.array(list(train['Class']))
	
	
def extract_fold_info(training_fh):
	"""
	This function extracts the cross-validation fold labeling 
	of the training examples. 
	
	training_fh = Assumes a filehandler that points to a .tsv
	 file with the 'Class' column. 
	 
	Returns two objects: an array with all fold labels and an 
	array with all unique fold labels (total number of cross 
	validation folds)
	 """
	train = pd.read_csv(training_fh, sep='\t')
	folds, unique_folds = (train.loc[:,'Cross-validation fold'], train.loc[:,'Cross-validation fold'].unique().tolist())
	return folds, unique_folds
	
	
def cross_validation_init_grid(sequences,Y, folds, unique_folds, hyper_param_dict):
	"""
	This function is meant to be called by grid_search_validate() so that it can
	perform the cross-validation within the grid search routine. 
	
	sequences = An list that contains the list of sequences to be
	encoded. It is assumed to be the first N-terminal residues of the sequence.  
	
	Y = Numpy array containing the true classes corresponding to each training 
	example.
	
	folds = List of length #ofexamples containing the fold labels for all 
	training examples
	
	unique_folds = List containing all different fold labels (the length equals
	the number of cross validation iterations).
	
	hyper_param_dict = Dictionary of unique hyperparameter conditions
	"""
	#Converting folds into an array
	if isinstance(folds, list):
		folds = np.array(folds[:])
		
	#Obtaining X, the 2D array of 20-dim vectors per example
	X = enco.encode_sequences(sequences, hyper_param_dict['K'], env.alphabet)
	
	#Cross validation iteration
	for f in unique_folds:
		#Separating training and testing
		test_indeces = np.where(folds == f)[0] #extract np array indeces where folds == current f
		train_indeces = np.where(folds != f)[0]
		test_iter_X = X[test_indeces, :]
		test_iter_Y = Y[test_indeces]
		train_iter_X = X[train_indeces, :]
		train_iter_Y = Y[train_indeces]
		
		#Define the model
		mySVC = svm.SVC(C=hyper_param_dict['C'], kernel=‘rbf’, gamma=hyper_param_dict['Gamma'])
		
		#Train the model
		mySVC.fit(train_iter_X, train_iter_Y)
		
	
	
	
	
def grid_search_validate(sequences, Y, k_list, c_list, gamma_list, folds, unique_folds):
	"""
	This function is meant to implement the grid search for the
	tuning of different SVM hyperparameters. The input should be the following:
	
	k_list = An array type object that contains the list of k values to be evaluated
	
	c_list =  An array type object that contains the list of C (soft margin) values 
	to be evaluated
	
	gamma_list = An array type object that contains the list of gamma values to 
	be evaluated (for the RBF kernel)
	
	sequences = An list that contains the list of sequences to be
	encoded. It is assumed to be the first N-terminal residues of the sequence.  
	
	Y = Numpy array containing the true classes corresponding to each training 
	example.
	
	folds = List of length #ofexamples containing the fold labels for all 
	training examples
	
	unique_folds = List containing all different fold labels (the length equals
	the number of cross validation iterations).
	"""
	#Creating the combination of hyperparameters
	hyper_param = [(k,c,g) for k in k_list for c in c_list for g in gamma_list]
	
	#iterating over all hyperparameter combinations
	for comb in hyper_param:
		hyper_param_dict = dict("K"=comb[0], "C" = comb[1], "Gamma" = comb[2])
		cross_validation_init_grid(sequences,Y, folds, unique_folds, hyper_param_dict)
		
		
		
		
		
	



if __name__ == "__main__":
	#Opening the input examples file and defining the output image folder path
	try:
		train_fh = sys.argv[1]		
	except IndexError:
		train_fh = input("insert the training data path   ")
		
	#Encoding workflow
	sequences = enco.extract_sequences(train_fh)
	train_Y = extract_true_classes(train_fh)	
	folds, unique_folds = extract_fold_info(train_fh)
	grid_search_validate(sequences, train_Y, env.k_list, env.c_list, env.gamma_list, folds, unique_folds)
	
	
	
	
	
