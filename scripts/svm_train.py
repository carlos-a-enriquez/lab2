#! /usr/bin/env python

import sys
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix, matthews_corrcoef
#import matplotlib.pyplot as plt
#import seaborn as sn
import environmental_variables as env
import svm_encode as enco
from cross_validation import graphics_confusion_matrix
#from accuracy import get_mcc


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
	folds, unique_folds = (train.loc[:,'Cross-validation fold'].tolist(), train.loc[:,'Cross-validation fold'].unique().tolist())
	return folds, unique_folds
	
	
def cross_validation_init_grid(sequences,Y, folds, unique_folds, hyper_param_dict, comb_id):
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
	
	comb = Combination of hyperparameter id
	"""
	#Converting folds into an array
	if isinstance(folds, list):
		folds = np.array(folds[:])
		
	#Obtaining X, the 2D array of 20-dim vectors per example
	X = enco.encode_sequences(sequences, hyper_param_dict['K'], env.alphabet)
	
	#List of MCC values
	MCC_list = list()
	
	#Cross validation iteration
	for f in unique_folds:
		#Separating training and testing
		test_indeces = np.where(folds == f)[0] #extract np array indeces where folds == current f
		train_indeces = np.where(folds != f)[0]
		test_iter_X = X[test_indeces, :]
		test_iter_Y = Y[test_indeces] #should be true classes
		train_iter_X = X[train_indeces, :]
		train_iter_Y = Y[train_indeces] #should be true classes
		
		#Define the model
		mySVC = svm.SVC(C=hyper_param_dict['C'], kernel='rbf', gamma=hyper_param_dict['Gamma'])
		
		#Train the model
		mySVC.fit(train_iter_X, train_iter_Y)
		
		#Predict on test set
		y_pred_test = mySVC.predict(test_iter_X)
		
		#Validate
		cm = confusion_matrix(test_iter_Y, y_pred_test)
		#input(cm)
		figure_id = str(comb_id)+'_fl_'+str(f)
		graphics_confusion_matrix(cm, "N/A", image_folder_path, "svm_%s"%(figure_id))
		MCC_list.append(matthews_corrcoef(test_iter_Y, y_pred_test)) #Mathew's correlation
	
	#List to numpy
	MCC_list = np.array(MCC_list[:])
	
	return np.mean(MCC_list)
		
		
	
	
	
	
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
	
	#Hyperparameter MCC list
	hyper_results = list()
	
	#iterating over all hyperparameter combinations
	comb_id = 0
	for comb in hyper_param:
		comb_id += 1
		hyper_param_dict = dict(K=comb[0], C = comb[1], Gamma = comb[2])
		hyper_results.append(cross_validation_init_grid(sequences,Y, folds, unique_folds, hyper_param_dict, comb_id))
		
	return hyper_param, hyper_results, max(hyper_results)
		
		
		
		
		
	



if __name__ == "__main__":
	#Opening the input examples file and defining the output image folder path
	try:
		train_fh = sys.argv[1]
		image_folder_path = sys.argv[2]		
	except IndexError:
		train_fh = input("insert the training data path   ")
		image_folder_path = input("insert the output image folder path  ")
		
	if image_folder_path[-1] != "/":
		image_folder_path += "/"		
		
	#Encoding workflow
	sequences = enco.extract_sequences(train_fh)
	train_Y = extract_true_classes(train_fh)	
	folds, unique_folds = extract_fold_info(train_fh)
	comb, mccs, best_mcc= grid_search_validate(sequences, train_Y, env.k_list, env.c_list, env.gamma_list, folds, unique_folds)
	print("\nModel Tuning results:\nCombinations: %s\nMCC scores: %s\nBest score: %0.2f"%(str(comb), str(mccs), best_mcc))
	
	
	
	
	
