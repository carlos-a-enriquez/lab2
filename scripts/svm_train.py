#! /usr/bin/env python

import sys
import pandas as pd
import numpy as np
from sklearn import svm
import pickle, gzip
from sklearn.metrics import confusion_matrix, matthews_corrcoef, precision_score, \
recall_score, f1_score, accuracy_score
#import matplotlib.pyplot as plt
#import seaborn as sn
import environmental_variables as env
import svm_encode as enco
from cross_validation import graphics_confusion_matrix
import time

start_time = time.time()


def extract_true_classes(training_fh):
	"""
	This function extracts the true sequence classifications
	 from the dataframe.
	
	training_fh = Assumes a filehandler that points to a .tsv
	 file with the 'Class' column. 
	 
	 Returns an array of the coded classes. 
	"""
	train = pd.read_csv(training_fh, sep='\t')
	class_list = [int(val == 'SP') for val in list(train['Class'])]
	return np.array(class_list)
	
	
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
	perform the cross-validation within the grid search routine. It will return 
	a dictionary indicating the average metrics and standard error of the 
	cross_validation routine.
	
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
	
	#List of evaluation values
	MCC_list = list() #Will be used to choose the best hyperparameters
	acc_list, prec_list, rec_list , f1_list= [], [], [], []	
	
	
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
		
		#Other statistics
		acc_list.append(accuracy_score(test_iter_Y, y_pred_test))
		prec_list.append(precision_score(test_iter_Y, y_pred_test, zero_division=0))
		rec_list.append(recall_score(test_iter_Y, y_pred_test, zero_division=0))
		f1_list.append(f1_score(test_iter_Y, y_pred_test))
		
	
	#List to numpy
	MCC_list = np.array(MCC_list[:])
	acc_list, prec_list, rec_list , f1_list = np.array(acc_list[:]), np.array(prec_list[:]), np.array(rec_list[:]), np.array(f1_list[:])
	metric_lists = MCC_list, acc_list, prec_list, rec_list , f1_list
	
	#Obtaining the averages and the standard error
	names = ['MCC', 'Accuracy', 'Precision', 'Recall', 'F1']
	metrics = {name:(np.mean(data), (np.std(data, ddof=1) / np.sqrt(np.size(data)))) for name,data in zip(names, metric_lists)} #Obtain average, standard error pairs for each metric
	
	return metrics
		
		
	
	
	
	
def grid_search_validate(sequences, Y, k_list, c_list, gamma_list, folds, unique_folds):
	"""
	This function is meant to implement the grid search for the
	tuning of different SVM hyperparameters. The input should be the following:
	
	k_list = An array type object that contains the list of k values to be evaluated
	
	c_list =  An array type object that contains the list of C (soft margin) values 
	to be evaluated
	
	gamma_list = An array type object that contains the list of gamma values to 
	be evaluated (for the RBF kernel)
	
	sequences = A list that contains the list of sequences to be
	encoded. It is assumed to be the first N-terminal residues of the sequence.  
	
	Y = Numpy array containing the true classes (binary) corresponding to each training 
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
	
	#Finding the best MCC and the best hyperparameters
	mcc_results = [mcc_dic['MCC'][0] for mcc_dic in hyper_results]
	best_MCC = max(mcc_results)
	best_mcc_index = mcc_results.index(best_MCC)
	
	#Formatting results for printing
	mcc_results_np = np.around(np.array(mcc_results[:]), 2)
	
	return hyper_param, mcc_results_np, hyper_param[best_mcc_index], hyper_results[best_mcc_index]
		
		
		
def final_train_SVM(sequences, Y, comb):
	"""
	This function will generate the final training model based on the combination
	of hyperparameters that were found to be the best during cross-validation. 
	
	sequences = A list that contains the sequences to be encoded. 
	It is assumed to be the first N-terminal residues of the sequence.  
	It should contain the entire training set. 
	
	Y = Numpy array containing the true classes (binary) corresponding to each training 
	example.
	
	best_combination = A 3 component tuple containing the values of the best 
	hyperparameters in the order (k,c,g). It should correspond to hyper_param[best_mcc_index]
	from the grid_search_validate() function. 
	
	"""
	#Model folder
	if not os.path.exists('../svm_models'):
		os.system('mkdir -p -v '+'../svm_models')
	
	#Hyperparameter dictionary
	hyper_param_dict = dict(K=comb[0], C = comb[1], Gamma = comb[2])
	
	#Obtaining X, the 2D array of 20-dim vectors per example
	X = enco.encode_sequences(sequences, hyper_param_dict['K'], env.alphabet)
	
	#Define and train the model
	mySVC = svm.SVC(C=hyper_param_dict['C'], kernel='rbf', gamma=hyper_param_dict['Gamma'])
	mySVC.fit(X, Y)
	
	# Save the model to file 'myModel.pkl' using pickle
	pickle.dump(mySVC, gzip.open('../svm_models/myModel.pkl.gz', 'w'))
		
	



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
	comb, mccs, best_comb, best_metrics = grid_search_validate(sequences, train_Y, env.k_list, env.c_list, env.gamma_list, folds, unique_folds)
	
	#Creating and saving training model
	final_train_SVM(sequences, train_Y, best_comb)
	
	#Printing output
	print("\nModel Tuning results:\nCombinations: %s\nMCC scores: %s\nBest Combination: %s"%(str(comb), str(mccs), str(best_comb)))
	print("Best MCC: %0.2f +/- %0.2f"%best_metrics['MCC'])
	print("Best Accuracy: %0.2f +/- %0.2f"%best_metrics['Accuracy'])
	print("Best Precision: %0.2f +/- %0.2f"%best_metrics['Precision'])
	print("Best Recall: %0.2f +/- %0.2f"%best_metrics['Recall'])
	print("Best F1: %0.2f +/- %0.2f"%best_metrics['F1'])
	print("--- %0.2f seconds ---" % (time.time() - start_time))
	
	
	
	
	
