#! /usr/bin/env python
"""
Note: In comparison to the vH benchmark module, this module cannot
execute the training as a subroutine. Instead, it expects to receive the 
SVM model as input (provide the filehandler).

The SVM model file can be created using the svm_train module.  
"""
import sys
import time
import pandas as pd
import numpy as np
from sklearn import svm
import pickle, gzip
from sklearn.metrics import confusion_matrix, matthews_corrcoef, precision_score, \
recall_score, f1_score, accuracy_score
import environmental_variables as env
import svm_encode as enco
import svm_train as svt
from cross_validation import graphics_confusion_matrix

start_time = time.time()

def import_svm(svm_model_fh):
	"""
	This function will be used to obtain the svm model that
	is needed to obtain predictions on the benchmark. 
	
	svm_model_fh = specifies the pickle model dump
	"""
	with gzip.open(svm_model_fh, 'rb') as f:
		data = pickle.load(f)	
	return data
	
	
def bench_evaluation(svm_model, sequences, true_Y, hyper_param_dict):
	"""
	This function is meant to take care of the benchmark evaluation
	routine. A confusion matrix should be returned, as well as 
	evaluation statistics. 
	
	svm_model = an svm model file that should have been generated
	as a data dump (pickle package) from the svm_train module.
	
	sequences = An list that contains the list of sequences to be
	encoded. It is assumed to be the first N-terminal residues of the sequence.  
	
	true_Y = Numpy array containing the true classes corresponding to each 
	BENCHMARK example.
	
	hyper_param_dict = Dictionary of unique hyperparameter conditions
	"""
	
	#Obtaining X, the 2D array of 20-dim vectors per example
	X = enco.encode_sequences(sequences, hyper_param_dict['K'], env.alphabet)
	
	#Predict on test set
	y_pred_bench = mySVC.predict(X)
	
	#Validate
	cm = confusion_matrix(true_Y, y_pred_bench)	
	graphics_confusion_matrix(cm, "N/A", image_folder_path, "svm_bench")
	
	#Validate statistics
	MCC = matthews_corrcoef(true_Y, y_pred_bench)
	acc = accuracy_score(true_Y, y_pred_bench)
	prec = precision_score(true_Y, y_pred_bench, zero_division='warn')
	rec = recall_score(true_Y, y_pred_bench, zero_division='warn')
	f1 = f1_score(true_Y, y_pred_bench)
	metric_list = [MCC, acc, prec, rec, f1]
	
	#Metrics
	names = ['MCC', 'Accuracy', 'Precision', 'Recall', 'F1']
	metric_dict = {name:data for name,data in zip(names, metric_list)} #Obtain average, standard error pairs for each metric
	
	return metric_dict
	
	



if __name__ == "__main__":
	try:
		svm_model_fh = sys.argv[0]
		image_folder_path = sys.argv[2]
		bench_fh = sys.argv[3]
		best_comb= sys.argv[4:7] #Tuple of the best hyperparameters
		
	except IndexError:
		svm_model_fh = input("Insert the svm model path   ")
		image_folder_path = input("insert the output image folder path  ")
		bench_fh = input("insert the benchmark data paths  ")
		best_comb= tuple(input("Insert the hyperparameter combination in the order \"K C Gamma\" separated by spaces").split())
		
	if image_folder_path[-1] != "/":
		image_folder_path += "/"		
		
	#Hyperparameter dictionary
	hyper_param_dict = dict(K=best_comb[0], C = best_comb[1], Gamma = best_comb[2])
			
	#Obtaining prediction input
	mySVC= import_svm(svm_model_fh)
	sequences = enco.extract_sequences(bench_fh)
	bench_Y = svt.extract_true_classes(bench_fh)
	
	#Evaluating and printing results
	results = bench_evaluation(mySVC, sequences, bench_Y, hyper_param_dict)
	
	print("MCC: %0.2f +/- %0.2f"%results['MCC'])
	print("Accuracy: %0.2f +/- %0.2f"%results['Accuracy'])
	print("Precision: %0.2f +/- %0.2f"%results['Precision'])
	print("Recall: %0.2f +/- %0.2f"%results['Recall'])
	print("F1: %0.2f +/- %0.2f"%results['F1'])
	print("--- %0.2f seconds ---" % (time.time() - start_time))
	
	
	
	
	
	
	
	print("--- %0.2f seconds ---" % (time.time() - start_time))
