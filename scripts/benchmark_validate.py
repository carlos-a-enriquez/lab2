#! /usr/bin/env python

import sys, os
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sn
#from sklearn.preprocessing import OneHotEncoder

#from sklearn.metrics import precision_recall_curve
#from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
#from confusion_matrix.cf_matrix import make_confusion_matrix

import environmental_variables as env
#import vH_train as tra
import vH_predict as pre
import cross_validation as cr
import svm_train as svt


start_time = time.time()


def extract_X_and_Y(bench_df):
	"""
	This function extractsboth the 50 residue sequence (for PSWM predictions)
	 and the true sequence classifications from the dataframe.
	
	bench_fh = Assumes a filehandler that points to a .tsv
	 file with the 'Class' column. 
	 
	 Returns an list of sequences (Positive and negative examples included. 
	 Also, the full 50 aa sequence is used.) and an array of the coded classes. 
	"""
	bench = pd.read_csv(bench_fh, sep='\t')
	X = list(bench['Sequence (first 50 N-terminal residues)']) 
	true_Y  = svt.extract_true_classes(bench_fh)
	return X,true_Y
	


def benchmark_scores(train_fh, bench_fh, alphabet, aa_ratios_alphabet, image_folder_path):
	"""
	This function will determine a Position Specific Weight Matrix for the SP
	subsequence of the ENTIRE training set. Then, it will use it to assign scores
	to every sequence in the benchmark dataset. 
	
	train_fh = the training dataset path
	
	bench_fh = the benchmark dataset path
	
	alphabet= Order of amino acid residues to be used for matrix generation
	
	aa_ratios_alphabet = List of background amino acid ratio composition (Swissprot). 
	Should be in the same order as "alphabet".
	
	image_folder_path = Location where the benchmark results will be saved
	
	Requirements: The function PSWM_gen_folds() from cross_validate. 
	"""
	#Folder creation	
	if not os.path.exists(image_folder_path[:-1]):
		os.system('mkdir -p -v '+image_folder_path[:-1])
		
	#Loading training and benchmark data
	train = pd.read_csv(train_fh, sep='\t')	
	bench_X, _ = extract_X_and_Y(bench_fh)
		
	#PSWM profile generation for training dataset
	pswm = cr.PSWM_gen_folds(train, alphabet, aa_ratios_alphabet) #PSWM for the entire training dataset
	
	#Predict on benchmark sequences
	bench_predictions = pre.predict_seq(bench_X, pswm, alphabet)
	
	#Generating benchmark score dataframe
	bench_predictions = pd.Series(bench_predictions[:])
	bench.loc[:,'scores'] = bench_predictions
	bench.to_csv(image_folder_path+'benchmark_set_scores_raw.csv')
	
	
	
def benchmark_eval(best_thresholds, image_folder_path):
	"""
	This function will load the file 'benchmark_set_scores.csv' and it will use the column
	'Class' to obtain y_true and the column 'scores' to obtain y_pred. Then, a classification
	accuracy procedure will be carried out.
	
	best_thresholds = Final list of training thresholds obtained from the cross validation procedure. 
	
	Requirements:
	- skewed_class_eval() function from the cross_validation module. 
	"""
	#Folder creation	
	if not os.path.exists(image_folder_path[:-1]):
		os.system('mkdir -p -v '+image_folder_path[:-1])	
	
	#Extracting the benchmark true scores and predictions
	_, true_Y = extract_X_and_Y(image_folder_path+'benchmark_set_scores_raw.csv')
	pred_Y = [int(scr >= optimal_threshold) for scr in bench.loc[:,'scores'].to_list()]
	
	#Finding the best threshold
	best_thresholds = np.array(best_thresholds[:])
	optimal_threshold = np.average(best_thresholds)	
	
	#Doing the skewed class analysis based on the threshold
	cm = confusion_matrix(true_Y, pred_Y)	
	cr.graphics_confusion_matrix(cm, optimal_threshold, image_folder_path, 'bench')
	cr.graphics_density_distribution(bench_scores, optimal_threshold, image_folder_path, 'bench')
	
	#Validate statistics
	MCC = matthews_corrcoef(true_Y, pred_Y)
	acc = accuracy_score(true_Y, pred_Y)
	prec = precision_score(true_Y, pred_Y, zero_division='warn')
	rec = recall_score(true_Y, pred_Y, zero_division='warn')
	f1 = f1_score(true_Y, pred_Y)
	metric_list = [MCC, acc, prec, rec, f1]
	
	#Metrics
	names = ['MCC', 'Accuracy', 'Precision', 'Recall', 'F1']
	metric_dict = {name:data for name,data in zip(names, metric_list)} #Obtain error metric
	
	#Printing results (replaces raw score with binary prediction)
	bench.loc[:,'scores'] = y_pred
	bench.to_csv(image_folder_path+'benchmark_set_scores.csv')
	
	return metric_dict
	
	
	
	
	








if __name__ == "__main__":
	#Opening the input examples file and defining the output image folder path
	try:
		train_fh = sys.argv[1]
		image_folder_p = sys.argv[2]
		bench_fh = sys.argv[3]
		cross_validate= sys.argv[4] #Write yes to specifiy that the cross_validation_init code must be executed again
		
	except IndexError:
		train_fh = input("insert the training data path   ")
		image_folder_p = input("insert the output image folder path  ")
		bench_fh = input("insert the benchmark data paths  ")
		cross_validate= input("Should the cross-validation procedure be repeated for the training data? (Y/N)  ")
	
	if image_folder_p[-1] != "/":
		image_folder_p += "/"
		
	image_folder_bench = image_folder_p + 'benchmark/'
	image_folder_train = image_folder_p + 'train/'
		
	#Generating the scores for cross_validation
	if cross_validate.lower()[0] == "y":
		print("Repeating cross-validation data frame generation")
		cr.cross_validation_init(train, env.alphabet, env.aa_ratios_alphabet, image_folder_train)
	else:
		print("The cross-validation data frame (with scores) generation procedure was skipped")
		
	
	### Workflow###
	
	#Finding the best thresholds from the cross-validation score results
	n_folds = len(train.loc[:,'Cross-validation fold'].unique().tolist())
	best_thresholds = cr.threshold_optimization(n_folds, image_folder_train)
	
	#Obtaining the prediction input
	bench_X, bench_Y = extract_X_and_Y(bench_df)
	
	
	#Threshold evaluation on benchmark
	benchmark_scores(train_fh, bench_fh, env.alphabet, env.aa_ratios_alphabet, image_folder_bench)
	results = benchmark_eval(best_thresholds, image_folder_bench)
	
	#Print results
	print("MCC: %0.2f"%results['MCC'])
	print("Accuracy: %0.2f"%results['Accuracy'])
	print("Precision: %0.2f"%results['Precision'])
	print("Recall: %0.2f"%results['Recall'])
	print("F1: %0.2f"%results['F1'])
	print("--- %0.2f seconds ---" % (time.time() - start_time))
	
