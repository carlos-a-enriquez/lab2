#! /usr/bin/env python

import sys, os
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sn
#from sklearn.preprocessing import OneHotEncoder

#from sklearn.metrics import precision_recall_curve
#from sklearn.metrics import roc_curve, auc
#from sklearn.metrics import confusion_matrix
#from confusion_matrix.cf_matrix import make_confusion_matrix

import environmental_variables as env
#import vH_train as tra
import vH_predict as pre
import cross_validation as cr




def benchmark_scores(train, bench, alphabet, aa_ratios_alphabet):
	"""
	This function will determine a Position Specific Weight Matrix for the SP
	subsequence of the ENTIRE training set. Then, it will use it to assign scores
	to every sequence in the benchmark dataset. 
	
	train = the training dataset dataframe
	
	bench = the benchmark dataset dataframe
	
	alphabet= Order of amino acid residues to be used for matrix generation
	
	aa_ratios_alphabet = List of background amino acid ratio composition (Swissprot). 
	Should be in the same order as "alphabet".
	
	Requirements: The function PSWM_gen_folds() from cross_validate. 
	"""
	#PSWM profile generation for training dataset
	pswm = cr.PSWM_gen_folds(train, alphabet, aa_ratios_alphabet) #PSWM for the entire training dataset
	
	#Predict on benchmark sequences
	bench_predict = list(bench['Sequence (first 50 N-terminal residues)']) #Positive and negative examples included. Also, the full 50 aa sequence is used. 
	bench_predictions = pre.predict_seq(bench_predict, pswm, alphabet)
	
	#Generating benchmark score dataframe
	bench_predictions = pd.Series(bench_predictions[:])
	bench.loc[:,'scores'] = bench_predictions
	bench.to_csv('benchmark_set_scores.csv')
	
	
	
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
	image_folder_path = image_folder_path + 'benchmark/'
	if not os.path.exists(image_folder_path[:-1]):
		os.system('mkdir -p -v '+image_folder_path[:-1])	
	
	#Extracting the benchmark dataframe
	bench_scores = pd.read_csv('benchmark_set_scores.csv')
	
	#Finding the best threshold
	best_thresholds = np.array(best_thresholds[:])
	optimal_threshold = np.average(best_thresholds)	
	
	#Doing the skewed class analysis based on the threshold
	cm = cr.confusion_matrix_generator(bench_scores, optimal_threshold)
	cr.graphics_confusion_matrix(cm, optimal_threshold, image_folder_path, 'bench')
	cr.graphics_density_distribution(bench_scores, optimal_threshold, image_folder_path, 'bench')
	
	
	








if __name__ == "__main__":
	#Opening the input examples file and defining the output image folder path
	try:
		train_fh = sys.argv[1]
		image_folder_path = sys.argv[2]
		bench_fh = sys.argv[3]
		cross_validate= sys.argv[4] #Write yes to specifiy that the cross_validation_init code must be executed again
		
	except IndexError:
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
		print("Repeating cross-validation data frame generation")
		cr.cross_validation_init(train, env.alphabet, env.aa_ratios_alphabet)
	else:
		print("The cross-validation data frame (with scores) generation procedure was skipped")
		
		
	#Finding the best thresholds from the cross-validation score results
	n_folds = len(train.loc[:,'Cross-validation fold'].unique().tolist())
	best_thresholds = cr.threshold_optimization(n_folds, image_folder_path)
	
	#Threshold evaluation on benchmark
	benchmark_scores(train, bench, env.alphabet, env.aa_ratios_alphabet)
	benchmark_eval(best_thresholds, image_folder_path)
	
