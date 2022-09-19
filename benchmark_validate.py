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


def benchmark_scores(train, bench, alphabet, aa_ratios_alphabet):
	"""
	This function will determine a Position Specific Weight Matrix for the SP
	subsequence of the ENTIRE training set. Then, it will use it to assign scores
	to every sequence in the benchmark dataset. 
	
	train = the training dataset dataframe
	
	bench = the benchmark dataset dataframe
	
	alphabet= Order of amino acid residues to be used for matrix generation
	
	aa_ratios_alphabet = List of background amino acid ratio composition (Swissprot). Should be in the same order as "alphabet".
	
	Requirements: The function PSWM_gen_folds() from cross_validate. 
	"""
	#PSWM profile generation for training dataset
	pswm = cr.PSWM_gen_folds(train, alphabet, aa_ratios_alphabet) #PSWM for the entire training dataset
	
	#Predict on benchmark sequences
	bench_predict = [seq for seq in bench['Sequence (first 50 N-terminal residues)']] #Positive and negative examples included. Also, the full 50 aa sequence is used. 
	bench_predictions = pre.predict_seq(bench_predict, pswm, alphabet)
	
	#Generating benchmark score dataframe
	bench_predictions = pd.Series(bench_predictions[:])
	bench.loc[:,'scores'] = bench_predictions
	bench.to_csv('benchmark_set_scores.csv')
	
	
	
def benchmark_eval(best_threshold):
	"""
	This function will load the file 'benchmark_set_scores.csv' and it will use the column
	'Class' to obtain y_true and the column 'scores' to obtain y_pred. Then, a classification
	accuracy procedure will be carried out.
	
	best_threshold = Final training threshold obtained from the cross validation procedure. 
	
	Requirements:
	- skewed_class_eval() function from the cross_validation module. 
	"""
	#Extracting the benchmark dataframe
	bench = pd.read_csv('benchmark_set_scores.csv', sep='\t')
	
	# classify examples in the benchmark set using the predicted score and trained threshold
	y_pred = [int(scr >= best_threshold) for scr in bench.loc[:,'scores'].to_list()]
	
	# binary representation of the true (observed) class for each testing example: 0=NO_SP, 1=SP
	y_true = [int(val == 'SP') for val in bench.loc[:,'Class'].tolist()
	
	
	








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
	else:
		print("The cross-validation procedure was skipped")
		
		
	#Finding the best threshold from the cross-validation score results
	n_folds = len(train.loc[:,'Cross-validation fold'].unique().tolist())
	best_thresholds = cr.threshold_optimization(n_folds, image_folder_path)
	
	#Threshold evaluation on benchmark
	benchmark_scores(train, bench, env.alphabet, env.aa_ratios_alphabet)
	
