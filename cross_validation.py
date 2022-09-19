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

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from confusion_matrix.cf_matrix import make_confusion_matrix



def PSWM_gen_folds(train, alphabet, aa_ratios_alphabet):
	'''
	This function is meant to automatize the procedure that can be carried out 
	by executing diverse functions from vh_train, hence streamlining the generation of an SP profile into a single step.
	A pswm matrix is generated from a dataframe of examples as input.
	Requirement: vH_train
	 '''
	#Profile generation (vH_train)
	train_sp = train.loc[train.loc[:,'Class']=='SP', :] #Eliminating negative examples to generate the profile
	train_seq_list = tra.cleavage_seq(train_sp)  #obtaining sequence list
	sequence_length = len(train_seq_list[0])
	one_hot_sequences= [tra.encode_seq(sequence, alphabet) for sequence in train_seq_list] #one hot encoding
	pspm = tra.PSPM_gen(one_hot_sequences, sequence_length) #Generating the PSPM matrix
	pswm = tra.PSWM_gen(pspm, aa_ratios_alphabet) #obtaining the PSWM matrix
	
	return pswm
	 



def cross_validation_init(train, alphabet, aa_ratios_alphabet):
	''''This function takes a dataframe of examples as input. The table should contain 
	a "Cross-validation fold" column that specifies each of the k-folds to which each 
	example belongs.
	The function will return one updated dataframe table with corresponding vH scores for each of the k
	training iterations. In the case of the testing examples, they will have the value 'test' assigned
	so they can be filtered out and used later.'''
	for fold in train.loc[:,'Cross-validation fold'].unique().tolist(): #Iterate over every subset in order to assign it the rule of "testing subset"
		#Separating training and testing
		train_iter = train.loc[train.loc[:,'Cross-validation fold'] != fold, :] #Exclude the testing examples
		test_iter = train.loc[train.loc[:,'Cross-validation fold'] == fold, :]  #dataframe of testing examples
		train_iter.reset_index(drop=True, inplace=True) #Reset indices
		test_iter.reset_index(drop=True, inplace=True) #Reset indices
		
		#Profile generation (vH_train)
		pswm = PSWM_gen_folds(train_iter, alphabet, aa_ratios_alphabet)
		
		#Prediction on training
		train_predict = [seq for seq in train_iter['Sequence (first 50 N-terminal residues)']] #Positive and negative examples included. Also, the full 50 aa sequence is used. 
		train_predictions = pre.predict_seq(train_predict, pswm, alphabet)
		
		#Prediction on testing
		test_predict = [seq for seq in test_iter['Sequence (first 50 N-terminal residues)']] #Positive and negative examples included. Also, the full 50 aa sequence is used. 
		test_predictions = pre.predict_seq(test_predict, pswm, alphabet)
		
		
		#Creating train dataframe
		train_predictions = pd.Series(train_predictions[:])
		train_iter.loc[:,'scores'] = train_predictions
		train_iter.to_csv('iteration_'+str(fold)+'_vh_training.csv')
		
		#Creating test dataframe
		test_predictions = pd.Series(test_predictions[:])
		test_iter.loc[:,'scores'] = test_predictions
		test_iter.to_csv('iteration_'+str(fold)+'_vh_testing.csv')
		


def skewed_class_eval(df, optimal_threshold, image_folder_path, cm_suffix, dist_suffix):
	"""
	The df is a dataframe that must follow a specific format that is specified in the project documentation.
	In particular, the "Class" and "Scores" columns are needed.
	
	The optimal_threshold is a predefined threshold that will be used to classifiy examples according to their score. 
	The threshold could be predefined or selected according to a precious training procedure 
	(see the threshold optimization function).
	
	The cm_suffix will be added to the predefined confusion matrix image name:
	'test_confusion_matrix_%s.png'%(cm_suffix)
	
	The cm_suffix will be added to the predefined score distribution plot image name:
	'test_score_dist_%s.png'%(dist_suffix)
	
	This function has no output, but it produces images that are added to the image_folder_path.
	
	"""
	#Image folder
	if not os.path.exists(image_folder_path[:-1]):
		os.system('mkdir -p -v '+image_folder_path[:-1])	
	
	#Extracting test set predicted scores
	#y_test_score = df.loc[:,'scores'].to_list()

	# classify examples in the testing set using predicted score and trained threshold
	y_pred_test = [int(scr >= optimal_threshold) for scr in df.loc[:,'scores'].to_list()]

	# binary representation of the true (observed) class for each testing example: 0=NO_SP, 1=SP
	y_true_test = [int(val == 'SP') for val in df.loc[:,'Class'].tolist()]

	#Confusion matrix generation
	cm = confusion_matrix(y_true_test, y_pred_test)
	labels = ['True Neg','False Pos','False Neg','True Pos']
	categories = ['non-SP', 'SP']
	make_confusion_matrix(cm, group_names=labels, categories=categories, cmap='binary', title='Test set %s at trained threshold %0.2f'%(cm_suffix, optimal_threshold), sum_stats=True)
	plt.savefig(image_folder_path+'test_confusion_matrix_%s.png'%(cm_suffix), bbox_inches='tight')

	#Score distribution plot
	plt.figure() #ensures a clean canvas before plotting
	sn.kdeplot(df.loc[:,'scores'], shade=True, hue=df.loc[:,'Class']).set(xlabel='Test set %s score distribution'%(dist_suffix))
	children = plt.gca().get_children() #Extracting the plot handles in order to pass them to plt.legend
	l = plt.axvline(optimal_threshold, 0, 1, c='r')
	plt.legend([children[1], children[0], l], df.loc[:,'Class'].unique().tolist()+['Threshold = %0.2f'%(optimal_threshold)])
	plt.savefig(image_folder_path+'test_score_dist_%s.png'%(dist_suffix), bbox_inches='tight')		
		


def threshold_optimization(n_folds, image_folder_path):
	'''
	The number of nfolds must be specified. It is assumed that the function will have the csv files
	will have the following format:
	'iteration_'+str(fold)+'_vh_training.csv'
	'iteration_'+str(fold)+'_vh_testing.csv'
	The input dataframes must also follow a specific format that is specified in the project documentation.
	The output of this function is a list of the optimized thresholds obtained for each cross-validation iteration.  
	
	Dependencies: Depends on the skewed_class_eval() function.
	'''
	threshold_list = [] #Threshold list to be used for downstream benchmark analysis
	
	#Image folder
	if not os.path.exists(image_folder_path[:-1]):
		os.system('mkdir -p -v '+image_folder_path[:-1])
	
	for fold in range(n_folds):
		#Loading training data
		df_train = pd.read_csv('iteration_%d_vh_training.csv'%(fold))
		y_score = df_train.loc[:,'scores'].to_list()
		
		#binary representation of the true (observed) class for each training example: 0=NO_SP, 1=SP
		y_true = [int(val == 'SP') for val in df_train.loc[:,'Class'].tolist()]
		
		#Precision recall curve
		precision, recall, th = precision_recall_curve(y_true, y_score)
		roc_auc = auc(recall, precision)

		plt.figure()
		lw = 2
		plt.plot(recall, precision, color="darkorange", lw=lw, label="Precision recall (area = %0.2f)" % roc_auc,)
		plt.plot([1, 0], [0, 1], color="navy", lw=lw, linestyle="--")
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel("Recall")
		plt.ylabel("Precision")
		plt.title("Precision-recall curve for set-%d"%(fold))
		plt.legend(loc="lower right")
		plt.savefig(image_folder_path+'precision_recall_curve_%d.png'%(fold), bbox_inches='tight')

		#Getting the best threshold
		fscore = (2 * precision * recall) / (precision + recall)    
		index = np.argmax(fscore)
		optimal_threshold = th[index]
		threshold_list.append(optimal_threshold)       

		#Extracting testing dataframe
		df_test = pd.read_csv('iteration_%d_vh_testing.csv'%(fold))

		#Doing skewed class evaluation on this test set iteration (df_test)
		skewed_class_eval(df_test, optimal_threshold, image_folder_path, str(fold), str(fold))
		

	return threshold_list
	
	
	


		
	


if __name__ == "__main__":
	#Opening the input examples file and defining the output image folder path
	try:
		train_fh = sys.argv[1]
		image_folder_path = sys.argv[2]
	except:
		train_fh = input("insert the training data path   ")
		image_folder_path = input("insert the output image folder path  ")
	
	if image_folder_path[-1] != "/":
		image_folder_path += "/"		
		
	
	#Loading the train file
	train = pd.read_csv(train_fh, sep='\t')
	
	#Generating the scores for cross_validation
	cross_validation_init(train, env.alphabet, env.aa_ratios_alphabet)
	#print(result) #debug
	
	#Finding the best threshold from the cross-validation score results
	n_folds = len(train.loc[:,'Cross-validation fold'].unique().tolist())
	best_thresholds = threshold_optimization(n_folds, image_folder_path)
	
	
