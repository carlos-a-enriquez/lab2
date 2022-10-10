#! /usr/bin/env python

import sys, os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
#from sklearn.preprocessing import OneHotEncoder

#from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix, matthews_corrcoef, precision_score, \
recall_score, f1_score, accuracy_score, auc, precision_recall_curve
from confusion_matrix.cf_matrix import make_confusion_matrix

import environmental_variables as env
import vH_train as tra
import vH_predict as pre

start_time = time.time()

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
	 



def cross_validation_init(train, alphabet, aa_ratios_alphabet, output_folder):
	''''This function takes a dataframe of examples as input. The table should contain 
	a "Cross-validation fold" column that specifies each of the k-folds to which each 
	example belongs.
	The function will return one updated dataframe table with corresponding vH scores for each of the k
	training iterations. In the case of the testing examples, they will have the value 'test' assigned
	so they can be filtered out and used later.
	
	output_folder= path where the training csv files will be saved
	'''
	#Folder creation	
	if not os.path.exists(output_folder[:-1]):
		os.system('mkdir -p -v '+output_folder[:-1])
		
	
	for fold in train.loc[:,'Cross-validation fold'].unique().tolist(): #Iterate over every subset in order to assign it the rule of "testing subset"
		#Separating training and testing
		train_iter = train.loc[train.loc[:,'Cross-validation fold'] != fold, :] #Exclude the testing examples
		test_iter = train.loc[train.loc[:,'Cross-validation fold'] == fold, :]  #dataframe of testing examples
		train_iter.reset_index(drop=True, inplace=True) #Reset indices
		test_iter.reset_index(drop=True, inplace=True) #Reset indices
		
		#Profile generation (vH_train)
		pswm = PSWM_gen_folds(train_iter, alphabet, aa_ratios_alphabet)
		
		#Prediction on training
		train_predict = list(train_iter['Sequence (first 50 N-terminal residues)']) #Positive and negative examples included. Also, the full 50 aa sequence is used. 
		train_predictions = pre.predict_seq(train_predict, pswm, alphabet)
		
		#Prediction on testing
		test_predict = list(test_iter['Sequence (first 50 N-terminal residues)']) #Positive and negative examples included. Also, the full 50 aa sequence is used. 
		test_predictions = pre.predict_seq(test_predict, pswm, alphabet)
		
		
		#Creating train dataframe
		train_predictions = pd.Series(train_predictions[:])
		train_iter.loc[:,'scores'] = train_predictions
		train_iter.to_csv(output_folder+'iteration_'+str(fold)+'_vh_training.csv')
		
		#Creating test dataframe
		test_predictions = pd.Series(test_predictions[:])
		test_iter.loc[:,'scores'] = test_predictions
		test_iter.to_csv(output_folder+'iteration_'+str(fold)+'_vh_testing.csv')
		


def graphics_confusion_matrix(cm, optimal_threshold, image_folder_path, cm_suffix):
	"""
	The purpose of this function is to create confusion matrix images. 
	This function is meant to receive a numpy array of shape (2,2) that 
	represents the confusion matrix of a binary classification problem. 
	
	The cm_suffix will be added to the predefined confusion matrix image name:
	'test_confusion_matrix_%s.png'%(cm_suffix)
	
	The dist_suffix will be added to the predefined score distribution plot image name:
	'test_score_dist_%s.png'%(dist_suffix)
	
	This function has no output, but it produces images that are added to the image_folder_path.
	
	"""
	#Image folder
	if not os.path.exists(image_folder_path[:-1]):
		os.system('mkdir -p -v '+image_folder_path[:-1])	
	
	#Changing the threshold into a string
	if isinstance(optimal_threshold, float):
		optimal_threshold = str(round(optimal_threshold, 2))
		
	#Confusion matrix generation
	labels = ['True Neg','False Pos','False Neg','True Pos']
	categories = ['non-SP', 'SP']
	make_confusion_matrix(cm, group_names=labels, categories=categories, cmap='binary', title='Test set "%s" at trained threshold %s'%(cm_suffix, optimal_threshold), sum_stats=True)
	plt.savefig(image_folder_path+'test_confusion_matrix_%s.png'%(cm_suffix), bbox_inches='tight')
	plt.close()
	

		
def graphics_density_distribution(scores, classes, optimal_threshold, image_folder_path, dist_suffix):
	"""
	The purpose of this function is to create density distribution images. 
	The images represent the distribution of scores according to true classification
	and include a threshold value as a vertical line.
		
	The dist_suffix will be added to the predefined score distribution plot image name:
	'test_score_dist_%s.png'%(dist_suffix)
	
	This function has no output, but it produces images that are added to the image_folder_path.
	
	scores = this object would be a numpy array containing the vH model's output raw score
	 on the benchmark examples.
	 
	classes = Pandas series of the binary representation of the true (observed) class for each training 
	example: 0=NO_SP, 1=SP
	
	"""
	#Image folder
	if not os.path.exists(image_folder_path[:-1]):
		os.system('mkdir -p -v '+image_folder_path[:-1])	
	
	#Score distribution plot
	plt.figure() #ensures a clean canvas before plotting
	sn.kdeplot(scores, shade=True, hue=classes).set(xlabel='Test set "%s" score distribution'%(dist_suffix))
	children = plt.gca().get_children() #Extracting the plot handles in order to pass them to plt.legend
	l = plt.axvline(optimal_threshold, 0, 1, c='r')
	plt.legend([children[1], children[0], l], classes.unique().tolist()+['Threshold = %0.2f'%(optimal_threshold)])
	plt.savefig(image_folder_path+'test_score_dist_%s.png'%(dist_suffix), bbox_inches='tight')
	plt.close()		
	

def confusion_matrix_generator(df, optimal_threshold):
	"""
	This function produces the confusion matrix based on a dataframe of examples (with scores and true Classes) 
	and a predefined threshold.
	
	The df is a dataframe that must follow a specific format that is specified in the project documentation.
	In particular, the "Class" and "Scores" columns are needed.
	
	The optimal_threshold is a predefined threshold that will be used to classifiy examples according to their score. 
	The threshold could be predefined or selected according to a precious training procedure 
	(see the threshold optimization function).
	"""
	# classify examples in the testing set using predicted score and trained threshold
	y_pred_test = [int(scr >= optimal_threshold) for scr in df.loc[:,'scores'].to_list()]

	# binary representation of the true (observed) class for each testing example: 0=NO_SP, 1=SP
	y_true_test = [int(val == 'SP') for val in df.loc[:,'Class'].tolist()]

	#Confusion matrix generation
	return confusion_matrix(y_true_test, y_pred_test)
	

def average_confusion_matrix(cm_list):
	"""
	Generates the average confusion matrix based on a list of confusion matrices.
	
	cm_list: A list (2,2) numpy arrays expected.
	"""
	total = len(cm_list)
	if total < 1:
		return 'error'
	added_cm = np.zeros((2,2))
	for cm in cm_list:
		added_cm += cm
	return added_cm/total
		


def threshold_optimization(n_folds, image_folder_path):
	'''
	The number of nfolds must be specified. It is assumed that the function will have the csv files
	will have the following format:
	'iteration_'+str(fold)+'_vh_training.csv'
	'iteration_'+str(fold)+'_vh_testing.csv'
	The input dataframes must also follow a specific format that is specified in the project documentation.
	The output of this function is a list of the optimized thresholds obtained for each cross-validation iteration.  
	
	Dependencies: Depends on the confusion_matrix_generator(), graphics_confusion_matrix() and
	graphics_density_distribution() functions.
	'''
	threshold_list = [] #Threshold list to be used for downstream benchmark analysis
	cm_list = [] #Confusion matrix list to be used for downstream analysis
	
	#Image folder
	if not os.path.exists(image_folder_path[:-1]):
		os.system('mkdir -p -v '+image_folder_path[:-1])
		
	#Instantiating the statistic list
	mcc_list, acc_list, prec_list, rec_list , f1_list= [], [], [], [], []	
	
	#Obtaining the thresholds for each cross-validation iteration
	for fold in range(n_folds):
		#Loading training data
		df_train = pd.read_csv(image_folder_path+'iteration_%d_vh_training.csv'%(fold))
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

		#Extracting testing dataframe, known classes and predicted classes
		df_test = pd.read_csv(image_folder_path+'iteration_%d_vh_testing.csv'%(fold))
		y_true_test = [int(val == 'SP') for val in df_test.loc[:,'Class'].tolist()]
		y_pred_test = [int(scr >= optimal_threshold) for scr in df_test.loc[:,'scores'].to_list()]
		

		#Doing skewed class evaluation on this test set iteration (df_test)
		cm = confusion_matrix(y_true_test, y_pred_test)
		cm_list.append(cm) #Appending cm to list
		graphics_confusion_matrix(cm, optimal_threshold, image_folder_path, str(fold))
		graphics_density_distribution(df_test.loc[:, 'scores'], df_test.loc[:, 'Class'], optimal_threshold, image_folder_path, str(fold))
		
		#Obtaining additional skewed-class statistics
		mcc_list.append(matthews_corrcoef(y_true_test, y_pred_test)) 
		acc_list.append(accuracy_score(y_true_test, y_pred_test))
		prec_list.append(precision_score(y_true_test, y_pred_test, zero_division=0))
		rec_list.append(recall_score(y_true_test, y_pred_test, zero_division=0))
		f1_list.append(f1_score(y_true_test, y_pred_test))
		
		
		
	#Obtaining the average cm and printing it
	avg_cm = average_confusion_matrix(cm_list)
	graphics_confusion_matrix(avg_cm, 'NaN', image_folder_path, 'average')
	
	#Turning statistic lists to arrays
	mcc_list, acc_list, prec_list, rec_list , f1_list = np.array(mcc_list[:]), np.array(acc_list[:]), np.array(prec_list[:]), \
	np.array(rec_list[:]), np.array(f1_list[:])
	metric_lists = mcc_list, acc_list, prec_list, rec_list , f1_list
	
	#Obtaining the averages and the standard error
	names = ['MCC', 'Accuracy', 'Precision', 'Recall', 'F1']
	metrics = {name:(np.mean(data), (np.std(data, ddof=1) / np.sqrt(np.size(data)))) for name,data in zip(names, metric_lists)} #Obtain average, standard error pairs for each metric
	
	return threshold_list, metrics
	
	
	


		
	


if __name__ == "__main__":
	#Opening the input examples file and defining the output image folder path
	try:
		train_fh = sys.argv[1] 
		image_folder_train = sys.argv[2] 
		cross_validate= sys.argv[3] #Write yes to specifiy that the cross_validation_init code must be executed again
	except IndexError:
		train_fh = input("insert the training data path   ")
		image_folder_train = input("insert the output image folder path  ") 
		cross_validate= input("Should the cross-validation procedure be repeated for the training data? (Y/N)  ")
	
	#Prepping image_folder path
	if image_folder_train[-1] != "/":
		image_folder_train += "/"		
	image_folder_train += 'train/' #adding the train subdirectory
		
	
	#Loading the train file
	train = pd.read_csv(train_fh, sep='\t')
	
	#Generating the scores for cross_validation
	if cross_validate.lower()[0] == "y":
		print("Repeating cross-validation data frame generation")
		cross_validation_init(train, env.alphabet, env.aa_ratios_alphabet, image_folder_train)
	else:
		print("The cross-validation data frame (with scores) generation procedure was skipped")
	
	#Finding the best threshold from the cross-validation score results
	n_folds = len(train.loc[:,'Cross-validation fold'].unique().tolist())
	best_thresholds, stats = threshold_optimization(n_folds, image_folder_train)
	
	#Printing out metric results
	print("Training vH Cross-validation statistics")
	print("MCC: %0.2f +/- %0.2f"%stats['MCC'])
	print("Accuracy: %0.2f +/- %0.2f"%stats['Accuracy'])
	print("Precision: %0.2f +/- %0.2f"%stats['Precision'])
	print("Recall: %0.2f +/- %0.2f"%stats['Recall'])
	print("F1: %0.2f +/- %0.2f"%stats['F1'])
	print("--- %0.2f seconds ---" % (time.time() - start_time))
	
	
	
