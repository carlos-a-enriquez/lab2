#! /usr/bin/env python

import pandas as pd
import sys
import numpy as np
#from sklearn.preprocessing import OneHotEncoder
import environmental_variables as env
import vH_train as tra
import vH_predict as pre



def cross_validation_init(train, alphabet, aa_ratios_alphabet):
	''''This function takes a dataframe of examples as input. The table should contain 
	a "Cross-validation fold" column that specifies each of the k-folds to which each 
	example belongs.
	The function will return one updated dataframe table with corresponding vH scores for each of the k
	training iterations. In the case of the testing examples, they will have the value 'test' assigned
	so they can be filtered out and used later.'''
	for fold in train.loc[:,'Cross-validation fold'].unique().tolist(): #Iterate over every subset in order to assign it the rule of "testing subset"
		#List of folds
		train_iter = train.loc[train.loc[:,'Cross-validation fold'] != fold, :] #Exclude the testing examples
		train_iter.reset_index(drop=True, inplace=True)
		#dim = train_iter.shape #debug
		#input('%s is the dimensions of fold %d'%(str(dim), fold)) #debug
		train_sp = train_iter.loc[train_iter.loc[:,'Class']=='SP', :] #Eliminating negative examples to generate the profile
		
		#Profile generation (vH_train)
		train_seq_list = tra.cleavage_seq(train_sp)  #obtaining sequence list
		sequence_length = len(train_seq_list[0])
		one_hot_sequences= [tra.encode_seq(sequence, alphabet) for sequence in train_seq_list] #one hot encoding
		pspm = tra.PSPM_gen(one_hot_sequences, sequence_length) #Generating the PSPM matrix
		pswm = tra.PSWM_gen(pspm, aa_ratios_alphabet) #obtaining the PSWM matrix
		
		#Prediction
		to_predict = [seq for seq in train_iter['Sequence (first 50 N-terminal residues)']] #Positive and negative examples included. Also, the full 50 aa sequence is used. 
		predictions = pre.predict_seq(to_predict, pswm, alphabet)
		# dim = sum(np.isnan(predictions)) #debug
		#input('%s is the nan of predict in fold %d'%(str(dim), fold)) #debug
		
		#Creating dataframes
		predictions = pd.Series(predictions)
		#dim = predictions.isna().sum()
		#input('%d is the nan of predict in fold %d'%(dim, fold))
		train_iter.loc[:,'scores'] = predictions
		#dim = train_iter['scores'].isna().sum()
		#print('Predictions:'+'\n', train.head())
		#input('%d is the nan of scores in fold %d'%(dim, fold))
		train_iter.to_csv('iteration_'+str(fold)+'_vh_training.csv')
		
		
		
	
		
	


if __name__ == "__main__":
	#Opening the input examples file
	try:
		train_fh = sys.argv[1]
	except:
		train_fh = input("insert the training data path   ")
	
	train = pd.read_csv(train_fh, sep='\t')
	result = cross_validation_init(train, env.alphabet, env.aa_ratios_alphabet)
	#print(result) #debug
	
	
