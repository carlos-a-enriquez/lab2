#! /usr/bin/env python

import pandas as pd
import sys
import numpy as np
#from sklearn.preprocessing import OneHotEncoder
import environmental_variables as env
import vH_train as tra



def predict_seq(sequences, pswm, alphabet):
	'''This function will return a list of scores given a list of sequences, 
	a PSWM matrix and an alphabet order. 
	The alphabet order should perfectly match that of the matrix.'''
	#window
	scores = []
	for seq in sequences:
		score_list = []
		for i in range(0,(len(seq)-15+1)):
			#window = 
			window_score = 0
			for j,char in enumerate(seq[i:i+15]):
				window_score += pswm[next(k for k,res in enumerate(alphabet) if res == char), j] #select the correct residue row (comprehension to '.find' within list) and then select the position column to obtain a score
			score_list.append(window_score)
		scores.append(max(score_list))
	#Rounding the scores
	scores_array = np.array(scores)
	scores_array = np.around(scores_array, 2)
	return scores_array
	



if __name__ == "__main__":
	try:
		train_fh = sys.argv[1]
	except:
		train_fh = input("insert the training data path   ")
		
	train = pd.read_csv(train_fh, sep='\t')
	train_sp = train[train.Class=='SP'] #Used to call on vH-train.py
	
	
	#Running workflow (vh-train)
	train_seq_list = tra.cleavage_seq(train_sp)  #obtaining sequence list
	sequence_length = len(train_seq_list[0])
	one_hot_sequences= [tra.encode_seq(sequence, env.alphabet) for sequence in train_seq_list] #one hot encoding
	pspm = tra.PSPM_gen(one_hot_sequences, sequence_length) #Generating the PSPM matrix
	pswm = tra.PSWM_gen(pspm, env.aa_ratios_alphabet) #obtaining the PSWM matrix
	
	#Running workflow (vh-predict)
	to_predict = [seq for seq in train['Sequence (first 50 N-terminal residues)']] #Positive and negative examples included. Also, the full 50 aa sequence is used. 
	predictions = predict_seq(to_predict, pswm, env.alphabet)
	
	
	#Adding predictions to dataframe
	predictions = pd.Series(predictions)
	train['scores'] = predictions
	print('Predictions:'+'\n', train.head())
	train.to_csv('vh_results.csv')
