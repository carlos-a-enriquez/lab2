#! /usr/bin/env python

import pandas as pd
import sys
import numpy as np
#from sklearn.preprocessing import OneHotEncoder
import environmental_variables as env
import vH-train as tra



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
			for j,char in enumerate(seq[i:i+14]):
				window_score += pswm[alphabet.find(char), j] #select the residue row and then select the position column to obtain a score
			score_list.append(window_score)
		max_score = max(score_list)
		scores.append(max_score)
	return scores


if __name__ == "__main__":
	try:
		train_fh = sys.argv[1]
	except:
		train_fh = input("insert the training data path   ")
		
	train = pd.read_csv(train_fh, sep='\t')
	train_sp = train[train.Class=='SP'] #Used to call on vH-train.py
	to_predict = [seq for seq in data['Sequence (first 50 N-terminal residues)']]
	
	#Running workflow (vh-train)
	train_seq_list = tra.cleavage_seq(train_sp)  #obtaining sequence list
	sequence_length = len(train_seq_list[0])
	one_hot_sequences= [tra.encode_seq(sequence, env.alphabet) for sequence in train_seq_list] #one hot encoding
	pspm = tra.SPM_gen(one_hot_sequences, sequence_length) #Generating the PSPM matrix
	pswm = tra.PSWM_gen(pspm, env.aa_ratios_alphabet) #obtaining the PSWM matrix
	
	#Running workflow (vh-predict)
	predict_seq(to_predict, env.alphabet)
