#! /usr/bin/env python

import sys, os
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sn
import environmental_variables as env


def extract_sequences(training_fh):
	"""
	This function extracts the sequences from the dataframe.
	
	training_data = Assumes a filehandler that points to a .tsv
	 file with the 'Sequence (first 50 N-terminal residues)' column. 
	 
	 Returns a list of sequences. 
	"""
	train = pd.read_csv(training_fh, sep='\t')
	return list(train['Sequence (first 50 N-terminal residues)'])
	


def encode_sequences(sequences, k, alphabet):
	"""
	This function will return the 20 dimensional vector of amino acid composition 
	for every sequence in the training dataset dataframe.
	
	k: SP length to be used in order to extract the corresponding subsequence from
	each training example. This is one of the model's hyperparameters.
	
	training_data = An array like object that contains the list of sequences to be
	encoded. It is assumed to be the first N-terminal residues of the sequence.  
	
	alphabet = defines the aminoacid string alphabet and their default order. 
	
	Returns a numpy array in which axis 0 corresponds to different examples 
	and axis 1 corresponds to the different residue composition values. So, an array
	of shape (m,20) where n is the number of examples. 
	"""
	
	#Initialization
	encoded_seqs = np.zeros((len(sequences), 20))
	
	
	#Iterating over sequences
	for i,seq in enumerate(sequences):
		
		#Obtaining the residue count for the first k residues
		residue_count = {}	
		char_count = 0	
		for residue in seq[0:k]:
			residue_count[residue] = residue_count.get(residue, 0) + 1
			
		#Dict to array
		residue_comp_vector= np.array([residue_count[residue] for residue in alphabet])
		
		#Count to percentage composition
		residue_comp_vector = (residue_comp_vector/residue_comp_vector.sum())*100
		
		#Adding to matrix
		encoded_seqs[i,:] = residue_comp_vector
	
	return encoded_seqs
		
			
		
	
	




if __name__ == "__main__":
	#Opening the input examples file and defining the output image folder path
	try:
		train_fh = sys.argv[1]		
	except IndexError:
		train_fh = input("insert the training data path   ")
		
	#Encoding workflow
	sequences = extract_sequences(train_fh)
	encode = encode_sequences(sequences, 24, env.alphabet)
	print(encode)
		
	
	
