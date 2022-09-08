#! /usr/bin/env python

import pandas as pd
import sys
import numpy as np
from sklearn.preprocessing import OneHotEncoder





def cleavage_seq(data):
    '''This function extracts the 15 residues of interest from each training example'''
    sequence = [seq for seq in data['Sequence (first 50 N-terminal residues)']]
    sp = [seq for seq in data['SP cleavage-site annotation']]
    seq_list = []
    for i in range(len(sequence)):
        #cleav_seq = ''
        for j in range(len(sequence[i])):
            if sp[i][j] == "S":
                pass
            elif sp[i][j-1] == "S":
                cleav_seq = sequence[i][j-13:j+2]
        seq_list.append(cleav_seq)
    return seq_list
    


# Function to encode sequences
def encode_seq(sequence, alphabet):
    """ One hot encoding will generate one numpy matrix for each sequence.
    This is a binary representation in which each 'inner list' has one 1 value that corresponds to one of the
    latters of the alphabet.
    Note: This function expects a single sequence"""
    alphabet = ['A', 'C', 'D', 'E', 'F', 'G','H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    integer_encoded = [char_to_int[char] for char in sequence]
    onehot_encoded = list()

    for value in integer_encoded:
        letter = [0 for _ in range(len(alphabet))]
        letter[value] = 1
        onehot_encoded.append(letter)

    return np.array(onehot_encoded).transpose()
    
    
def PSPM_gen(sequences):
	'''This function will generate the Position Specific Probability Matrix.
	The argument "sequences" expects a list of hot-encoded sequences'''
	pspm = np.ones((20, 15), dtype=int)
	for seq in sequences:
		#pass
		pspm = pspm+seq
	pspm = pspm/(len(sequences)+20)
	return pspm

    
    

  
  
if __name__ == "__main__":
	#Env variables
	alphabet = ['A', 'C', 'D', 'E', 'F', 'G','H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
	
	#Extracting sequences of interest
	try:
		train_fh = sys.argv[1]
	except:
		train_fh = input("insert the training data path   ")
		
	train = pd.read_csv(train_fh, sep='\t')
	train_sp = train[train.Class=='SP']
	
	#Running workflow
	train_seq_list = cleavage_seq(train_sp)  #obtaining sequence list
	one_hot_sequences= [encode_seq(sequence, alphabet) for sequence in train_seq_list] #one hot encoding
	pspm = PSPM_gen(one_hot_sequences) #Generating the PSPM matrix
	#print(one_hot_sequences, train_seq_list[0], pspm)
	print(pspm, pspm.shape)  #debugging one-hot encoding
	
	


