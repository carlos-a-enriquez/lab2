#! /usr/bin/env python

import pandas as pd
import sys
import numpy as np
#from sklearn.preprocessing import OneHotEncoder
import environmental_variables as env

""" 
The PSWM matrix is constructed with 20 rows (representing the 20 amino acids)
and as many columns as positions in the analyzed cleavage sequence. 

The order of the rows is that of the alphabet found in env.alphabet. 
"""



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
    Note: This function expects a single sequence
    Source: https://stackoverflow.com/a/69268524
    """
    #alphabet = ['A', 'C', 'D', 'E', 'F', 'G','H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    integer_encoded = [char_to_int[char] for char in sequence]
    onehot_encoded = list()

    for value in integer_encoded:
        letter = [0 for _ in range(len(alphabet))]
        letter[value] = 1
        onehot_encoded.append(letter)

    return np.array(onehot_encoded).transpose()
    
    
def PSPM_gen(sequences, sequence_length):
	'''This function will generate the Position Specific Probability Matrix.
	The argument "sequences" expects a list of hot-encoded sequences'''
	pspm = np.ones((20, sequence_length), dtype=int)
	for seq in sequences:
		#pass
		pspm = pspm+seq
	pspm = pspm/(len(sequences)+20)
	return pspm
	
	
def PSWM_gen(pspm, background_vector):
	'''This function determines the Position-specific weight matrix by using
	the PSPM and a background (Swissprot) distribution vector.
	The vector should be in ratios (not percentages) and in the order of the alphabet
	Note: only lists or numpy arrays are accepted for the background_vector'''
	if isinstance(background_vector, list):
		background_vector = np.array(background_vector).reshape(20,1) #we need to broadcast this 20x1 transpose vector to the entire 20x15 matrix
	else:
		background_vector = background_vector.reshape(20,1) 
	pswm = pspm/background_vector #broadcasting the background distribution vector to divide all columns by the same values
	
	#Compute the logarithms
	pswm = np.around(np.log2(pswm),2)#obtaining the base 2 log and rounding to 2 decimals
	return pswm
		

    
    

  
  
if __name__ == "__main__":
	#Extracting sequences of interest
	try:
		train_fh = sys.argv[1]
	except:
		train_fh = input("insert the training data path   ")
		
	train = pd.read_csv(train_fh, sep='\t')
	train_sp = train[train.Class=='SP']
	
	#Running workflow
	train_seq_list = cleavage_seq(train_sp)  #obtaining sequence list
	sequence_length = len(train_seq_list[0])
	one_hot_sequences= [encode_seq(sequence, env.alphabet) for sequence in train_seq_list] #one hot encoding
	pspm = PSPM_gen(one_hot_sequences, sequence_length) #Generating the PSPM matrix
	#print(one_hot_sequences, train_seq_list[0], pspm)
	
	pswm = PSWM_gen(pspm, env.aa_ratios_alphabet) #obtaining the PSWM matrix
	print(pspm, pspm.shape, env.aa_ratios_alphabet)  #debugging one-hot encoding
	print('#################')
	print('TRAINING PSWM')
	print(pswm)
	#print(env.aa_ratios_alphabet, pspm/pswm)  #DEBUG: recovering the background division broadcast and checking that it matches
	
	
	#Debugging with Castrene's example sequences
	debug = False
	if debug:
		print('###########################')
		print('Debugging section')
		train_seq_list = env.castrense_seq 
		sequence_length = len(train_seq_list[0])
		one_hot_sequences= [encode_seq(sequence, env.debug_alphabet) for sequence in train_seq_list] #one hot encoding
		pspm = PSPM_gen(one_hot_sequences, sequence_length) #Generating the PSPM matrix
		pswm = PSWM_gen(pspm, env.aa_ratios_debug) #obtaining the PSWM matrix
		print(pspm)
		print(pspm.shape)
		print('Background vector'+'\n', env.aa_ratios_debug)
		print(pswm)
	
	


