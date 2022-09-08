#! /usr/bin/env python

import pandas as pd
import sys
import numpy as np
from sklearn.preprocessing import OneHotEncoder

try:
	train_fh = argv[1]
except:
	train_fh = input("insert the training data path")
	
### Extracting sequences of interest

train = pd.read_csv(train_fh, sep='\t')
train_sp = train[train.Class=='SP']

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
    
train_seq_list = cleavage_seq(train_sp)


