#!/usr/bin/env python
"""
This module is meant to create a text file with the UNIPROT
accession number of those sequences that were misclassified.

Therefore, a false positive and false negative file will be created. 
"""

import sys
import pandas as pd

def extract_missclass(bench_fh):
	"""
	Thus function will run the main routine of this module
	
	bench = The filehandler to the CSV file that contains both the original data and 
	the SVM predictions (column \'scores\')
	path = path where the new false positive and false negative files will be added
	"""
	bench = bench = pd.read_csv(bench_fh)
	false_positives = bench.query('scores == 1 and `Class` == \"NO_SP\"').loc[:, 'UniProtKB accession'].tolist()
	false_negatives = bench.query('scores == 0 and `Class` == \"SP\"').loc[:, 'UniProtKB accession'].tolist()
	
	return false_positives, false_negatives
    
    
def print_files(false_positives, false_negatives, export_pos, export_neg):
	"""
	Prints the false positive and false negative files
	"""
	#False positives
	with open(export_pos, 'w') as f:
		for accession in false_positives:
			f.write(accession.strip()+'\n')
			
	#False negatives
	with open(export_neg, 'w') as f:
		for accession in false_negatives:
			f.write(accession.strip()+'\n')
			
	

if __name__ == '__main__':
	try:
		bench_pred_fh = sys.argv[1] #CSV file that contains both the original data and the SVM predictions
		export_positive = sys.argv[2] #path where the new false positive will be added
		export_negative = sys.argv[3] #path where the new false negatives will be added
		
		
	except IndexError:
		bench_pred_fh = input("Insert the benchmark csv file with predictions   ")
		export_positive = input("insert the export path where the false positive entries will be added     ")
		export_negative = input("insert the export path where the false negatives entries will be added    ")
	
	
	#Run flow
	fp, fn = extract_missclass(bench_pred_fh)
	print_files(fp, fn, export_positive, export_negative)
	
	
	
	
