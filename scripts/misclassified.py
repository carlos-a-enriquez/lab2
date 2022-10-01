#!/usr/bin/env python
"""
This module is meant to create a text file with the UNIPROT
accession number of those sequences that were misclassified.

Therefore, a false positive and false negative file will be created. 
In addition, the set of real negatives (TN+FP) and real positives (TP+FN)
will also be returned. 
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
	real_negatives = bench.query('`Class` == \"NO_SP\"').loc[:, 'UniProtKB accession'].tolist()
	real_positives = bench.query('`Class` == \"SP\"').loc[:, 'UniProtKB accession'].tolist()
	
	lists = false_positives, false_negatives, real_negatives, real_positives
	
	return lists
    
    
def print_files(lists, export_path_s):
	"""
	Prints the false positive and false negative files
	
	lists = Tuple containing the following:
	false_positives, false_negatives, real_negatives, real_positives
	
	export_path = Path were all the text files will be saved.
	"""
	false_positives, false_negatives, real_negatives, real_positives = lists
	#False positives
	with open(export_path_s+'false_positives.txt', 'w') as f:
		for accession in false_positives:
			f.write(accession.strip()+'\n')
			
	#False negatives
	with open(export_path_s+'false_negatives.txt', 'w') as f:
		for accession in false_negatives:
			f.write(accession.strip()+'\n')
	
	#Real negatives
	with open(export_path_s+'real_negatives.txt', 'w') as f:
		for accession in real_negatives:
			f.write(accession.strip()+'\n')
			
	#Real positives
	with open(export_path_s+'real_positives.txt', 'w') as f:
		for accession in real_positives:
			f.write(accession.strip()+'\n')
	
			
	

if __name__ == '__main__':
	try:
		bench_pred_fh = sys.argv[1] #CSV file that contains both the original data and the SVM predictions
		export_path = sys.argv[2] #path for: false_positives, false_negatives, real_negatives, real_positives
		
		
	except IndexError:
		bench_pred_fh = input("Insert the benchmark csv file with predictions   ")
		export_path = input("insert the export path where the files will be exported     ")
		
	
	
	if export_path[-1] != "/":
		export_path += "/"		
	
	#Run flow
	accessions = extract_missclass(bench_pred_fh)
	print_files(accessions, export_path)
	
	
	
	
