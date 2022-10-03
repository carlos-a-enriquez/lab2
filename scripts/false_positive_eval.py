#!/usr/bin/env python
"""
The purpose of this module is to verify two hypothetical causes for false positive 
results obtained by both the vH and SVM algorithms:

1. Transmembrane alpha-helical domains within the first 50 residues
2. Transit peptide signals in the N-terminal region. 

The potential overlap between "characteristic SP peptide regions",
transmembrane domains and other transit peptides justify this analysis.

Therefore, this module is meant to produce the following statistics:
A. The baseline FPR for a model benchmark evaluation
B. The FPR given only proteins that have a transmembrane domain in the first 50
residues. 

Therefore, the following is necessary as input:
1. Real negative (FN+TN) accession numbers
2. False positive accession numbers
"""

import sys
import pandas as pd
import numpy as np
import time
import uniprot_idmap as ut
import environmental_variables as env

start_time = time.time()


def parse_transmembrane(annotation, eco_exp):
	"""
	This function takes a particular transmembrane annotation
	and checks 

	1) If it is present in the first 50 residues.
	2) If the evidence code is non-automatic

	If the 2 conditions are fulfilled, 
	the funtions returns TRUE. Otherwise, FALSE. 
	#From here parse the eco code content from the simplest to
	hard ones like "ECO:0000250|UniProtKB:E9Q4Z2": 
	"""
	#Conditions
	if not annotation:
		return False

	#Initial conditions
	annotation_count = 0
	range_t, evidence = False, False

	for term in annotation.split(";"): #Split annotation
		term = term.strip()

		if term.startswith("TRANSMEM"):
			#checking previous annotation
			if range_t and evidence:
				#print("Passed with annotation number %d"%annotation_count)
				return True
			range_t, evidence = False, False #Checking a new annotation
			annotation_count += 1

			#Checking range_t
			range_TM_l, range_TM_h = term.split()[1].split("..") #extract the transmembrane range
			try:
				range_TM_h = int(range_TM_h)
			except ValueError:
				continue
			if range_TM_h <= 50:
				range_t = True
				#print("Transmembrane in the first 50 residues for annotation %d"%annotation_count)

		elif term.startswith("/evidence="):
			title, content = term.split("=")
			eco_code = content[1:12] #excluding the \" symbol and extracting the exact number of digits
			if eco_code in eco_exp:
				evidence = True
				#print("acceptable evidence for annotation %d"%annotation_count)

		else:
			pass

	#Checking last annotation
	if range_t and evidence:
		#print("Passed with annotation number %d"%annotation_count)
		return True
	else:
		return False
		
		
def parse_transit(annotation, eco_exp):
	"""
	This function takes a particular transit annotation
	and checks 

	1) If the evidence code is non-automatic

	If the first condition is fulfilled, 
	the funtions returns TRUE. Otherwise, FALSE. 
	#From here parse the eco code content from the simplest to
	hard ones like "ECO:0000250|UniProtKB:E9Q4Z2": 


	On the other hand, we will also extract the organelle information,
	which will be returned alongside the Boolean value.
	"""
	#Reject empty annotation
	if not annotation:
		return False, np.nan

	#Initial conditions
	annotation_count = 0
	evidence = False
	organelle = np.nan

	for term in annotation.split(";"): #Split annotation
		term = term.strip()

		if term.startswith("TRANSIT"):
			#checking previous annotation
			if evidence:
				#print("Passed with annotation number %d"%annotation_count)
				return True, organelle
			evidence, organelle = False, np.nan #Checking a new annotation
			annotation_count += 1
			
		elif term.startswith("/evidence="):
			title, content = term.split("=")
			eco_code = content[1:12] #excluding the \" symbol and extracting the exact number of digits
			if eco_code in eco_exp:
				evidence = True
				#print("acceptable evidence for annotation %d"%annotation_count)

		elif term.startswith("/note="):
			title, content = term.split("=")
			organelle = content[1:-1]

		else:
			pass

	#Checking last annotation
	if evidence:
		#print("Passed with annotation number %d"%annotation_count)
		return True, organelle
	else:
		return False, np.nan
		

def apply_parse_transit(x, codes):
	"""
	Defining apply function (needed to use df.apply() without lambda function)
	x= specific dataframe row obtained with .apply()
	codes= List of acceptable eco-codes (for filtering out automatic annotations)
	"""
	return parse_transit(x['Transit peptide'], codes)
	
def extracting_counts(x):
	"""Overlapping organelle counts are extracted by parsing
	the 'organelle' annotation.
	Note: this function is only meant to be used with valid organelle annotations
	(no NaN).
	"""
	#Initial conditions and checking that the string is valid
	mito, chloro, pero = False, False, False
	if (not isinstance(x['org_annotation'], str)) or x['org_annotation'].lower()=='nan':
		return mito, chloro, pero
	
	#Parsing multiple organelle annotations
	annot_list = x['org_annotation'].lower().strip().split("and")

	for annot in annot_list:
		annot = annot.strip()
		if annot == "mitochondrion":
			mito = True
		elif annot == "chloroplast":
			chloro = True
		elif annot == "Peroxisome":
			pero = True
  
	return mito, chloro, pero
		

def fpr(false_pos, real_neg):
	"""
	Generates the baseline false positive rate
	from the counts of accesion ids in each file
	(should correspond to confusion matrix based metrics).
	
	false_pos = list of FP accession ids
	real_neg = list of FP+TN accession ids
	"""
	return len(false_pos)/len(real_neg)
	
	
def fpr_transmembrane(false_pos, real_neg, eco_exp):
	"""
	Outputs the FPR for proteins with a transmembrane 
	domain in the first 50 residues. 
	Returns the FPR specific for examples of this subset.
	
	false_pos = list of FP accession ids
	real_neg = list of FP+TN accession ids
	eco_exp = List of acceptable eco-codes (for filtering out automatic annotations)
	"""
	#Obtaining the actual negative (FP+TN) count
	df_tot = ut.tsv_extractor(real_neg)
	transmembrane_tot = df_tot.loc[:,'Transmembrane'].apply(func=parse_transmembrane, args=(eco_exp,)).sum()
	
	#Obtaining the false positive count
	df_tot = ut.tsv_extractor(false_pos)
	transmembrane_fp = df_tot.loc[:,'Transmembrane'].apply(func=parse_transmembrane, args=(eco_exp,)).sum()
	
	#Obtaining the TM FPR	
	try:
		FPR = transmembrane_fp/transmembrane_tot
	except ZeroDivisionError:
		FPR = np.nan
		
	return FPR
	

def transit_analyze(df, eco_exp):
	"""
	This function is mean to carry out the procedure for extracting the 
	organelle-specific transit peptide annotation count for a specific 
	dataframe of examples. This is a subroutine of fpr_transit, 
	which will use this function twice: for false positives and false negatives.
	
	df = Dataframe produced by the function ut.tsv_extractor() from the 
	uniprot_idmap module. 
	
	eco_exp = List of acceptable eco-codes (for filtering out automatic annotations)
	"""
	#Applying the parsing function to the transit annotation column (as a series)
	transit_pass = df.apply(func=apply_parse_transit, axis=1, args=(eco_exp,), result_type='expand')\
	.rename(mapper = {0:'checks_out', 1:'org_annotation'}, axis=1)
	
	#Final amount of examples with valid transit peptide annotations
	all_transit_count = transit_pass.loc[:,'checks_out'].sum()
	
	#Finding the individual organelle counts:
	#Obtaining boolean columns for each possible organelle based on the annotation
	organelle_parsed = transit_pass.query('checks_out == True').apply(func=extracting_counts, axis=1, result_type='expand')\
	.rename(mapper = {0:'Mitochondria', 1:"Chloroplast", 2:"Peroxisome"}, axis=1)
	
	#Finding the final counts
	mitoch_count = len(organelle_parsed.query('Mitochondria == True').index)
	chloro_count = len(organelle_parsed.query('Chloroplast == True').index)
	perox_count = len(organelle_parsed.query('Peroxisome == True').index)
	
	#saving results to numpy array
	#counts = dict(all_transit=all_transit_count, mithocondria=mitoch_count, chloroplast=chloro_count, peroxisome=perox_count)
	counts = np.array([all_transit_count, mitoch_count, chloro_count, perox_count])
		
	
	return counts
	
	
def fpr_transit(false_pos, real_neg, eco_exp):
	"""
	Outputs the FPR for proteins with a transit signal peptide in the N terminal region.
	
	false_pos = list of FP accession ids
	real_neg = list of FP+TN accession ids
	eco_exp = List of acceptable eco-codes (for filtering out automatic annotations)
	"""
	#Obtaining the actual negative (FP+TN) count
	df_tot = ut.tsv_extractor(real_neg)
	
	
	
	
	
	

		

    


if __name__ == '__main__':
	#Opening files
	try:
		real_neg_fh = sys.argv[1]
		false_pos_fh = sys.argv[2]
	except IndexError:
		real_neg_fh = input("insert the path for the real negative accesion id list   ")
		false_pos_fh = input("insert the path for the false positive accesion id list   ")
		
		
	#Workflow
	false_pos, real_neg = (ut.parse_accession_list(acc_fh) for acc_fh in (false_pos_fh, real_neg_fh))
	FPR = fpr(false_pos, real_neg)
	FPR_TM = fpr_transmembrane(false_pos, real_neg, env.eco_exp)
	
	print("--- %0.2f seconds ---" % (time.time() - start_time))
	
		
	
    
