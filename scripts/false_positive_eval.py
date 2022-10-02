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

import pandas as pd
import numpy as np
import uniprot_idmap as ut


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
	range, evidence = False, False

	for term in annotation.split(";"): #Split annotation
		term = term.strip()

		if term.startswith("TRANSMEM"):
			#checking previous annotation
			if range and evidence:
			#print("Passed with annotation number %d"%annotation_count)
			return True
			range, evidence = False, False #Checking a new annotation
			annotation_count += 1

			#Checking range
			range_TM_l, range_TM_h = term.split()[1].split("..") #extract the transmembrane range
			try:
			 range_TM_l = int(range_TM_l)
			except ValueError:
			continue
			if range_TM_l <= 50:
			range = True
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
	if range and evidence:
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
		

    


if __name__ == '__main__':
	#Opening files
	try:
		real_neg = sys.argv[1]
		false_pos = sys.argv[2]
	except IndexError:
		real_neg = input("insert the path for the real negative accesion id list   ")
		false_pos = input("insert the path for the false positive accesion id list   ")
		
	
    
