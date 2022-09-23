#Env variables


alphabet = ['A', 'C', 'D', 'E', 'F', 'G','H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

#Swissprot background composition
aa_background_composition = {}
aa_list=['A','Q','L','S','R','E','K','T','N','G','M','W','D','H','F','Y','C','I','P','V']
aa_percentage=[8.25,3.93,9.65,6.64,5.53,6.72,5.80,5.35,4.06,7.07,2.41,1.10,5.46,2.27,3.86,2.92,1.38,5.91,4.74,6.86]

for i,j in zip(aa_list,aa_percentage):
  aa_background_composition[i]=j
  
aa_ratios_alphabet = [aa_background_composition[residue]/100 for residue in alphabet] #changing from percentage to ratio and in the alphabet order


#Sequences for debugging (using Castrense's example)
castrense_seq = ['STAAQAEP', 'AVESSPIF', 'LTVALAAE', 'LSLSQSTN', 'MIGVESVR', 'SKPTRAFS']
debug_alphabet = [residue for residue in 'ARNDCQEGHILKMFPSTWYV']
aa_ratios_debug = [aa_background_composition[residue]/100 for residue in debug_alphabet]




#SVM grid search hyperparameter lists
k_list= [20, 22, 24]
c_list = [1, 2, 4]
gamma_list = [0.5, 1, "scale"]





