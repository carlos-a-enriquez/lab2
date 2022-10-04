# vH Results: Training

In this folder, all training output files are stored. 
This includes:

- cross-validation csvs
- Confusion matrices for each cross validation
- Density distributions for the prediction scores of real negative and real positive samples
- Precision-recall curves

## Pipeline:
04/10/2022

1. Training cross validation and benchmark evaluation
```command time -v ./benchmark_validate.py ../input_data/training_set.tsv ../output_graphs3 ../input_data/benchmark_set.tsv y
```

2. Extraction of accession numbers for each 
