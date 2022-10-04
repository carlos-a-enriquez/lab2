# Output graphs: aka. vH results



## Pipeline:
04/10/2022

### 1. Training cross validation and benchmark evaluation
```command time -v ./benchmark_validate.py ../input_data/training_set.tsv ../output_graphs3 ../input_data/benchmark_set.tsv y
```

### 2. Extraction of accession numbers for each confusion matrix category
```./misclassified.py ../output_graphs3/benchmark/benchmark_set_scores.csv ../output_graphs3/benchmark/
```

### 3. False positive evaluation for the vH benchmark results

