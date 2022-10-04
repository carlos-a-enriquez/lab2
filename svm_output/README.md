# svm output

## Pipeline

### 1. Training the model using different hyperparameter combinations and cross-validation. Find the best performing model. 
```
command time -v ./svm_train.py ../input_data/training_set.tsv ../svm_output/ >../svm_output/svm_out_28092022.txt

```

### 2. SVM model evaluation using the benchmark

```
./svm_benchmark.py ../svm_models/myModel.pkl.gz ../svm_output/benchmark/ ../input_data/benchmark_set.tsv 20  2 'scale' > ../svm_output/benchmark/eval_stats.txt

```

### 3. Extract the accession numbers of all confusion matrix groups
```
/misclassified.py ../svm_output/benchmark/benchmark_set_scores.csv ../svm_output/benchmark/

```

### 4. False positive benchmark evaluation
```
./false_positive_eval.py ../svm_output/benchmark/real_negatives.txt ../svm_output/benchmark/false_positives.txt > ../svm_output/benchmark/fpr_eval_stats.txt
```

