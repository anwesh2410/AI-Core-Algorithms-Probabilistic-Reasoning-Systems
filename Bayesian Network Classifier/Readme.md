Bayesian Network Classifier

Description:
A Python implementation of a Bayesian Network classifier for multi-label classification tasks.


Usage:

Run the classifier with training and testing data files:

python B22AI005_3.py train_file test_file

- train_file: Path to the training data file (without the `.txt` extension).
- test_file: Path to the testing data file (without the `.txt` extension).

Example:

python B22AI005_3.py /path/to/eurlex_train /path/to/eurlex_test

Data Format:

- Training and Testing Files: Each line contains label indices followed by feature indices and values.
- Example Line:
  1,3 10:0.5 20:1.2 30:0.3

Evaluation:

The script prints the precision at k=1 after prediction.
