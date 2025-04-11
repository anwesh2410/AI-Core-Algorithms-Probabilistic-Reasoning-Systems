import numpy as np
from scipy.sparse import csr_matrix
import re
from collections import defaultdict
from sklearn.base import BaseEstimator, ClassifierMixin
import sys

def load_sparse_data(file_path):
    with open(file_path, 'r') as file:
        num_data_points, feature_dim, num_labels = map(int, file.readline().strip().split())
        rows, cols, data = [], [], []
        labels = []
        for i, line in enumerate(file):
            line = line.strip()
            match = re.match(r'^([\d,]*)\s*(.*)$', line)
            if not match:
                raise ValueError(f"Line format is incorrect: {line}")
            label_part, feature_part = match.groups()
            label_indices = list(map(int, label_part.split(','))) if label_part else []
            labels.append(label_indices)
            for feature in feature_part.split():
                if ':' in feature:
                    index, value = feature.split(':')
                    if index and value:
                        rows.append(i)
                        cols.append(int(index))
                        data.append(float(value))
        X = csr_matrix((data, (rows, cols)), shape=(num_data_points, feature_dim))
        # print(f"Loaded {num_data_points} data points with {feature_dim} features and {num_labels} labels.")
        # print(f"Number of data points with no labels: {sum(len(label_set) == 0 for label_set in labels)}")
        return X, labels, num_labels
    
class BayesianNetwork(BaseEstimator, ClassifierMixin):
    def __init__(self, max_parents=3, smooth_factor=1e-5):
        self.max_parents = max_parents
        self.smooth_factor = smooth_factor
        self.label_probs = None
        self.feature_probs = defaultdict(lambda: defaultdict(float))
        self.num_labels = None
        self.num_samples = 0

    def fit(self, X, y, num_labels):
        self.num_samples = X.shape[0]
        self.num_labels = num_labels
        print(f"Training on {self.num_samples} samples with {num_labels} possible labels...")
        
        # Calculate P(label) - Prior probabilities
        label_counts = np.zeros(num_labels)
        for label_set in y:
            for label in label_set:
                label_counts[label] += 1
        
        self.label_probs = {label: (count + self.smooth_factor) / 
                           (self.num_samples + self.smooth_factor * num_labels)
                           for label, count in enumerate(label_counts)}
        
        # Calculate P(feature|parents) - Conditional probabilities
        feature_parent_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        feature_parent_totals = defaultdict(lambda: defaultdict(int))
        
        for i in range(X.shape[0]):
            row = X[i]
            labels = y[i]
            feature_indices = row.indices
            feature_values = row.data
            
            for feat_idx, feat_val in zip(feature_indices, feature_values):
                # Find the max `max_parents` features that co-occur with this feature
                parent_indices = self._find_parents(X, i, feat_idx, self.max_parents)
                
                # Increment counts for this feature-parent combination
                for parent_idx in parent_indices:
                    feature_parent_counts[feat_idx][tuple(parent_indices)][parent_idx] += 1
                    feature_parent_totals[feat_idx][tuple(parent_indices)] += 1
        
        # Calculate P(feature|parents) with smoothing
        for feat_idx in feature_parent_counts:
            for parent_indices in feature_parent_counts[feat_idx]:
                total = feature_parent_totals[feat_idx][parent_indices]
                for parent_idx in feature_parent_counts[feat_idx][parent_indices]:
                    count = feature_parent_counts[feat_idx][parent_indices][parent_idx]
                    self.feature_probs[feat_idx][(*parent_indices, parent_idx)] = \
                        (count + self.smooth_factor) / (total + self.smooth_factor * (self.max_parents + 1))

    def _find_parents(self, X, sample_idx, feature_idx, max_parents):
        """Find the `max_parents` most co-occurring features for a given feature."""
        row = X[sample_idx]
        feature_indices = row.indices
        
        # Get all other feature indices that co-occur with the given feature
        co_occurring = [idx for idx in feature_indices if idx != feature_idx]
        
        # Sort by co-occurrence frequency and take the top `max_parents`
        parent_counts = defaultdict(int)
        for idx in co_occurring:
            parent_counts[idx] += 1
        sorted_parents = sorted(parent_counts.items(), key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in sorted_parents[:max_parents]]

    def predict_proba(self, X):
        n_samples = X.shape[0]
        probas = np.zeros((n_samples, self.num_labels))
        
        # print(f"Predicting probabilities for {n_samples} samples...")
        
        for i in range(n_samples):
            row = X[i]
            feature_indices = row.indices
            feature_values = row.data
            
            # Initialize with log of prior probabilities P(label)
            label_scores = np.log([self.label_probs[label] for label in range(self.num_labels)])
            
            # Add log of conditional probabilities P(feature|parents)
            for feat_idx, feat_val in zip(feature_indices, feature_values):
                # Find the parent features for this feature
                parent_indices = self._find_parents(X, i, feat_idx, self.max_parents)
                
                # Multiply probabilities for this feature and its parents
                for label in range(self.num_labels):
                    prob = self.label_probs[label]
                    for parent_idx in parent_indices:
                        prob *= self.feature_probs[feat_idx][(*parent_indices, parent_idx)]
                    label_scores[label] += np.log(prob)
            
            # Convert log probabilities to probabilities
            label_scores -= np.max(label_scores)
            exp_scores = np.exp(label_scores)
            probas[i] = exp_scores / np.sum(exp_scores)
        
        return probas

def precision_at_k(y_test, y_pred, k=1):
    correct = 0
    for i in range(len(y_test)):
        top_k_pred = np.argsort(y_pred[i])[-k:]
        if any(label in top_k_pred for label in y_test[i]):
            correct += 1
    return correct / len(y_test)


class BayesianNetwork1(BaseEstimator, ClassifierMixin):
    def __init__(self, smooth_factor=1e-2):  # Increased smoothing
        self.smooth_factor = smooth_factor
        self.label_probs = None
        self.max_parents = 2
        self.feature_probs = None
        self.num_labels = None
        self.num_features = None

    def fit(self, X, y, num_labels):
        self.num_samples, self.num_features = X.shape
        self.num_labels = num_labels
        # print(f"Training on {self.num_samples} samples with {num_labels} possible labels...")
        
        # Calculate P(label) - Prior probabilities
        label_counts = np.zeros(num_labels)
        for label_set in y:
            for label in label_set:
                label_counts[label] += 1
        self.label_probs = (label_counts + self.smooth_factor) / (self.num_samples + self.smooth_factor * num_labels)
        
        # Calculate P(feature|label) - Conditional probabilities
        self.feature_probs = np.ones((num_labels, self.num_features)) * self.smooth_factor  # Initialize with smoothing
        label_totals = label_counts + self.smooth_factor * 2  # For binary features

        for i in range(self.num_samples):
            row = X[i]
            labels_in_sample = y[i]
            feature_indices = row.indices
            for label in labels_in_sample:
                self.feature_probs[label, feature_indices] += 1
        
        self.feature_probs /= label_totals[:, None]
        # print("Training completed.")

    def predict_proba(self, X):
        n_samples = X.shape[0]
        probas = np.zeros((n_samples, self.num_labels))
        # print(f"Predicting probabilities for {n_samples} samples...")
        
        log_label_probs = np.log(self.label_probs + 1e-10)  # Avoid log(0)
        log_feature_probs = np.log(self.feature_probs + 1e-10)
        log_feature_neg_probs = np.log(1 - self.feature_probs + 1e-10)
        
        for i in range(n_samples):
            row = X[i]
            feature_indices = row.indices
            # Initialize with log P(label)
            label_scores = log_label_probs.copy()
            # Add log P(feature|label) for present features
            label_scores += log_feature_probs[:, feature_indices].sum(axis=1)
            # Optionally, add log P(~feature|label) for absent features
            absent_features = set(range(self.num_features)) - set(feature_indices)
            if absent_features:
                label_scores += log_feature_neg_probs[:, list(absent_features)].sum(axis=1)
            # Normalize using log-sum-exp
            max_score = np.max(label_scores)
            label_scores -= max_score
            exp_scores = np.exp(label_scores)
            probas[i] = exp_scores / np.sum(exp_scores)
        return probas

def precision_at_k(y_test, y_pred, k=1):
    correct = 0
    for i in range(len(y_test)):
        top_k_pred = np.argsort(y_pred[i])[-k:]
        if any(label in top_k_pred for label in y_test[i]):
            correct += 1
    return correct / len(y_test) if len(y_test) > 0 else 0

def select_top_features(X, top_n=1000):
    feature_sums = np.array(X.sum(axis=0)).flatten()
    top_indices = feature_sums.argsort()[-top_n:]
    return X[:, top_indices], top_indices

def main(train_file, test_file):
    train_file=train_file+".txt"
    test_file=test_file+".txt"
    X_train, y_train, num_labels_train = load_sparse_data(train_file)
    X_test, y_test, num_labels_test = load_sparse_data(test_file)
    num_labels = max(num_labels_train, num_labels_test)
    
    # Feature selection
    X_train, selected_features = select_top_features(X_train, top_n=5000)
    X_test = X_test[:, selected_features]
    
    model = BayesianNetwork1()
    model.fit(X_train, y_train, num_labels)
    
    y_pred = model.predict_proba(X_test)
    precision = precision_at_k(y_test, y_pred, k=1)
    print(f"{precision:.4f}")

if __name__ == "__main__":
    # main("/home/anwesh2410/Sem 5/AI/ASS3/Eurlex/eurlex_train.txt", "/home/anwesh2410/Sem 5/AI/ASS3/Eurlex/eurlex_test.txt")
    if len(sys.argv) != 3:
        print("Usage: ./runmycode train_file test_file")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])