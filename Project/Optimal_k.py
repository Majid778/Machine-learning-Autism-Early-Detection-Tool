import sys
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, cohen_kappa_score, mean_absolute_error, mean_squared_error
import numpy as np
import os
from joblib import dump, load
from pyAgrum.skbn import BNClassifier

kfile = '/home/nyuad/Capstone/Scikit/Optimal k/'

def print_and_save_evaluation_results(results, model_info, filename):
    with open(filename, 'w') as f:
        f.write(model_info + '\n\n')
        print(model_info + '\n')
        for key, value in results.items():
            if "Instances" in key:
                line = "{}: {}".format(key, int(value))
            else:
                line = "{}: {:.4f}".format(key, value) if isinstance(value, float) else "{}: {}".format(key, value)
            print(line)
            f.write(line + '\n')

if len(sys.argv) < 4:
    print("Usage: python3 feature_selection_training.py -s [F/M/N/feature_subset_filename] -t [RF/NB/DT/LR/SVM/model_filename] -k [number of top features to select (optional, only for F or M)]")
    sys.exit(1)

# Load data
data = pd.read_csv("master_otu.tsv", sep="\t")
X = data.drop(columns=['sra_name', 'Class'])
y = data['Class']

# Feature selection
k_range = None
if '-k' in sys.argv:
    k_index = sys.argv.index('-k') + 1
    if k_index < len(sys.argv):
        k_values = sys.argv[k_index].split(':')
        if len(k_values) == 2 and all(val.isdigit() for val in k_values):
            k_range = range(int(k_values[0]), int(k_values[1]) + 1)
        else:
            print("Invalid value for -k. Please provide a range in the format start:end.")
            sys.exit(1)
    else:
        print("Invalid value for -k. Please provide a range in the format start:end.")
        sys.exit(1)

feature_subset_loaded = False
if sys.argv[1] == "-s":
    if sys.argv[2] in ["F", "M", "N"]:
        if k_range is None and sys.argv[2] in ["F", "M"]:
            print("You must provide a value for -k when using feature selection methods 'F' or 'M'.")
            sys.exit(1)
        if sys.argv[2] == "N":
            X_new = X
            k_range = [X.shape[1]]
            feature_selection_method = "Baseline"
    elif os.path.isfile(sys.argv[2]):
        if k_range is not None:
            print("You cannot provide a value for -k when loading a feature subset file.")
            sys.exit(1)
        if sys.argv[2].endswith('_selected_features.npy'):
            # Load feature subset
            X_new = np.load(sys.argv[2])
            k_range = [X_new.shape[1]]
            feature_selection_method = "Loaded Feature Subset"
            feature_subset_loaded = True
            print(f"Feature subset loaded from {sys.argv[2]}")
        else:
            print("Invalid feature subset file provided.")
            sys.exit(1)
    else:
        print("Invalid feature selection method or file. Use 'F' for ANOVA F-value based, 'M' for Mutual Information based, 'N' for no feature selection, or provide a valid feature subset filename.")
        sys.exit(1)
else:
    print("Invalid argument for feature selection. Please use '-s'.")
    sys.exit(1)

# Classifier training, evaluation, and saving/loading
if sys.argv[3] == "-t":
    if sys.argv[4] in ["RF", "NB", "DT", "LR", "SVM", "AdaBoost_RF", "Bagging_RF", "Bagging_NB", "MLP", "BN"]:
        if sys.argv[4] == "RF":
            clf = RandomForestClassifier(n_estimators=100, random_state=1)
            classifier_name = "Random Forest"
        elif sys.argv[4] == "NB":
            clf = GaussianNB()
            classifier_name = "Naive Bayes"
        elif sys.argv[4] == "DT":
            clf = DecisionTreeClassifier(random_state=1)
            classifier_name = "Decision Tree"
        elif sys.argv[4] == "LR":
            clf = LogisticRegression(random_state=1)
            classifier_name = "Logistic Regression"
        elif sys.argv[4] == "SVM":
            clf = SVC(random_state=1)
            classifier_name = "SVM"
        elif sys.argv[4] == "AdaBoost_RF":
            clf = AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=100), random_state=1)
            classifier_name = "AdaBoost with Random Forest"
        elif sys.argv[4] == "Bagging_RF":
            clf = BaggingClassifier(base_estimator=RandomForestClassifier(n_estimators=100), random_state=1)
            classifier_name = "Bagging with Random Forest"
        elif sys.argv[4] == "Bagging_NB":
            clf = BaggingClassifier(base_estimator=GaussianNB(), random_state=1)
            classifier_name = "Bagging with Naive Bayes"
        elif sys.argv[4] == "MLP":
            clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, activation='relu', solver='adam', random_state=1)
            classifier_name = "Multilayer Perceptron"
        elif sys.argv[4] == "BN":
           clf = BNClassifier(learningMethod='MIIC', prior='Smoothing', priorWeight=1,
                     discretizationNbBins=3,discretizationStrategy="kmeans",discretizationThreshold=10)
           classifier_name = "Bayes Net"
    elif os.path.isfile(sys.argv[4]):
        clf = load(sys.argv[4])
        classifier_name = "Loaded Model"
        print(f"Model loaded from {sys.argv[4]}")
    else:
        print("Invalid classifier or model filename. Use 'RF' for Random Forest, 'NB' for Naive Bayes, 'DT' for Decision Tree, or provide a valid model filename.")
        sys.exit(1)
else:
    print("Invalid argument for classifier or model loading. Please use '-t'.")
    sys.exit(1)

# Custom evaluation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
scoring = {'accuracy': make_scorer(accuracy_score), 'kappa': make_scorer(cohen_kappa_score), 
            'mae': make_scorer(mean_absolute_error), 
            'rmse': make_scorer(lambda y, y_pred: np.sqrt(mean_squared_error(y, y_pred))), 
            'rae': make_scorer(lambda y, y_pred: np.mean(np.abs(y - y_pred)) / np.mean(np.abs(y - np.mean(y))) * 100), 
            'rrse': make_scorer(lambda y, y_pred: np.sqrt(np.mean((y - y_pred) ** 2)) / np.sqrt(np.mean((y - np.mean(y)) ** 2)) * 100)}

best_k = None
best_accuracy = 0
best_evaluation_results = None
for k in k_range:
    if sys.argv[2] in ["F", "M"]:
        if sys.argv[2] == "F":
            selector = SelectKBest(score_func=f_classif, k=k)
            X_new = selector.fit_transform(X, y)
            feature_selection_method = "ANOVA F-value"
        elif sys.argv[2] == "M":
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
            X_new = selector.fit_transform(X, y)
            feature_selection_method = "Mutual Information"
    
    results = cross_validate(clf, X_new, y, cv=cv, scoring=scoring)
    
    # Calculate total instances and incorrect instances
    total_instances = len(y)
    correct_instances = np.mean(results['test_accuracy']) * total_instances
    incorrect_instances = total_instances - correct_instances
    
    # Compile results
    evaluation_results = {
        "Correctly Classified Instances": correct_instances,
        "Incorrectly Classified Instances": incorrect_instances,
        "Accuracy": np.mean(results['test_accuracy']) * 100,
        "Kappa statistic": np.mean(results['test_kappa']),
        "Mean absolute error": np.mean(results['test_mae']),
        "Root mean squared error": np.mean(results['test_rmse']),
        "Relative absolute error": np.mean(results['test_rae']),
        "Root relative squared error": np.mean(results['test_rrse']),
        "Total Number of Instances": total_instances
    }
    
    # Print results
    model_info = f"Model: {classifier_name}\nFeature Selection Method: {feature_selection_method}\nNumber of Features Selected: {k}"
    print(model_info)
    print("Accuracy: {:.4f}".format(evaluation_results['Accuracy']))
    
    # Update best_k if this is the best accuracy so far
    if evaluation_results['Accuracy'] > best_accuracy:
        best_accuracy = evaluation_results['Accuracy']
        best_k = k
        best_evaluation_results = evaluation_results

print("\nBest Number of Features: {}".format(best_k))
print("Best Accuracy: {:.4f}".format(best_accuracy))

# Save best k evaluation results to a file
best_model_info = f"Model: {classifier_name}\nFeature Selection Method: {feature_selection_method}\nNumber of Features Selected: {best_k}"
evaluation_filename = f"{feature_selection_method} | {classifier_name} Optimal k value.txt"
print_and_save_evaluation_results(best_evaluation_results, best_model_info, kfile + evaluation_filename)
