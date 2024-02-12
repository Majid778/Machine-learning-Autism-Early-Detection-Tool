import sys
import pandas as pd
import numpy as np
import os
from joblib import dump, load
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (make_scorer, accuracy_score, cohen_kappa_score, mean_absolute_error,
                             mean_squared_error, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix)
from pyAgrum.skbn import BNClassifier


evalfile = '/home/nyuad/Capstone/Scikit/Evaluation Results/'
modelfile = '/home/nyuad/Capstone/Scikit/Models/'
featurefile = '/home/nyuad/Capstone/Scikit/Selected OTUs/'

def print_and_save_evaluation_results(results, model_info, filename):
    with open(filename, 'w') as f:
        f.write(model_info + '\n\n')
        print(model_info + '\n')
        for key, value in results.items():
            if "Instances" in key or "Matrix" in key:
                line = "{}: {}".format(key, value)
            else:
                line = "{}: {:.4f}".format(key, value) if isinstance(value, float) else "{}: {}".format(key, value)
            print(line)
            f.write(line + '\n')

# Check for correct number of arguments
if len(sys.argv) < 4:
    print("Usage: python3 feature_selection_training.py -s [F/M/N/feature_subset_filename] -t [RF/NB/DT/LR/SVM/model_filename] -k [number of top features to select (optional, only for F or M)]")
    sys.exit(1)

# Load data
data = pd.read_csv("master_otu.tsv", sep="\t")
X = data.drop(columns=['sra_name', 'Class'])
y = data['Class']

# Feature selection
k = None
selected_features_indices = None
if '-k' in sys.argv :
    k_index = sys.argv.index('-k') + 1
    if k_index < len(sys.argv) and sys.argv[k_index].isdigit():
        k = int(sys.argv[k_index])
    else:
        print("Invalid value for -k. Please provide a positive integer.")
        sys.exit(1)

feature_subset_loaded = False
if sys.argv[1] == "-s":
    if sys.argv[2] in ["F", "M", "N", "H"]:
        if k is None and sys.argv[2] in ["F", "M"]:
            print("You must provide a value for -k when using feature selection methods 'F' or 'M'.")
            sys.exit(1)
        if sys.argv[2] == "F":
            selector = SelectKBest(score_func=f_classif, k=k)
            X_new = selector.fit_transform(X, y)
            selected_features_indices = selector.get_support(indices=True)
            feature_selection_method = "ANOVA F-value"
        elif sys.argv[2] == "M":
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
            X_new = selector.fit_transform(X, y)
            selected_features_indices = selector.get_support(indices=True)
            feature_selection_method = "Mutual Information"
        elif sys.argv[2] == "H":
            Hdata = pd.read_csv("hierarchical_featureset.csv", sep=",")
            X_new = Hdata.drop(columns=['label'])
            y = Hdata['label']
            y = y.replace('H', 0)
            y = y.replace('A', 1)
            k = X_new.shape[1]
            feature_selection_method = "Hierarchical"
        elif sys.argv[2] == "N":
            X_new = X
            k = X.shape[1]
            feature_selection_method = "Baseline"
    elif os.path.isfile(sys.argv[2]):
        if k is not None:
            print("You cannot provide a value for -k when loading a feature subset file.")
            sys.exit(1)
        if sys.argv[2].endswith('.npy'):
            # Load feature subset
            X_new = np.load(sys.argv[2])
            k = X_new.shape[1]
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
    if sys.argv[4] in ["RF", "NB", "DT", "LR", "SVM", "AdaBoost_RF", "Bagging_RF", "Bagging_NB", "MLP"]:
        if sys.argv[4] == "RF":
            clf = RandomForestClassifier(n_estimators=100, random_state=1)
            model_filename = f"{feature_selection_method} | {k} Features | Random Forest Model.joblib"
            classifier_name = "Random Forest"
        elif sys.argv[4] == "NB":
            clf = GaussianNB()
            model_filename = f"{feature_selection_method} | {k} Features | Naive Bayes Model.joblib"
            classifier_name = "Naive Bayes"
        elif sys.argv[4] == "DT":
            clf = DecisionTreeClassifier(random_state=1)
            model_filename = f"{feature_selection_method} | {k} Features | Decision Tree Model.joblib"
            classifier_name = "Decision Tree"
        elif sys.argv[4] == "LR":
            clf = LogisticRegression(random_state=1)
            model_filename = f"{feature_selection_method} | {k} Features | Logistic Regression Model.joblib"
            classifier_name = "Logistic Regression"
        elif sys.argv[4] == "SVM":
            clf = SVC(random_state=1)
            model_filename = f"{feature_selection_method} | {k} Features | SVM Model.joblib"
            classifier_name = "SVM"
        elif sys.argv[4] == "AdaBoost_RF":
            clf = AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=100), random_state=1)
            model_filename = f"{feature_selection_method} | {k} Features | AdaBoost with RF Model.joblib"
            classifier_name = "AdaBoost with Random Forest"
        elif sys.argv[4] == "Bagging_RF":
            clf = BaggingClassifier(base_estimator=RandomForestClassifier(n_estimators=100), random_state=1)
            model_filename = f"{feature_selection_method} | {k} Features | Bagging with RF Model.joblib"
            classifier_name = "Bagging with Random Forest"
        elif sys.argv[4] == "Bagging_NB":
            clf = BaggingClassifier(base_estimator=GaussianNB(), random_state=1)
            model_filename = f"{feature_selection_method} | {k} Features | Bagging with Naive Bayes Model.joblib"
            classifier_name = "Bagging with Naive Bayes"
        elif sys.argv[4] == "MLP":
            clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, activation='relu', solver='adam', random_state=1)
            model_filename = f"{feature_selection_method} | {k} Features | MLP Model.joblib"
            classifier_name = "Multilayer Perceptron"
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

def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred).ravel()[0]
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred).ravel()[1]
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred).ravel()[2]
def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred).ravel()[3]

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'kappa': make_scorer(cohen_kappa_score),
    'mae': make_scorer(mean_absolute_error),
    'rmse': make_scorer(lambda y, y_pred: np.sqrt(mean_squared_error(y, y_pred))),
    'rae': make_scorer(lambda y, y_pred: np.mean(np.abs(y - y_pred)) / np.mean(np.abs(y - np.mean(y))) * 100),
    'rrse': make_scorer(lambda y, y_pred: np.sqrt(np.mean((y - y_pred) ** 2)) / np.sqrt(np.mean((y - np.mean(y)) ** 2)) * 100),
    'precision': make_scorer(precision_score, zero_division=0),
    'recall': make_scorer(recall_score, zero_division=0),
    'f1_score': make_scorer(f1_score, zero_division=0),
    # Confusion matrix components
    'tn': make_scorer(tn), 
    'fp': make_scorer(fp), 
    'fn': make_scorer(fn), 
    'tp': make_scorer(tp),
    # Include AUC if the classifier supports probability estimates
    'roc_auc': 'roc_auc' if hasattr(clf, "predict_proba") else None
}
results = cross_validate(clf, X_new, y, cv=cv, scoring=scoring)

# Calculate total instances and incorrect instances
total_instances = len(y)
correct_instances = np.mean(results['test_accuracy']) * total_instances
incorrect_instances = total_instances - correct_instances


# Compile results including new metrics
evaluation_results = {
    "Correctly Classified Instances": round(correct_instances),
    "Incorrectly Classified Instances": round(incorrect_instances),
    "Accuracy": np.mean(results['test_accuracy']) * 100,
    "Kappa statistic": np.mean(results['test_kappa']),
    "Mean absolute error": np.mean(results['test_mae']),
    "Root mean squared error": np.mean(results['test_rmse']),
    "Relative absolute error": np.mean(results['test_rae']),
    "Root relative squared error": np.mean(results['test_rrse']),
    "Precision": np.mean(results['test_precision']),
    "Recall": np.mean(results['test_recall']),
    "F1 Score": np.mean(results['test_f1_score']),
    "Total Number of Instances": total_instances,
    "Confusion Matrix": {
        "True Negative": int(np.sum(results['test_tn'])),
        "False Positive": int(np.sum(results['test_fp'])),
        "False Negative": int(np.sum(results['test_fn'])),
        "True Positive": int(np.sum(results['test_tp']))
    },
    # Include AUC in the results if it was calculated
    "AUC": np.mean(results['test_roc_auc']) if 'test_roc_auc' in results else "Not available"
}

# Print and save the updated evaluation results
model_info = f"Model: {classifier_name}\nFeature Selection Method: {feature_selection_method}\nNumber of Features Selected: {k}"
evaluation_filename = f"{feature_selection_method} | {k} Features | {classifier_name} Evaluation Results.txt"
print_and_save_evaluation_results(evaluation_results, model_info,  evalfile + evaluation_filename)


# Train final model and save
if sys.argv[4] in ["RF", "NB", "DT", "LR", "SVM", "AdaBoost_RF", "Bagging_RF", "Bagging_NB", "MLP"]:
    clf.fit(X_new, y)
    dump(clf, modelfile + model_filename)
    print(f"Model saved to {model_filename}")

# Save feature subset if not loaded
if not feature_subset_loaded and sys.argv[2] in ["F", "M", "H"]:
    # subset_filename = f"{feature_selection_method} | {k} Features Selected.npy"
    # np.save(subset_filename, X_new)
    # print(f"Feature subset saved to {subset_filename}")
    
    # Save selected feature names
    selected_feature_names = X.columns[selected_features_indices]
    feature_names_filename = f"{feature_selection_method} | {k} Features Selected OTUs.txt"
    np.savetxt(featurefile + feature_names_filename, selected_feature_names, fmt='%s')
    print(f"Selected feature names saved to {feature_names_filename}")
