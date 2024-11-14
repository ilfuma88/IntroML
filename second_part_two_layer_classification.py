""" 
GOOD
This code performs classification using Logistic Regression and ANN models,
evaluates them using cross-validation, and adds McNemar's test to compare the models.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from statsmodels.stats.contingency_tables import mcnemar, Table2x2
import matplotlib.pyplot as plt

# Load and preprocess data
filename = "raw data.csv"
df = pd.read_csv(filename, sep=";")
df_cleaned = df.dropna(axis=1, how='all').dropna(axis=0, how='all')
label_encoder = LabelEncoder()
df_cleaned['PLAYER'] = label_encoder.fit_transform(df_cleaned['PLAYER'])

X = df_cleaned.drop(columns=["PLAYER"]).values
y = df_cleaned['PLAYER'].values
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Define hyperparameters
lambda_list = [0.001, 0.01, 0.1, 1, 2, 4, 6, 8, 10, 12, 14, 18, 20, 30, 50, 100]
lambda_list = [0.001, 0.01, 0.1, 1, 2, 4, 6, 8, 10, 12, 14, 18, 20, 30, 50, 100]
C_list = [1 / l if l != 0 else 1e6 for l in lambda_list]
hidden_units_list = [ (10, 10), (20, 20), (30, 30), (30,10),  (40, 40), (40,15)]  # Example tuples for two hidden layers

# Cross-validation setup
outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Initialize results storage
fold_results = []
logreg_coefficients = []

# Initialize lists to collect all test predictions and true labels
y_test_all = []
baseline_preds_all = []
logreg_preds_all = []
ann_preds_all = []

for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
    X_train_outer, X_test_outer = X[train_idx], X[test_idx]
    y_train_outer, y_test_outer = y[train_idx], y[test_idx]

    # Baseline model
    baseline_model = DummyClassifier(strategy="most_frequent", random_state=42)
    baseline_model.fit(X_train_outer, y_train_outer)
    y_pred_baseline = baseline_model.predict(X_test_outer)
    error_baseline = np.mean(y_pred_baseline != y_test_outer)

    # Inner cross-validation splits
    inner_splits = list(inner_cv.split(X_train_outer, y_train_outer))

    # Hyperparameter tuning for Logistic Regression
    logreg_val_scores = []
    for lmbd, C in zip(lambda_list, C_list):
        val_scores = []
        for train_inner_idx, val_inner_idx in inner_splits:
            X_train_inner, X_val_inner = X_train_outer[train_inner_idx], X_train_outer[val_inner_idx]
            y_train_inner, y_val_inner = y_train_outer[train_inner_idx], y_train_outer[val_inner_idx]

            log_reg = LogisticRegression(penalty='l2', C=C, multi_class='multinomial',
                                         solver='lbfgs', random_state=42, max_iter=200)
            log_reg.fit(X_train_inner, y_train_inner)
            val_scores.append(log_reg.score(X_val_inner, y_val_inner))
        mean_val_score = np.mean(val_scores)
        logreg_val_scores.append(mean_val_score)

    # Select best hyperparameter for Logistic Regression
    best_index_logreg = np.argmax(logreg_val_scores)
    best_lambda = lambda_list[best_index_logreg]
    best_C = C_list[best_index_logreg]

    # Retrain Logistic Regression with best hyperparameter
    log_reg_final = LogisticRegression(penalty='l2', C=best_C, multi_class='multinomial',
                                       solver='lbfgs', random_state=42, max_iter=200)
    log_reg_final.fit(X_train_outer, y_train_outer)
    y_pred_logreg = log_reg_final.predict(X_test_outer)
    error_logreg = np.mean(y_pred_logreg != y_test_outer)

    # Store coefficients
    logreg_coefficients.append(log_reg_final.coef_)

    # Hyperparameter tuning for ANN
    ann_val_scores = []
    for hidden_units in hidden_units_list:
        val_scores = []
        for train_inner_idx, val_inner_idx in inner_splits:
            X_train_inner, X_val_inner = X_train_outer[train_inner_idx], X_train_outer[val_inner_idx]
            y_train_inner, y_val_inner = y_train_outer[train_inner_idx], y_train_outer[val_inner_idx]

            ann_model = MLPClassifier(hidden_layer_sizes=hidden_units, activation='relu', solver='adam',
                                      max_iter=200, random_state=42)
            ann_model.fit(X_train_inner, y_train_inner)
            val_scores.append(ann_model.score(X_val_inner, y_val_inner))
        mean_val_score = np.mean(val_scores)
        ann_val_scores.append(mean_val_score)

    # Select best hyperparameter for ANN
    best_index_ann = np.argmax(ann_val_scores)
    best_hidden_units = hidden_units_list[best_index_ann]

    # Retrain ANN with best hyperparameter
    ann_model_final = MLPClassifier(hidden_layer_sizes=best_hidden_units, activation='relu', solver='adam',
                                    max_iter=200, random_state=42)
    ann_model_final.fit(X_train_outer, y_train_outer)
    y_pred_ann = ann_model_final.predict(X_test_outer)
    error_ann = np.mean(y_pred_ann != y_test_outer)

    # Store results for the fold
    fold_results.append({
        'Fold': fold + 1,
        'Baseline Error': error_baseline,
        'LogReg Error': error_logreg,
        'Lambda (LogReg)': best_lambda,
        'ANN Error': error_ann,
        'Hidden Units (ANN)': best_hidden_units
    })

    # Collect predictions and true labels
    y_test_all.extend(y_test_outer)
    baseline_preds_all.extend(y_pred_baseline)
    logreg_preds_all.extend(y_pred_logreg)
    ann_preds_all.extend(y_pred_ann)

# Convert lists to numpy arrays
y_test_all = np.array(y_test_all)
baseline_preds_all = np.array(baseline_preds_all)
logreg_preds_all = np.array(logreg_preds_all)
ann_preds_all = np.array(ann_preds_all)

# Perform McNemar's tests
model_pairs = [
    ('Baseline', baseline_preds_all),
    ('Logistic Regression', logreg_preds_all),
    ('ANN', ann_preds_all)
]

mcnemar_results = []

for i in range(len(model_pairs)):
    for j in range(i+1, len(model_pairs)):
        model_a_name, model_a_preds = model_pairs[i]
        model_b_name, model_b_preds = model_pairs[j]

        # Create contingency table
        correct_a = (model_a_preds == y_test_all)
        correct_b = (model_b_preds == y_test_all)

        both_correct = np.sum(correct_a & correct_b)
        a_correct_b_incorrect = np.sum(correct_a & ~correct_b)
        a_incorrect_b_correct = np.sum(~correct_a & correct_b)
        both_incorrect = np.sum(~correct_a & ~correct_b)

        contingency_table = [[both_correct, a_correct_b_incorrect],
                             [a_incorrect_b_correct, both_incorrect]]

        # Perform McNemar's test
        result = mcnemar(contingency_table, exact=False, correction=True)
        
        # Calculate confidence interval for the odds ratio
        table = Table2x2(contingency_table)
        odds_ratio = table.oddsratio
        ci_low, ci_upp = table.oddsratio_confint()

        mcnemar_results.append({
            'Model A': model_a_name,
            'Model B': model_b_name,
            'Statistic': result.statistic,
            'p-value': result.pvalue,
            'Odds Ratio': odds_ratio,
            'CI Lower': ci_low,
            'CI Upper': ci_upp
        })

# Create DataFrame to display results
mcnemar_df = pd.DataFrame(mcnemar_results)

# Create DataFrame to display results
results_df = pd.DataFrame(fold_results)

# Print the cross-validation results
print("Cross-Validation Results:")
print(results_df.to_string(index=False))

# Print McNemar's test results
print("\nMcNemar's Test Results:")
print(mcnemar_df.to_string(index=False))

# Compute mean errors across folds
mean_errors = results_df[['Baseline Error', 'LogReg Error', 'ANN Error']].mean()
std_errors = results_df[['Baseline Error', 'LogReg Error', 'ANN Error']].std()

# Visualization of errors
model_names = ['Baseline', 'Logistic Regression', 'ANN']

plt.figure(figsize=(10, 6))
plt.bar(model_names, mean_errors, yerr=std_errors, capsize=5, alpha=0.7)
plt.title('Model Misclassification Error Across Folds', fontsize=16)
plt.ylabel('Misclassification Error', fontsize=14)
plt.xlabel('Models', fontsize=14)
plt.ylim(0, 1)
plt.grid(True)
plt.show()

# Compute average coefficients
average_coefficients = np.mean(logreg_coefficients, axis=0)

# Print average coefficients
feature_names = df_cleaned.drop(columns=["PLAYER"]).columns
for i, class_coefficients in enumerate(average_coefficients):
    print(f"Class {i} coefficients:")
    for feature_name, coef in zip(feature_names, class_coefficients):
        print(f"{feature_name}: {coef}")