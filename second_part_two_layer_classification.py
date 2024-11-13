import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
# lambda_list = [0.001, 0.01, 0.1, 1, 10, 100]
lambda_list = [0.001, 0.01, 0.1, 1, 2, 4, 6, 8, 10, 12, 14, 18, 20, 30, 50, 100]
C_list = [1 / l if l != 0 else 1e6 for l in lambda_list]


hidden_units_list = [(5,5), (10, 10), (20, 20), (30, 30), (30,10),  (40, 40), (50,20), (50,50)]  # Example tuples for two hidden layers

# Cross-validation setup
outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Initialize results storage
fold_results = []

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

# Create DataFrame to display results
results_df = pd.DataFrame(fold_results)

# Print the table
print("Cross-Validation Results:")
print(results_df.to_string(index=False))

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