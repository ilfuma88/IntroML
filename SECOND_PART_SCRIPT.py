# # PROJECT 2 INTRO ML GROUP 248

# %% Import libraries
import os
import importlib_resources
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector
from scipy.linalg import svd
from matplotlib.pylab import (
    figure,
    grid,
    legend,
    loglog,
    semilogx,
    show,
    subplot,
    title,
    xlabel,
    ylabel,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, KFold
from sklearn import model_selection
from dtuimldmtools import *
from sklearn.metrics import mean_squared_error 
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

# Set the working directory (adjust the path as necessary)
os.chdir(r"C:\Users\elefa\OneDrive - Danmarks Tekniske Universitet\DTU\FALL2024\02450_ITMLADM\PROJECT\IntroML")

# %% Load dataset
filename = "raw data.csv"
df = pd.read_csv(filename, sep=";")
# Remove columns and rows with all NaN values
df_cleaned = df.dropna(axis=1, how='all').dropna(axis=0, how='all')

print(df_cleaned.head())
# %% Encode player names as integer labels
# Encode player names to numerical values for machine learning
players = df_cleaned.iloc[:, 0]  # Assuming first column contains player names
player_codes = players.astype('category').cat.codes
K = player_codes.max() + 1  # Number of unique players

# Create one-hot encoding for players and append to dataset
player_encoding = np.zeros((player_codes.size, K))
player_encoding[np.arange(player_codes.size), player_codes] = 1

# Combine one-hot encoded players with the rest of the dataset (without player names)
X_raw = np.hstack((player_encoding, df_cleaned.iloc[:, 1:].values))

# Separate numeric features for further processing
features = df_cleaned.columns[1:]  # Skipping player names column
df_numeric = df_cleaned[features]
AllAttributeNames = players.unique().tolist() + features.tolist()

# %% STANDARDIZE DATA ULTRA IMPORTANTE
N, M = X_raw.shape  # Set dimensions of the dataset
# Standardize data prima di tutto

X_tosstandard = X_raw[:, 7:]
X_stand_mean = X_tosstandard - np.ones((N, 1)) * X_tosstandard.mean(axis=0) #########solo media
X_noplayers_standard = X_stand_mean * (1 / np.std(X_stand_mean, 0)) #########solo deviazione standard
X_combined_standard = np.hstack((X_raw[:, :7], X_noplayers_standard))
### qua è standardizzato anche plus minus
# print(X_combined_standard)
# %% Finalize Data Matrix for Analysis

# X = X_combined_standard[:, final_selected_features]
X = X_combined_standard[:, :-1]
y = X_combined_standard[:, -1]

N, M = X.shape  # Set dimensions of the final dataset

# Add offset attribute
X = np.concatenate((np.ones((X.shape[0], 1)), X), 1)
attributeNames = ["Offset"] + AllAttributeNames
M = M + 1

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 5
CV = model_selection.KFold(K, shuffle=True)
# CV = model_selection.KFold(K, shuffle=False)

# Values of lambda
lambdas = np.power(10.0, np.arange(-5, 7))

# Initialize variables
# T = len(lambdas)
Error_train = np.empty((K, 1))
Error_test = np.empty((K, 1))
Error_train_rlr = np.empty((K, 1))
Error_test_rlr = np.empty((K, 1))
Error_train_nofeatures = np.empty((K, 1))
Error_test_nofeatures = np.empty((K, 1))
w_rlr = np.empty((M, K))
mu = np.empty((K, M - 1))
sigma = np.empty((K, M - 1))
w_noreg = np.empty((M, K))

k = 0
for train_index, test_index in CV.split(X, y):
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    internal_cross_validation = 10

    (
        opt_val_err,
        opt_lambda,
        mean_w_vs_lambda,
        train_err_vs_lambda,
        test_err_vs_lambda,
    ) = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)

    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model (they would be needed for
    # making new predictions) - for brevity we won't always store these in the scripts
    ## QUI STANDARDIZZA DI NUOVO CHE FARE???????????????????????????
    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)

    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :]) / sigma[k, :]
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :]) / sigma[k, :]

    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train

    # Compute mean squared error without using the input data at all
    Error_train_nofeatures[k] = (
        np.square(y_train - y_train.mean()).sum(axis=0) / y_train.shape[0]
    )
    Error_test_nofeatures[k] = (
        np.square(y_test - y_test.mean()).sum(axis=0) / y_test.shape[0]
    )

    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0, 0] = 0  # Do no regularize the bias term
    w_rlr[:, k] = np.linalg.solve(XtX + lambdaI, Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    Error_train_rlr[k] = (
        np.square(y_train - X_train @ w_rlr[:, k]).sum(axis=0) / y_train.shape[0]
    )
    Error_test_rlr[k] = (
        np.square(y_test - X_test @ w_rlr[:, k]).sum(axis=0) / y_test.shape[0]
    )

    # Estimate weights for unregularized linear regression, on entire training set
    w_noreg[:, k] = np.linalg.solve(XtX, Xty).squeeze()
    # Compute mean squared error without regularization
    Error_train[k] = (
        np.square(y_train - X_train @ w_noreg[:, k]).sum(axis=0) / y_train.shape[0]
    )
    Error_test[k] = (
        np.square(y_test - X_test @ w_noreg[:, k]).sum(axis=0) / y_test.shape[0]
    )
    # OR ALTERNATIVELY: you can use sklearn.linear_model module for linear regression:
    # m = lm.LinearRegression().fit(X_train, y_train)
    # Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
    # Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]

    # Display the results for the last cross-validation fold
    if k == K - 1:
        figure(k, figsize=(12, 8))
        subplot(1, 2, 1)
        semilogx(lambdas, mean_w_vs_lambda.T[:, 1:], ".-")  # Don't plot the bias term
        xlabel("Regularization factor")
        ylabel("Mean Coefficient Values")
        grid()
        # You can choose to display the legend, but it's omitted for a cleaner
        # plot, since there are many attributes
        # legend(attributeNames[1:], loc='best')

        subplot(1, 2, 2)
        title("Optimal lambda: 1e{0}".format(np.log10(opt_lambda)))
        loglog(
            lambdas, train_err_vs_lambda.T, "b.-", lambdas, test_err_vs_lambda.T, "r.-"
        )
        xlabel("Regularization factor")
        ylabel("Squared error (crossvalidation)")
        legend(["Train error", "Validation error"])
        grid()

    # To inspect the used indices, use these print statements
    # print('Cross validation fold {0}/{1}:'.format(k+1,K))
    # print('Train indices: {0}'.format(train_index))
    # print('Test indices: {0}\n'.format(test_index))

    k += 1

show()
# Display results
print("Linear regression without feature selection:")
print("- Training error: {0}".format(Error_train.mean()))
print("- Test error:     {0}".format(Error_test.mean()))
print(
    "- R^2 train:     {0}".format(
        (Error_train_nofeatures.sum() - Error_train.sum())
        / Error_train_nofeatures.sum()
    )
)
print(
    "- R^2 test:     {0}\n".format(
        (Error_test_nofeatures.sum() - Error_test.sum()) / Error_test_nofeatures.sum()
    )
)
print("Regularized linear regression:")
print("- Training error: {0}".format(Error_train_rlr.mean()))
print("- Test error:     {0}".format(Error_test_rlr.mean()))
print(
    "- R^2 train:     {0}".format(
        (Error_train_nofeatures.sum() - Error_train_rlr.sum())
        / Error_train_nofeatures.sum()
    )
)
print(
    "- R^2 test:     {0}\n".format(
        (Error_test_nofeatures.sum() - Error_test_rlr.sum())
        / Error_test_nofeatures.sum()
    )
)

print("Weights in last fold:")
for m in range(M):
    print("{:>15} {:>15}".format(attributeNames[m], np.round(w_rlr[m, -1], 2)))



# Sample data (replace with actual data)
X = X_combined_standard[:, :-1]
y = X_combined_standard[:, -1]

# Hyperparameter ranges
hidden_units = [1]  # Example for ANN
lambdas = [0.01, 0.05, 0.1, 1, 10, 100]     # Example for Ridge regression

# Outer and inner CV folds
K1 = 5  # Outer CV
K2 = 5  # Inner CV

# Prepare for results collection
results = []

# Outer CV loop
outer_cv = KFold(n_splits=K1, shuffle=True, random_state=42)

for i, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Variables to store best hyperparameters and errors
    best_h = None
    best_lambda = None
    best_ann_error = np.inf
    best_lr_error = np.inf

    # Inner CV for hyperparameter tuning
    inner_cv = KFold(n_splits=K2, shuffle=True, random_state=42)

    for h in hidden_units:
        for lmbd in lambdas:
            ann_errors = []
            lr_errors = []

            # Inner CV loop
            for inner_train_idx, inner_val_idx in inner_cv.split(X_train):
                X_inner_train, X_val = X_train[inner_train_idx], X_train[inner_val_idx]
                y_inner_train, y_val = y_train[inner_train_idx], y_train[inner_val_idx]

                # Train ANN
                ann = MLPRegressor(hidden_layer_sizes=(h,), max_iter=1000, random_state=42)
                ann.fit(X_inner_train, y_inner_train)
                y_pred_ann = ann.predict(X_val)
                ann_errors.append(mean_squared_error(y_val, y_pred_ann))

                # Train Ridge Regression
                lr = Ridge(alpha=lmbd)
                lr.fit(X_inner_train, y_inner_train)
                y_pred_lr = lr.predict(X_val)
                lr_errors.append(mean_squared_error(y_val, y_pred_lr))

            # Evaluate and update best hyperparameters
            mean_ann_error = np.mean(ann_errors)
            mean_lr_error = np.mean(lr_errors)

            if mean_ann_error < best_ann_error:
                best_ann_error = mean_ann_error
                best_h = h
            if mean_lr_error < best_lr_error:
                best_lr_error = mean_lr_error
                best_lambda = lmbd

    # Train with best hyperparameters on full training set
    ann_best = MLPRegressor(hidden_layer_sizes=(best_h,), max_iter=1000, random_state=42)
    ann_best.fit(X_train, y_train)
    ann_test_error = mean_squared_error(y_test, ann_best.predict(X_test))

    lr_best = Ridge(alpha=best_lambda)
    lr_best.fit(X_train, y_train)
    lr_test_error = mean_squared_error(y_test, lr_best.predict(X_test))

    # Baseline: Predict mean of y_train for all test samples
    baseline_prediction = np.mean(y_train)
    baseline_test_error = mean_squared_error(y_test, np.full_like(y_test, baseline_prediction))

    # Append results for the current fold
    results.append({
        'fold': i + 1,
        'best_h': best_h,
        'ann_test_error': ann_test_error,
        'best_lambda': best_lambda,
        'lr_test_error': lr_test_error,
        'baseline_test_error': baseline_test_error
    })

# Print results table
print("Fold | Best h | ANN Test Error | Best λ | Linear Regression Test Error | Baseline Test Error")
for res in results:
    print(f"{res['fold']} | {res['best_h']} | {res['ann_test_error']:.2f} | {res['best_lambda']} | {res['lr_test_error']:.2f} | {res['baseline_test_error']:.2f}")
