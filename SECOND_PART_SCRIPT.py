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
from dtuimldmtools import rlr_validate 
from dtuimldmtools import bmplot, feature_selector_lr


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
Y = X_raw - np.ones((N, 1)) * X_raw.mean(axis=0) #########solo media
Y2 = Y * (1 / np.std(Y, 0))
X_combined_standard = Y2


# # %% Sequential Feature Selection for Linear Regression
# # Define a linear regression model and cross-validation strategy
# model = LinearRegression()
# cv = KFold(n_splits=10, shuffle=True, random_state=42)
# # Ensure the first 7 features are always included (the players encoding)
X_toselect = X_combined_standard[:, 7:]

# %% QUESTA SAREBBE STATA UNA FEATURE SELECTION MA POI L'HO RIFATTA COME NELL'ESERCIZIO A DUE LIVELLI
# # Separate target variable `y` (last column) and predictors `X`
# X = X_toselect[:, :-1]
# y = X_toselect[:, -1]
# #%%
# N, M = X.shape  # Set dimensions of the dataset

# def custom_scorer(estimator, X, y):
#     return cross_val_score(estimator, X, y, cv=cv, scoring='neg_mean_squared_error').mean()

# # Sequential Feature Selector with custom scoring function
# sfs = SequentialFeatureSelector(model, 
#                                 n_features_to_select='auto', 
#                                 direction='forward', 
#                                 scoring=custom_scorer, 
#                                 cv=cv)

# # Fit the feature selector to the combined dataset
# sfs.fit(X, y)
# selected_features = sfs.get_support(indices=True)
# features_used = selected_features#np.concatenate([np.arange(7), selected_features + 7])
# #%%
# print("Selected features:", selected_features)
# print("USED features:", features_used)

# # %% Evaluate Model with Selected Features
# # Fit the model with the selected features and evaluate
# model.fit(X_combined_standard[:, features_used], y)
# scores = cross_val_score(model, X_combined_standard[:, features_used], y, cv=cv, scoring='neg_mean_squared_error')
# print("Cross-validated MSE:", -scores.mean())

# # %% Prepare Selected Features for Analysis
# # Create a new feature set, `X_selected`, with first 7 and selected features
# X_selected = X_combined_standard[:,  features_used]
# print("Shape of X_selected:", X_selected.shape)

# # Update attribute names for selected features
# attributeNames = [AllAttributeNames[i] for i in features_used]
# print("Updated attributeNames:", attributeNames)

#%% FUTURE SELECTION COME LA FA LUI IN 6_2_1
# Ensure the dataset is correctly formatted
X = X_toselect[:, :-1]  #IF I WANT TO USE THE PLAYER NAMES COMPULSORILY
# X = X_combined_standard[:, :-1]  # Exclude target column LIKE THIS IT WOULD CONSIDER THE PLAYER NAME LIKE THE OTHER FEATURES
y = X_combined_standard[:, -1]   # Target variable

# Define model and cross-validation strategy
model = LinearRegression()
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
inner_cv_folds = 10

# Prepare arrays to track errors and feature selection results
Error_train_fs = []
Error_test_fs = []
Features = np.zeros((X.shape[1], outer_cv.get_n_splits()), dtype=int)

# Cross-validation loop
for k, (train_index, test_index) in enumerate(outer_cv.split(X)):
    # Split data
    X_train, X_test = X[train_index, :], X[test_index, :]
    y_train, y_test = y[train_index], y[test_index]
    
    # Baseline errors without feature selection
    baseline_model = LinearRegression().fit(X_train, y_train)
    Error_train_nofeatures = np.mean((y_train - y_train.mean())**2)
    Error_test_nofeatures = np.mean((y_test - y_test.mean())**2)

    # Feature selection using feature_selector_lr
    selected_features, features_record, loss_record = feature_selector_lr(X_train, y_train, inner_cv_folds, display='')

    if len(selected_features) == 0:
        print('No features selected, the data cannot describe the outcomes.')
        Error_train_fs.append(Error_train_nofeatures)
        Error_test_fs.append(Error_test_nofeatures)
    else:
        # Fit model with selected features
        model_fs = LinearRegression().fit(X_train[:, selected_features], y_train)
        Error_train_fs.append(np.mean((y_train - model_fs.predict(X_train[:, selected_features]))**2))
        Error_test_fs.append(np.mean((y_test - model_fs.predict(X_test[:, selected_features]))**2))
        Features[selected_features, k] = 1

    print(f"Fold {k+1}/{outer_cv.get_n_splits()} complete")

# Display feature selection results
print("Features selected in each fold:")
# bmplot([f"Feature {i}" for i in range(X.shape[1])], range(1, Features.shape[1] + 1), -Features)
bmplot([f"Feature {i}" for i in AllAttributeNames[7:-1]], range(1, Features.shape[1] + 1), -Features)
print("\nTraining Error with Feature Selection:", np.mean(Error_train_fs))
print("Testing Error with Feature Selection:", np.mean(Error_test_fs))
print("R^2 with Feature Selection:", 1 - np.sum(Error_test_fs) / Error_test_nofeatures)

# Model Evaluation on Selected Features
selected_features = np.nonzero(np.sum(Features, axis=1))[0]  # features selected across all folds
print("Final selected features across folds (excluding names):", selected_features)

final_selected_features = np.concatenate((np.arange(7), selected_features + 7))
attributeNames = [AllAttributeNames[i] for i in final_selected_features]
print("SELECTED attributeNames:", attributeNames)

# Update attribute names for selected features
attributeNames = [AllAttributeNames[i] for i in final_selected_features]

# Fit model on entire dataset with selected features
model.fit(X_combined_standard[:, final_selected_features], y)
scores = cross_val_score(model, X_combined_standard[:, final_selected_features], y, cv=outer_cv, scoring='neg_mean_squared_error')
print("Cross-validated MSE on selected features:", -scores.mean())

# %% Finalize Data Matrix for Analysis

X = X_combined_standard[:, final_selected_features]
y = X_combined_standard[:, -1]

N, M = X.shape  # Set dimensions of the final dataset

# Add offset attribute
X = np.concatenate((np.ones((X.shape[0], 1)), X), 1)
attributeNames = ["Offset"] + attributeNames
M = M + 1

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 5
CV = model_selection.KFold(K, shuffle=True)
# CV = model_selection.KFold(K, shuffle=False)

# Values of lambda
lambdas = np.power(10.0, np.arange(-5, 9))

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


#%%