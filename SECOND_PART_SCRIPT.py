# # PROJECT 2 INTRO ML GROUP 248

# %% Import libraries
import os
import importlib_resources
import torch
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
# os.chdir(r"C:\Users\elefa\OneDrive - Danmarks Tekniske Universitet\DTU\FALL2024\02450_ITMLADM\PROJECT\IntroML")

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


# Initial lambda values
initial_lambdas = [ 0.001, 0.01, 0.05, 0.1, 1, 10, 100, 1000, 10000]

# Generate additional values between 0.01 and 10
additional_lambdas = np.logspace(np.log10(10), np.log10(100), num=10)

# Combine the initial and additional values, ensuring uniqueness and sorting
lambdas = np.unique(np.concatenate((initial_lambdas, additional_lambdas)))

print(lambdas)


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
    # Display the results for the last cross-validation fold
    if k == K - 1:
        # figure(k, figsize=(12, 8))
        # subplot(1, 2, 1)
        figure(1)
        semilogx(lambdas, mean_w_vs_lambda.T[:, 1:], ".-")  # Don't plot the bias term
        xlabel("Regularization factor")
        ylabel("Mean Coefficient Values")
        grid()

        # subplot(1, 2, 2)
        figure(2)
        title("Optimal lambda: {0}".format(opt_lambda))
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



def train_neural_netNOPRINT(
    model, loss_fn, X, y, n_replicates=3, max_iter=10000, tolerance=1e-6
):
    """
    Train a neural network with PyTorch based on a training set consisting of
    observations X and class y. The model and loss_fn inputs define the
    architecture to train and the cost-function update the weights based on,
    respectively.

    Args:
        model: A function handle to make a torch.nn.Sequential.
        loss_fn: A torch.nn-loss, e.g. torch.nn.BCELoss() for binary
                 binary classification, torch.nn.CrossEntropyLoss() for
                 multiclass classification, or torch.nn.MSELoss() for
                 regression.
        X: The input observations as a PyTorch tensor.
        y: The target classes as a PyTorch tensor.
        n_replicates: An integer specifying number of replicates to train,
                      the neural network with the lowest loss is returned.
        max_iter: An integer specifying the maximum number of iterations
                  to do (default 10000).
        tolerance: A float describing the tolerance/convergence criterion
                   for minimum relative change in loss (default 1e-6)


    Returns:
        A list of three elements:
            best_net: A trained torch.nn.Sequential that had the lowest
                      loss of the trained replicates.
            final_loss: A float specifying the loss of best performing net.
            learning_curve: A list containing the learning curve of the best net.

    Usage:
        Assuming loaded dataset (X,y) has been split into a training and
        test set called (X_train, y_train) and (X_test, y_test), and
        that the dataset has been cast into PyTorch tensors using e.g.:
            X_train = torch.tensor(X_train, dtype=torch.float)
        Here illustrating a binary classification example based on e.g.
        M=2 features with H=2 hidden units:

        >>> # Define the overall architechture to use
        >>> model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, H),  # M features to H hiden units
                    torch.nn.Tanh(),        # 1st transfer function
                    torch.nn.Linear(H, 1),  # H hidden units to 1 output neuron
                    torch.nn.Sigmoid()      # final tranfer function
                    )
        >>> loss_fn = torch.nn.BCELoss() # define loss to use
        >>> net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train,
                                                       y=y_train,
                                                       n_replicates=3)
        >>> y_test_est = net(X_test) # predictions of network on test set
        >>> # To optain "hard" class predictions, threshold the y_test_est
        >>> See exercise ex8_2_2.py for indepth example.

        For multi-class with C classes, we need to change this model to e.g.:
        >>> model = lambda: torch.nn.Sequential(
                            torch.nn.Linear(M, H), #M features to H hiden units
                            torch.nn.ReLU(), # 1st transfer function
                            torch.nn.Linear(H, C), # H hidden units to C classes
                            torch.nn.Softmax(dim=1) # final tranfer function
                            )
        >>> loss_fn = torch.nn.CrossEntropyLoss()

        And the final class prediction is based on the argmax of the output
        nodes:
        >>> y_class = torch.max(y_test_est, dim=1)[1]
    """

    # Specify maximum number of iterations for training
    logging_frequency = 1000  # display the loss every 1000th iteration
    best_final_loss = 1e100
    for r in range(n_replicates):
        # print("\n\tReplicate: {}/{}".format(r + 1, n_replicates))
        # Make a new net (calling model() makes a new initialization of weights)
        net = model()

        # initialize weights based on limits that scale with number of in- and
        # outputs to the layer, increasing the chance that we converge to
        # a good solution
        torch.nn.init.xavier_uniform_(net[0].weight)
        torch.nn.init.xavier_uniform_(net[2].weight)

        # We can optimize the weights by means of stochastic gradient descent
        # The learning rate, lr, can be adjusted if training doesn't perform as
        # intended try reducing the lr. If the learning curve hasn't converged
        # (i.e. "flattend out"), you can try try increasing the maximum number of
        # iterations, but also potentially increasing the learning rate:
        # optimizer = torch.optim.SGD(net.parameters(), lr = 5e-3)

        # A more complicated optimizer is the Adam-algortihm, which is an extension
        # of SGD to adaptively change the learing rate, which is widely used:
        optimizer = torch.optim.Adam(net.parameters())

        # Train the network while displaying and storing the loss
        # print("\t\t{}\t{}\t\t\t{}".format("Iter", "Loss", "Rel. loss"))
        learning_curve = []  # setup storage for loss at each step
        old_loss = 1e6
        for i in range(max_iter):
            y_est = net(X)  # forward pass, predict labels on training set
            loss = loss_fn(y_est, y)  # determine loss
            loss_value = loss.data.numpy()  # get numpy array instead of tensor
            learning_curve.append(loss_value)  # record loss for later display

            # Convergence check, see if the percentual loss decrease is within
            # tolerance:
            p_delta_loss = np.abs(loss_value - old_loss) / old_loss
            if p_delta_loss < tolerance:
                break
            old_loss = loss_value

            # display loss with some frequency:
            if (i != 0) & ((i + 1) % logging_frequency == 0):
                print_str = (
                    "\t\t"
                    + str(i + 1)
                    + "\t"
                    + str(loss_value)
                    + "\t"
                    + str(p_delta_loss)
                )
                # print(print_str)
            # do backpropagation of loss and optimize weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # display final loss
        # print("\t\tFinal loss:")
        # print_str = (
        #     "\t\t" + str(i + 1) + "\t" + str(loss_value) + "\t" + str(p_delta_loss)
        # )
        # print(print_str)
        
        # Ensure loss_value is a float
        if isinstance(loss_value, np.ndarray):
            loss_value = loss_value.item()

        # Ensure best_final_loss is a float
        if isinstance(best_final_loss, np.ndarray):
            best_final_loss = best_final_loss.item()
    
        #
        # print(f"loss_value: {loss_value}, best_final_loss: {best_final_loss}")
        # print(f"loss_value type: {type(loss_value)}, best_final_loss type: {type(best_final_loss)}")

        if loss_value < best_final_loss:
            best_net = net
            best_final_loss = loss_value
            best_learning_curve = learning_curve

    # Return the best curve along with its final loss and learing curve
    return best_net, best_final_loss, best_learning_curve






# Sample data (replace with actual data)
X = X_combined_standard[:, :-1]
y = X_combined_standard[:, -1]

# Hyperparameter ranges
hidden_units = [1, 2, 3, 4, 5, 10]  # Example for ANN
lambdas = [0.01, 0.05, 0.1, 1, 10, 100]  # Example for Ridge regression

# Outer and inner CV folds
K1 = 10  # Outer CV
K2 = 10  # Inner CV

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

    for inner_train_idx, inner_val_idx in inner_cv.split(X_train):
        X_inner_train, X_val = X_train[inner_train_idx], X_train[inner_val_idx]
        y_inner_train, y_val = y_train[inner_train_idx], y_train[inner_val_idx]

        for h in hidden_units:
            ann_errors = []

            # Define the model structure
            model = lambda: torch.nn.Sequential(
                torch.nn.Linear(X_inner_train.shape[1], h),  # M features to H hidden units
                torch.nn.ReLU(),  # 1st transfer function
                torch.nn.Linear(h, 1),  # H hidden units to 1 output
                torch.nn.Sigmoid(),  # final transfer function
            )
            loss_fn = torch.nn.MSELoss()

            # Train the network
            net, _, _ = train_neural_netNOPRINT(
                model,
                loss_fn,
                X=torch.tensor(X_inner_train, dtype=torch.float),
                y=torch.tensor(y_inner_train.reshape(-1, 1), dtype=torch.float),  # Ensure y has the same shape as the input
                n_replicates=3,
                max_iter=10000,
            )

            # Predict and calculate error
            y_pred_ann = net(torch.tensor(X_val, dtype=torch.float)).data.numpy()
            ann_errors.append(mean_squared_error(y_val.reshape(-1, 1), y_pred_ann))  # Ensure y_val has the same shape as the prediction

            mean_ann_error = np.mean(ann_errors)
            if mean_ann_error < best_ann_error:
                best_ann_error = mean_ann_error
                best_h = h
            

        for lmbd in lambdas:
            lr_errors = []

            # Train Ridge Regression
            lambdaI = lmbd * np.eye(X_inner_train.shape[1])
            lambdaI[0, 0] = 0  # Do not regularize the bias term
            w_rlr = np.linalg.solve(X_inner_train.T @ X_inner_train + lambdaI, X_inner_train.T @ y_inner_train).squeeze()
            y_pred_lr = X_val @ w_rlr
            lr_errors.append(mean_squared_error(y_val, y_pred_lr))

            mean_lr_error = np.mean(lr_errors)
            if mean_lr_error < best_lr_error:
                best_lr_error = mean_lr_error
                best_lambda = lmbd

    # Train with best hyperparameters on full training set
    ann_best = MLPRegressor(hidden_layer_sizes=(best_h,), max_iter=1000, random_state=42)
    ann_best.fit(X_train, y_train)
    ann_test_error = mean_squared_error(y_test, ann_best.predict(X_test))

    lambdaI = best_lambda * np.eye(X_train.shape[1])
    lambdaI[0, 0] = 0  # Do not regularize the bias term
    w_rlr = np.linalg.solve(X_train.T @ X_train + lambdaI, X_train.T @ y_train).squeeze()
    lr_test_error = mean_squared_error(y_test, X_test @ w_rlr)

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

    print(f"Fold {i + 1} completed")

# Print results table
print("Fold | Best h | ANN Test Error | Best λ | Linear Regression Test Error | Baseline Test Error")
for res in results:
    print(f"{res['fold']} | {res['best_h']} | {res['ann_test_error']:.2f} | {res['best_lambda']} | {res['lr_test_error']:.2f} | {res['baseline_test_error']:.2f}")