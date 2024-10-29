import numpy as np
from sklearn import linear_model, model_selection


def glm_validate(X, y, cvf=10):
    """Validate linear regression model using 'cvf'-fold cross validation.
    The loss function computed as mean squared error on validation set (MSE).
    Function returns MSE averaged over 'cvf' folds.

    Parameters:
    X       training data set
    y       vector of values
    cvf     number of crossvalidation folds
    """
    y = y.squeeze()
    CV = model_selection.KFold(n_splits=cvf, shuffle=True)
    validation_error = np.empty(cvf)
    f = 0
    for train_index, test_index in CV.split(X):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        m = linear_model.LinearRegression(fit_intercept=True).fit(X_train, y_train)
        validation_error[f] = (
            np.square(y_test - m.predict(X_test)).sum() / y_test.shape[0]
        )
        f = f + 1
    return validation_error.mean()


def feature_selector_lr(
    X, y, cvf=10, features_record=None, loss_record=None, display=""
):
    """Function performs feature selection for linear regression model using
    'cvf'-fold cross validation. The process starts with empty set of
    features, and in every recurrent step one feature is added to the set
    (the feature that minimized loss function in cross-validation.)

    Parameters:
    X       training data set
    y       vector of values
    cvf     number of crossvalidation folds

    Returns:
    selected_features   indices of optimal set of features
    features_record     boolean matrix where columns correspond to features
                        selected in subsequent steps
    loss_record         vector with cv errors in subsequent steps

    Example:
    selected_features, features_record, loss_record = ...
        feature_selector_lr(X_train, y_train, cvf=10)

    """
    y = y.squeeze()  # ÆNDRING JLH #9/3
    # first iteration error corresponds to no-feature estimator
    if loss_record is None:
        loss_record = np.array([np.square(y - y.mean()).sum() / y.shape[0]])
    if features_record is None:
        features_record = np.zeros((X.shape[1], 1))

    # Add one feature at a time to find the most significant one.
    # Include only features not added before.
    selected_features = features_record[:, -1].nonzero()[0]
    min_loss = loss_record[-1]
    if display == "verbose":
        print(min_loss)
    best_feature = False
    for feature in range(0, X.shape[1]):
        if np.where(selected_features == feature)[0].size == 0:
            trial_selected = np.concatenate(
                (selected_features, np.array([feature])), 0
            ).astype(int)
            # validate selected features with linear regression and cross-validation:
            trial_loss = glm_validate(X[:, trial_selected], y, cvf)
            if display == "verbose":
                print(trial_loss)
            if trial_loss < min_loss:
                min_loss = trial_loss
                best_feature = feature

    # If adding extra feature decreased the loss function, update records
    # and go to the next recursive step
    if best_feature is not False:
        features_record = np.concatenate(
            (features_record, np.array([features_record[:, -1]]).T), 1
        )
        features_record[best_feature, -1] = 1
        loss_record = np.concatenate((loss_record, np.array([min_loss])), 0)
        selected_features, features_record, loss_record = feature_selector_lr(
            X, y, cvf, features_record, loss_record
        )

    # Return current records and terminate procedure
    return selected_features, features_record, loss_record


def rlr_validate(X, y, lambdas, cvf=10):
    """Validate regularized linear regression model using 'cvf'-fold cross validation.
    Find the optimal lambda (minimizing validation error) from 'lambdas' list.
    The loss function computed as mean squared error on validation set (MSE).
    Function returns: MSE averaged over 'cvf' folds, optimal value of lambda,
    average weight values for all lambdas, MSE train&validation errors for all lambdas.
    The cross validation splits are standardized based on the mean and standard
    deviation of the training set when estimating the regularization strength.

    Parameters:
    X       training data set
    y       vector of values
    lambdas vector of lambda values to be validated
    cvf     number of crossvalidation folds

    Returns:
    opt_val_err         validation error for optimum lambda
    opt_lambda          value of optimal lambda
    mean_w_vs_lambda    weights as function of lambda (matrix)
    train_err_vs_lambda train error as function of lambda (vector)
    test_err_vs_lambda  test error as function of lambda (vector)
    """
    CV = model_selection.KFold(cvf, shuffle=True)
    M = X.shape[1]
    w = np.empty((M, cvf, len(lambdas)))
    train_error = np.empty((cvf, len(lambdas)))
    test_error = np.empty((cvf, len(lambdas)))
    f = 0
    y = y.squeeze()
    for train_index, test_index in CV.split(X, y):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]

        # Standardize the training and set set based on training set moments
        mu = np.mean(X_train[:, 1:], 0)
        sigma = np.std(X_train[:, 1:], 0)

        X_train[:, 1:] = (X_train[:, 1:] - mu) / sigma
        X_test[:, 1:] = (X_test[:, 1:] - mu) / sigma

        # precompute terms
        Xty = X_train.T @ y_train
        XtX = X_train.T @ X_train
        for l in range(0, len(lambdas)):
            # Compute parameters for current value of lambda and current CV fold
            # note: "linalg.lstsq(a,b)" is substitue for Matlab's left division operator "\"
            lambdaI = lambdas[l] * np.eye(M)
            lambdaI[0, 0] = 0  # remove bias regularization
            w[:, f, l] = np.linalg.solve(XtX + lambdaI, Xty).squeeze()
            # Evaluate training and test performance
            train_error[f, l] = np.power(y_train - X_train @ w[:, f, l].T, 2).mean(
                axis=0
            )
            test_error[f, l] = np.power(y_test - X_test @ w[:, f, l].T, 2).mean(axis=0)

        f = f + 1

    opt_val_err = np.min(np.mean(test_error, axis=0))
    opt_lambda = lambdas[np.argmin(np.mean(test_error, axis=0))]
    train_err_vs_lambda = np.mean(train_error, axis=0)
    test_err_vs_lambda = np.mean(test_error, axis=0)
    mean_w_vs_lambda = np.squeeze(np.mean(w, axis=1))

    return (
        opt_val_err,
        opt_lambda,
        mean_w_vs_lambda,
        train_err_vs_lambda,
        test_err_vs_lambda,
    )
