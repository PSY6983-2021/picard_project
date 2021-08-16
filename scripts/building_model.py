
#import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import scipy.stats as stats
from random import seed
from random import randint
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_predict, cross_val_score, train_test_split, GroupShuffleSplit, permutation_test_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, explained_variance_score


def split_data(X,Y,group,procedure):
    """
    Split the data according to the group parameters
    to ensure that the train and test sets are completely independent

    Parameters
    ----------
    X: predictive variable
    Y: predicted variable
    group: group labels used for splitting the dataset
    procedure: strategy to split the data

    Returns
    ----------
    X_train: train set containing the predictive variable
    X_test: test set containing the predictive variable
    y_train: train set containing the predicted variable
    y_test: test set containing the predicted variable
    """
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for train_idx, test_idx in procedure.split(X, Y, group):
        X_train.append(X[train_idx])
        X_test.append(X[test_idx])
        y_train.append(Y[train_idx])
        y_test.append(Y[test_idx])
    
    return X_train, X_test, y_train, y_test


def verbose(splits, X_train, X_test, y_train, y_test, X_verbose = True, y_verbose = True):
    """
    Print the mean and the standard deviation of the train and test sets
   
    Parameters
    ----------
    splits: number of splits used for the cross-validation
    X_train: train set containing the predictive variable
    X_test: test set containing the predictive variable
    y_train: train set containing the predicted variable
    y_test: test set containing the predicted variable
    X_verbose (boolean): if X_verbose == True, print the descriptive stats for the X (train and test)
    y_verbose (boolean): if y_verbose == True, print the descriptive stats for the y (train and test)
    """
    for i in range(splits):
        if X_verbose:
            print(i,'X_Train: \n   Mean +/- std = ', X_train[i][:][:].mean(),'+/-', X_train[i][:][:].std())
            print(i,'X_Test: \n   Mean +/- std = ', X_test[i][:][:].mean(),'+/-', X_test[i][:][:].std())
        if y_verbose:
            print(i,'y_Train: \n   Mean +/- std = ', y_train[i][:].mean(),'+/-', y_train[i][:].std(), '\n   Skew = ', stats.skew(y_train[i][:]), '\n   Kurt = ', stats.kurtosis(y_train[i][:]))
            print(i,'y_Test: \n   Mean +/- std = ', y_test[i][:].mean(),'+/-', y_test[i][:].std(), '\n   Skew = ', stats.skew(y_test[i][:]), '\n   Kurt = ', stats.kurtosis(y_test[i][:]))
        print('\n')


def compute_metrics(y_test, y_pred, df, fold, print_verbose): 
    """
    Compute different metrics and print them

    Parameters
    ----------
    y_test: ground truth
    y_pred: predicted values
    df: dataFrame containing the result of the metrics
    fold: cross-validation fold for which the metrics are computed
    
    Returns
    ----------
    df_metrics: dataFrame containing the different metrics
    """  
    r2 = round(r2_score(y_test, y_pred),2)
    mae = round(mean_absolute_error(y_test, y_pred),2)
    mse = round(mean_squared_error(y_test, y_pred), 2)
    rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 2)
    
    df.loc[fold] = [r2, mae, mse, rmse]
    if print_verbose:
        print('------Metrics for fold {}------'.format(fold))
        print('R2 value = {}'.format(r2))
        print('MAE value = {}'.format(mae))
        print('MSE value = {}'.format(mse))
        print('RMSE value = {}'.format(rmse))
        print('\n')

    return df


def LASSO_PCR(n_component):
    """
    Parameters
    ----------
    n_component: number of components to keep in the PCA

    Returns
    ----------
    pipe: pipeline to apply PCA and Lasso regression sequentially
    """
    estimators = [('reduce_dim', PCA(n_component)), ('clf', Lasso())] 
    pipe = Pipeline(estimators)
    return pipe


def train_test_model(X, y, gr, splits=5,test_size=0.3, n_components=0.80, random_seed=42, print_verbose=True):

    """
    Build and evaluate the LASSO-PCR model
    First compute the PCA and then fit the LASSO regression on the PCs scores

    Parameters
    ----------
    X: predictive variable
    y: predicted variable
    gr: grouping variable
    splits: number of split for the cross-validation 
    test_size: percentage of the data in the test set
    n_components: number of components to keep for the PCA
    random_seed: controls the randomness of the train/test splits. 
    print_verbose: either or not the verbose is printed

    Returns
    ----------
    X_train: list containing the training sets of the predictive variable
    y_train: list containing the training sets of the predictive variable
    X_test: list containing the training sets of the predictive variable
    y_test: list containing the training sets of the predictive variable
    y_pred: list containing the predicted values for each fold
    model_voxel: list of arrays containing the coefficients of the model in the voxel space 
    df_metrics: DataFrame containing different metrics for each fold
    """ 
    #Initialize the variables
    y_pred = []
    model = []
    #model_voxel = []
    #intercept = []
    df_metrics = pd.DataFrame(columns=["r2", "mae", "mse", "rmse"])

    #Strategy to split the data
    group_Shuffle = GroupShuffleSplit(n_splits = splits, test_size = test_size, random_state = random_seed)

    #Split the data     
    X_train, X_test, y_train, y_test = split_data(X, y, gr, group_Shuffle)
    if print_verbose:
        verbose(splits, X_train, X_test, y_train, y_test, X_verbose = True, y_verbose = True)

    for i in range(splits):

        ###Build and test the model###
        print("----------------------------")
        print("Training model")
        model_lasso_pcr = LASSO_PCR(n_components)
        model.append(model_lasso_pcr.fit(X_train[i], y_train[i]))
        #intercept.append(model[i][1].intercept_) 
        y_pred.append(model[i].predict(X_test[i]))
        ###Scores###
        df_metrics = compute_metrics(y_test[i], y_pred[i], df_metrics, i, print_verbose)

        #model_voxel.append(model[i][0].inverse_transform(model[i][1].coef_))

    df_metrics.to_csv("dataframe_metrics.csv")

    return X_train, y_train, X_test, y_test, y_pred, model


def compute_permutation(X, y, gr, n_components=0.80, n_permutations=5000, scoring="r2", random_seed=42):
    """
    Compute the permutation test for a specified metric (r2 by default)
    Apply the PCA after the splitting procedure

    Parameters
    ----------
    X: predictive variable
    y: predicted variable
    gr: grouping variable
    n_components: number of components to keep for the PCA
    n_permutations: number of permuted iteration
    scoring: scoring strategy

    Returns
    ----------
    score (float): true score
    perm_scores (ndarray): scores for each permuted samples
    pvalue (float): probability that the true score can be obtained by chance

    See also scikit-learn permutation_test_score documentation
    """
    cv = GroupShuffleSplit(n_splits = 5, test_size = 0.3, random_state = random_seed)
    score, perm_scores, pvalue = permutation_test_score(estimator=LASSO_PCR(n_components), X=X, y=y, groups= gr, scoring=scoring, cv=cv, n_permutations=n_permutations, random_state=42, n_jobs=-1)
    
    return score, perm_scores, pvalue


def boostrap_test(X, y, gr, splits=5, test_size=0.30, n_components=0.80, n_resampling=5000, random_seed=42):
    """
    Parameters
    ----------
    X: predictive variable
    y: predicted variable
    gr: grouping variable
    splits: number of split for the cross-validation 
    test_size: percentage of the data in the test set
    n_components: number of components to keep for the PCA
    n_resampling: number of samples with replacement

    Returns
    ----------
    resampling_coef: list of arrays containing the coefficients for each voxel for each resampling
    """
    resampling_coef = []
    #seed random generator
    seed(random_seed)

    for i in range(n_resampling):
        value_resampled = []
        #Create new sample
        for _ in range(len(X)):
	        value_resampled.append(randint(0, len(X)-1))
        X_resampled = X[value_resampled,:]
        y_resampled = y[value_resampled]
        gr_resampled = gr[value_resampled]
	         
        _, _, _, _, _, model = train_test_model(X_resampled, y_resampled, gr_resampled, splits=splits,test_size=test_size, n_components=n_components, random_seed=random_seed, print_verbose=False)
        resampling_coef.extend(model)
    
    #Need to add the calculation of the p-value
    
    return resampling_coef