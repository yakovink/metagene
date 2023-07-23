#!pip install GEOparse pyod
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 18:31:10 2023

"""

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
import GEOparse
import json
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, make_scorer
from pyod.models import kde as KernelDensity
from sklearn.utils import resample
from copy import copy
from sklearn.model_selection import train_test_split
from sklearn.decomposition import MiniBatchSparsePCA, PCA
from joblib import Memory
import os
from time import time

if not os.path.exists("cachedir"):
    os.makedirs("cachedir")

memory = Memory(location='cachedir', verbose=0)


def loadData():
    """
    This function loads the data from an excel file that contains references to GSM files.
    It then organizes the loaded data into healthy and sick datasets by disease and saves the organized data into a JSON file.
    """
    
    # Load the excel file containing GSM file references
    summary = pd.read_excel("summary.xlsx", header=2)
    results = {}

    # Iterate over each row in the excel file
    for i in range(135):
        
        # Get the current row
        sample = summary.iloc[i]

        # Get the GSM references for the sick and healthy samples
        sick_samples = sample['HDisease'].split(',')
        healthy_samples = sample['HControl'].split(',')

        # Process the sick samples
        for sick in sick_samples:
            # If the sample is not already in the results, load the GSM file and add it
            if sick not in results.keys():
                geodata = GEOparse.get_GEO(filepath=f"samples/{sick}.txt", geotype="GSM")
                results[sick] = {geodata.table.iloc[j]['ID_REF']: geodata.table.iloc[j]['VALUE'] for j in range(len(geodata.table))}
                results[sick]['sickness'] = {}
            # Add the disease name to the sample
            results[sick]['sickness'][sample['DiseaseName']] = 1

        # Process the healthy samples
        for healthy in healthy_samples:
            # If the sample is not already in the results, load the GSM file and add it
            if healthy not in results.keys():
                geodata = GEOparse.get_GEO(filepath=f"samples/{healthy}.txt", geotype="GSM")
                results[healthy] = {geodata.table.iloc[j]['ID_REF']: geodata.table.iloc[j]['VALUE'] for j in range(len(geodata.table))}
                results[healthy]['sickness'] = {}
            # Add the disease name to the sample
            results[healthy]['sickness'][sample['DiseaseName']] = 0

    # Save the dictionary to a JSON file
    with open('datafile.json', 'w') as f:
        json.dump(results, f)
        

        
def balance_accuracy(estimator,X, y_true):
    y_pred=estimator.predict(X)
    pos_true=y_true[y_true==0]
    neg_true=y_true[y_true==1]
      
    pos_pred=y_pred[y_true==0]
    neg_pred=y_pred[y_true==1]
    
    acc_pos=np.sum(np.int64(pos_pred==pos_true))/(len(pos_true))
    acc_neg=np.sum(np.int64(neg_pred==neg_true))/(len(neg_true))
    
    return 0.5*acc_pos+0.5*acc_neg

        
def evaluate_datasets(datasets, cv=5):
    """
    This function takes in datasets and performs various machine learning methods on them.
    It then saves the results in a JSON file.
    """
    # Convert datasets into DataFrame
    datasets = {k: pd.DataFrame(v).T for k,v in datasets.items()}  
    #balance_acc = make_scorer(balance_accuracy, greater_is_better=True)          

    for desease in datasets.keys():
        # Check if results already exist
        if os.path.exists('results.json'):
            with open('results.json', 'r') as f:
                results = json.load(f)
                if desease in results:
                    print(f"{desease} already saved, continue...")
                    continue

        print(desease)

        # Split the dataset into features and target
        X, y = splitXY(datasets[desease])
        print(X.shape)

        # Prepare the data for analysis
        X_train, X_test, y_train, y_test = prepare_data_for_analysis(X, y)
        mincv = min(cv, np.min(np.bincount(y_train)))
        print(X_train.shape)

        # Define the tuning parameters for the classifiers
        TreeInterRange = range(1, len(X_train)-1)
        datasets[desease] = {"data": datasets[desease], "results": {}}

        # Define the classifiers to be used
        methods = {
            "lda": {"method": LinearDiscriminantAnalysis(), "tun": {"solver": ["svd"]}},
            "qda": {"method": QuadraticDiscriminantAnalysis(), "tun": {"reg_param":[0.25,0.5,0.75]}},
            "knn": {"method": KNeighborsClassifier(), "tun": {"n_neighbors": range(1, len(X_train)-1)}},
            "logistic_regression": {"method": LogisticRegression(), "tun": {'penalty': ['none', "l1", "l2"], 'solver': ['saga']}},
            "decision_tree": {"method": DecisionTreeClassifier(), "tun": {"max_depth": TreeInterRange, "min_samples_split": TreeInterRange, "min_samples_leaf": TreeInterRange}},
            "random_forest": {"method": RandomForestClassifier(), "tun": {"n_estimators": [10, 50, 100, 200], "max_depth": TreeInterRange, "min_samples_split": TreeInterRange, "min_samples_leaf": TreeInterRange}},
            "svm": {"method": SVC(), "tun": {"gamma": ["scale", "auto"], "C": [10**i for i in range(-5, 5)], "kernel": ["linear", "rbf", "poly"]}},
            "mlp": {"method": MLPClassifier(), "tun": {"hidden_layer_sizes": [(50,), (100,)], "alpha": [10**i for i in range(-5, -1)]}}
        }

        # Apply each method to the dataset
        for m in methods.keys():
            print(f"testing {m}")

            # Perform randomized search cross validation
            grid_search = RandomizedSearchCV(methods[m]["method"], param_distributions=methods[m]["tun"], n_iter=10000, cv=mincv, scoring=balance_accuracy, n_jobs=-1, return_train_score=True, verbose=3, random_state=42)
            
            # Fit the model and get the best parameters
            datasets[desease]["results"][m] = fit_and_get_best(grid_search, X_train, X_test, y_train, y_test)

        # Save results after each disease
        if not os.path.exists('../program2/results.json'):
            results = {}  # Initialize if results.json does not exist
            
        results[desease] = datasets[desease]["results"]
        
        # Save the results to a JSON file
        with open('../program2/results.json', 'w') as f:
            json.dump(results, f)
    
    return datasets


def fromJson():
    """Function to load and organize data from JSON file"""
    
    # Load data from json file
    with open('datafile.json') as f:
        obs = json.load(f)
    
    # Initialize dictionary to hold samples
    samps = {}

    # Iterate over all keys (observation IDs) in the loaded data
    for ob in obs.keys():
        
        # Get all diseases for the current observation
        diseases = obs[ob]['sickness']
        
        # Iterate over each disease
        for disease in diseases.keys():
            
            # Define rules for splitting datasets based on disease and presence of specific keys
            split_rules = {
                "Huntington": [("Huntington1", "1007_s_at"), ("Huntington2", None)],
                "Lupus": [("Lupus1", "244891_x_at"), ("Lupus3", "222382_x_at"), ("Lupus2", None)],
                "Duchenne Muscular Dystrophy (DMD)": [("Duchenne Muscular Dystrophy (DMD)1", "212893_at"), 
                                                      ("Duchenne Muscular Dystrophy (DMD)3", "1007_s_at"), 
                                                      ("Duchenne Muscular Dystrophy (DMD)4", "92117_at"), 
                                                      ("Duchenne Muscular Dystrophy (DMD)5", "54170_at"), 
                                                      ("Duchenne Muscular Dystrophy (DMD)2", None)],
                "Inflammatory Bowel Diseases (IBD)": [("Inflammatory Bowel Diseases (IBD)1", "AFFX-MurIL2_at"), 
                                                      ("Inflammatory Bowel Diseases (IBD)2", None)],
                "Sepsis": [("Sepsis3", "AFFX-TrpnX-5_at"), ("Sepsis2", None)],
            }
            
            # Exclude these diseases
            excluded_diseases = ["Cystic Fibrosis"]
            
            # Split datasets based on rules defined above
            if disease in split_rules:
                for subset, key in split_rules[disease]:
                    if key is None or key in obs[ob].keys():
                        if subset not in samps:
                            samps[subset] = {}
                        # Check if the feature '1090736.0' exists in the data for 'Sepsis', if so, skip this data point
                        if disease == "Sepsis" and "1090736.0" in obs[ob].keys():
                            continue
                        samps[subset][ob] = copy(obs[ob])
                        samps[subset][ob]['sickness'] = copy(obs[ob]['sickness'][disease])
                        break
            
            # Skip excluded diseases
            elif disease in excluded_diseases:
                continue
            
            # For other diseases, no splitting is needed
            else:       
                if disease not in samps:
                    samps[disease] = {}
                samps[disease][ob] = copy(obs[ob])
                samps[disease][ob]['sickness'] = copy(obs[ob]['sickness'][disease])
    
    return samps



# This function separates the features (X) from the target (y) in the given dataset
def splitXY(dataset):
    y = dataset['sickness']  # Target variable
    X = dataset.drop(columns=['sickness'])  # Features
    return X, y

# This function scales the features in the dataset using StandardScaler
def scale(X_train,X_test):
    cols = X_train.columns
    train_inds = X_train.index
    test_inds=X_test.index
    
    scaler = StandardScaler().fit(X_train) # Initialize the scaler
    X_train = pd.DataFrame(scaler.transform(X_train))  # Scale the features
    X_test = pd.DataFrame(scaler.transform(X_test))  # Scale the features

    X_train.columns = cols
    X_train.index = train_inds
    
    X_test.columns = cols
    X_test.index = test_inds
    
    return X_train,X_test

# This function detects outliers in the dataset using Kernel Density Estimation
def detect_outliers_kernel(X, y):
    outlier_indices = {}
    for i in X.columns:
        X_feature = np.array(X[i]).reshape(-1,1)
        kde = KernelDensity.KDE(contamination=0.1)
        kde.fit(X_feature)
    
        outlier_mask = kde.predict(X_feature)
        outliers = np.argwhere(outlier_mask).flatten()
        for idx in outliers:
            outlier_type = "low" if X_feature[idx] < np.median(X_feature) else "high"
            if y[0] != None:
                y_condition = "only_y_1" if y[idx] == 1 and not any((outlier_mask & (y == 0))) else "also_y_0"
                outlier_indices[idx] = {"feature": i, "type": outlier_type, "y_condition": y_condition}
            else:
                outlier_indices[idx] = {"feature": i, "type": outlier_type}
    return outlier_indices

# This function prepares the data for analysis by performing several data preprocessing steps
def prepare_data_for_analysis(X, y, explained_variance=0.999):
    x = X.shape[1]
    X = X.dropna(axis=1)  # Remove features that contain NaN values
    print(X.shape[1]-x ,"observations contains NaN and was removed")

    print("order features")
    # Order features based on their absolute correlation with the target variable
    correlations = X.corrwith(y).abs()
    sorted_features = correlations.sort_values(ascending=False).index
    X = X[sorted_features]

    print("remove outliers")
    # Remove outliers detected by Kernel Density Estimation
    outlier_indices = detect_outliers_kernel(X, y)
    params_to_drop = [info['feature'] for idx, info in outlier_indices.items() if info['y_condition'] == 'also_y_0']
    X_new = X.drop(columns=params_to_drop)


    print("split the data")
    # Split the data into training and testing sets
    
    

    
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=int(time()),stratify=y)
    print(len(X_test[y_test==0]))
    
    
    print("scale the data")
    X_train,X_test = scale(X_train,X_test)  # Scale the features
        
    print("remove unused data for var explain")
    # Perform PCA for dimensionality reduction
    pca = PCA().fit(X_train)
    X_train=pca.transform(X_train)
    X_test=pca.transform(X_test)


    
    
    return X_train, X_test, y_train, y_test



@memory.cache
def fit_and_get_best(grid_search, X_train, X_test, y_train, y_test):
    try:
        # Fit the grid search on the training data
        grid_search.fit(X_train, y_train)
        
        # Get the index of the best model in the grid search results
        best_index = grid_search.best_index_
        
        # Get the mean fit time, train accuracy, and best estimator from the grid search results
        rt = grid_search.cv_results_['mean_fit_time'][best_index]
        acc = grid_search.cv_results_['mean_train_score'][best_index]
        best_estimator = grid_search.best_estimator_
        
        # Predict the target variable using the best estimator
        
        y_pred_pos=best_estimator.predict(X_test[y_test==1])
        y_pred_neg=best_estimator.predict(X_test[y_test==0])
        #y_pred = best_estimator.predict(X_test)
        
        # Calculate the test accuracy
        test_acc_pos = accuracy_score(y_test[y_test==1], y_pred_pos)
        test_acc_neg = accuracy_score(y_test[y_test==0], y_pred_neg)
        # Print and return the results
        results = {
            "runningTime": rt,
            "pos_accuracy": test_acc_pos,
            "neg_accuracy": test_acc_neg,
            "train_accuracy":acc,
            "best_params": grid_search.best_params_
        }
    except ValueError:
        results = {
            "runningTime": 0,
            "pos_accuracy": 0,
            "neg_accuracy": 0,
            "train_accuracy":0,
            "best_params": {"reg_param":0}
        }
        
        

    print(results)
    return results

loadData()
datasets=fromJson()
datasets=evaluate_datasets(datasets, cv=5)










