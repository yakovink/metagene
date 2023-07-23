# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 09:49:50 2023

@author: yakov
"""


import tensorflow as tf
from tensorflow import keras

from keras import models,layers,regularizers,callbacks

#import models.Model as Model
#from models import Model
#from layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate, Dropout
#from callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

from scipy.stats import shapiro

import numpy as np
from copy import copy
from PIL import Image
import pandas as pd
import json
from time import time
import matplotlib.pyplot as plt
import warnings


warnings.filterwarnings('ignore')

def getGenes(datasets):
    # Function to translate gene names in datasets
    print("translate genes")
    count = 0  # Initialize a counter variable
    # Read the JSON file containing gene translations
    with open('program2/translate.json', 'r') as file:
        rename_json = file.read()
    trans = json.loads(rename_json)  # Load gene translations into a dictionary
    count = 0  # Reset the counter
    # Iterate over each dataset
    for des,data in datasets.items():
        # Identify common gene names between dataset columns and translation dictionary keys
        genes_to_translate = list(set(data.columns) & set(trans.keys()))
        local_translator = {k: trans[k] for k in genes_to_translate}  # Create a local translation dictionary
        # Rename columns in the dataset using the local translation dictionary
        datasets[des].rename(columns=local_translator, inplace=True)
        count += len(genes_to_translate)  # Update the counter
    print(f"{count} gene names translated!")

    gens = np.array(datasets["Adenocarcinoma"].keys())  # Get initial gene names from a specific dataset
    # Concatenate gene names from all datasets
    for des, data in datasets.items():
        datagens = np.array(data.keys())
        gens = np.unique(np.concatenate((gens, datagens), 0)) 
    return gens  # Return the unique gene names from all datasets


def loadData():
    # Load data from json file
    with open('program1/datafile.json') as f:
        obs = json.load(f)
    # Initialize dictionary to hold samples
    samps = {}
    # Iterate over all keys (obskernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)ervation IDs) in the loaded data
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
    samps={k:pd.DataFrame(v).T for k,v in samps.items()}
    return samps





def loadResults():
    # Load results from a JSON file
    with open('program2/results.json') as f:
        res = json.load(f)
    results = {}  # Initialize a dictionary to store results
    tunpar = {}  # Initialize a dictionary to store tuning parameters
    # Iterate over each dataset
    for des in res.keys():
        results[des] = {}  # Initialize a dictionary for dataset results
        tunpar[des] = {}  # Initialize a dictionary for dataset tuning parameters
        # Iterate over each method in the dataset
        for method in res[des].keys():
            desmethod=res[des][method]
            # Extract and store relevant results and parameters for each method
            results[des][method + "RT"] = desmethod["runningTime"]
            results[des][method + "NEGACC"] = desmethod["neg_accuracy"]
            results[des][method + "POSACC"] = desmethod["pos_accuracy"]
            results[des][method + "OF"] = desmethod["train_accuracy"] - (desmethod["neg_accuracy"] + desmethod["pos_accuracy"]) * 0.5
            tunpar[des][method] = desmethod["best_params"]
    # Function to flatten the tuning parameter dictionary
    def flat(dict):
        tun_dict = {}
        for k1 in dict.keys():
            for k2 in dict[k1].keys():
                tun_dict[k1 + "_" + k2] = dict[k1][k2]
        return tun_dict
    # Flatten the tuning parameter dictionary
    dict_tunpar = {k: flat(v) for k, v in tunpar.items()}
    # Create a table for tuning parameters
    tunpar_table = pd.DataFrame(dict_tunpar).T
    # Create a modified version of the tuning parameter table
    tunpar_table_l = pd.DataFrame({
        "logistic_regression_lasso": np.int64(tunpar_table["logistic_regression_penalty"] == "l1"),
        "logistic_regression_ridge": np.int64(tunpar_table["logistic_regression_penalty"] == "l2"),
        "svm_kernel_rbf": np.int64(tunpar_table["svm_kernel"] == "rbf"),
        "svm_kernel_poly": np.int64(tunpar_table["svm_kernel"] == "poly"),
        "svm_gamma_scale": np.int64(tunpar_table["svm_gamma"] == "scale")
    })
    tunpar_table_l.index = tunpar_table.index
    # Modify and format specific columns in the tuning parameter table
    tunpar_table["mlp_hidden_layer_sizes"] = np.array([tunpar_table["mlp_hidden_layer_sizes"][i][0] for i in range(31)])
    tunpar_table = tunpar_table.drop(columns=["lda_solver", "logistic_regression_solver", "logistic_regression_penalty", "svm_kernel", "svm_gamma"]).astype(float)
    tunpar_table_l["svm_gamma_scale"] = 1.0 - tunpar_table_l["svm_gamma_scale"]
    tunpar_table_l = tunpar_table_l - 0.5
    # Return the results, tuning parameter tables, and modified tuning parameter table
    return pd.DataFrame(results).drop(columns=[]).T, tunpar_table, tunpar_table_l.astype(float)

'''
 The transform_tun function divides the size-related tuning parameters by the corresponding lengths and takes the logarithm (base 10) of the logarithmic tuning parameters.
'''

def transform_tun(tunpar_table, lengths):
    # Select columns for size-related tuning parameters
    size_tuns = tunpar_table.columns[list(range(1, 5)) + list(range(6, 9))]
    # Select columns for logarithmic tuning parameters
    log_tuns = tunpar_table.columns[[5, 9, 11]]
    # Divide size-related tuning parameters by lengths
    tunpar_table[size_tuns] = tunpar_table[size_tuns].divide(lengths, axis=0)
    # Take the logarithm (base 10) of logarithmic tuning parameters
    tunpar_table[log_tuns] = np.log10(tunpar_table[log_tuns])


'''
The detransform_tun function multiplies the size-related tuning parameters by the corresponding relevant lengths and rounds them to the nearest integer.
It raises 10 to the power of the logarithmic tuning parameters and rounds them to the nearest integer as well.
It then classifies specific columns based on predefined intervals or targets.
'''


def detransform_tun(tunpar_table, lengths):
    # Obtain relevant lengths based on the index of tunpar_table
    rel_lengths = lengths.loc[tunpar_table.index]
    # Select columns for size-related tuning parameters
    size_tuns = tunpar_table.columns[list(range(1, 5)) + list(range(6, 9))]
    # Select columns for logarithmic tuning parameters
    log_tuns = tunpar_table.columns[[9, 11]]
    # Multiply size-related tuning parameters by relevant lengths and round to nearest integer
    tunpar_table[size_tuns] = np.round(tunpar_table[size_tuns].multiply(rel_lengths, axis=0))
    # Raise 10 to the power of the logarithmic tuning parameters and round to nearest integer
    tunpar_table[log_tuns] = 10 ** np.round(tunpar_table[log_tuns])
    # Classify the qda_reg_param values using an interval of 0.25
    tunpar_table["qda_reg_param"] = classify(tunpar_table["qda_reg_param"].values, interval=0.25)
    # Classify the mlp_hidden_layer_sizes values using an interval of 50
    tunpar_table["mlp_hidden_layer_sizes"] = classify(tunpar_table["mlp_hidden_layer_sizes"].values, interval=50)
    # Classify the random_forest_n_estimators values using predefined targets
    tunpar_table["random_forest_n_estimators"] = 10**classify(tunpar_table["random_forest_n_estimators"].values, targets=np.log10(np.array([10, 50, 100, 200])).T)

'''
The order_by_distance function takes a dataset as input and recursively orders it based on the Euclidean distance from each row to the first and last rows.
If the dataset has only 1 or 2 rows, it is considered already ordered and returned as is.
The function then proceeds to sort the dataset based on the mean value of each row.
It then iterates over the remaining rows and calculates the squared Euclidean distance from each row to the first and last rows.
Based on the distance, each row is assigned to either the first or last group.
The function recursively applies the order_by_distance function to the first and last groups.
Finally, it concatenates the ordered first and last groups and returns the result.
'''


def order_by_distance(dataset):
    # Check if the dataset has 1 or 2 rows, in which case it is already ordered
    if dataset.shape[0] in [1, 2]:
        return dataset
    # Sort the dataset based on the mean value of each row
    sorted_indices = dataset.mean(axis=1).argsort()
    dataset = dataset[sorted_indices]
    # Initialize the first and last groups with the first and last rows of the dataset
    firstgroup = [0]
    lastgroup = [len(dataset) - 1]
    # Iterate over the remaining rows of the dataset
    for i in range(1, len(dataset) - 1):
        first = dataset[firstgroup[-1]]
        last = dataset[lastgroup[-1]]
        tested = dataset[i]
        # Calculate the squared Euclidean distance from the current row to the first and last rows
        distfromfirst = np.sum((tested - first) ** 2)
        distfromlast = np.sum((tested - last) ** 2)
        # Assign the current row to the first or last group based on the distance
        if distfromfirst >= distfromlast:
            lastgroup += [i]
        else:
            firstgroup += [i]
    # Recursively order the first and last groups
    firstgroup = order_by_distance(dataset[firstgroup])
    lastgroup = order_by_distance(dataset[lastgroup])
    # Concatenate the ordered first and last groups and return the result
    return np.concatenate([firstgroup, lastgroup])

'''
The actionOnPandas function takes a pandas DataFrame dataframe and an action function as input.
It applies the action function on the dataframe and stores the result in a new DataFrame called newData.
It then sets the index and column names of the newData DataFrame to match the original dataframe.
Finally, it returns the newData DataFrame.
'''
    

def actionOnPandas(dataframe, action):
    # Apply the specified action on the dataframe and store the result in a new DataFrame
    newData = pd.DataFrame(action(dataframe))
    # Set the index and column names of the new DataFrame to match the original dataframe
    newData.index = dataframe.index
    newData.columns = dataframe.columns
    # Return the new DataFrame
    return newData

    '''
The prepare_input function takes a dataset, a dictionary of sizes, and a list of genes as input.
It performs several data preprocessing steps on the dataset.
It first drops columns with missing values and separates the target variable.
It then standardizes the dataset using StandardScaler and calculates feature correlations with the target variable, sorting the features based on the correlations.
The function performs the Shapiro-Wilk test and applies PCA on the dataset.
It also calculates additional statistics and scales the dataset to the range of 0-255 using MinMaxScaler.

The function splits the dataset into positive and negative instances, visualizes the images, and orders the instances by distance.
It then creates image objects and resizes them based on the specified image size.
Finally, it concatenates the image arrays and returns the concatenated array along with the calculated statistics.
    '''



def prepare_input(dataset, sizes, gens):
    # Drop columns with missing values from the dataset
    dataset1 = dataset.dropna(axis=1)
    # Separate the target variable from the dataset
    y = dataset["sickness"]
    # Standardize the dataset
    scaler1 = StandardScaler()
    dataset1 = actionOnPandas(dataset1, scaler1.fit_transform)
    # Calculate feature correlations with the target variable and sort the features
    correlations = dataset1.corrwith(y).abs()
    sorted_features = correlations.sort_values(ascending=False).index
    dataset1 = dataset1[sorted_features]
    # Perform Shapiro-Wilk test on the dataset
    shap = shapiro(dataset1)
    # Apply PCA on the dataset
    pca = PCA()
    dataset1 = pca.fit_transform(dataset1)
    # Count the number of positive instances in the target variable
    pos = int(np.sum(y))
    print(f"positive: {pos}")
    # Calculate additional statistics
    stats1 = pd.Series(gens).isin(dataset.columns).astype(int)
    stats2 = pd.Series([dataset.shape[1], dataset.shape[0] - pos, pos, shap.pvalue])
    stats = stats2.append(stats1)
    # Scale the dataset to the range of 0-255
    scaler2 = MinMaxScaler(feature_range=(0, 255))
    dataset1 = scaler2.fit_transform(dataset1)
    # Split the dataset into positive and negative instances and visualize the images
    positive = dataset1[:pos]
    negative = dataset1[pos:]
    plt.imshow(dataset1, cmap="Blues")
    plt.show()
    
    positive = order_by_distance(positive)
    img1 = Image.fromarray(positive, 'L')
    img1 = img1.resize((sizes["img"], sizes["img"] // 2))
    img1_arr = np.array(img1).tolist()
    
    negative = order_by_distance(negative)
    img2 = Image.fromarray(negative, 'L')
    img2 = img2.resize((sizes["img"], sizes["img"] // 2))
    img2_arr = np.array(img2).tolist()
    
    img_arr = img1_arr + img2_arr
    plt.imshow(img_arr, cmap="Blues")
    plt.show()
    
    return img_arr, stats


'''
This function takes in y_true (true values) and y_pred (predicted values) as inputs and calculates the Mean Squared Error (MSE) with weighted components.
It first extracts the relevant values for each component by indexing the y_true and y_pred tensors.
It then applies clipping to ensure the predicted values are within a specific range.
The MSE is calculated for each component using tf.reduce_mean(tf.square()) to compute the squared differences between true and predicted values and take their mean.
Finally, the weighted MSE is computed by combining the individual MSE values with specified weights for each component.
'''


# Function to calculate the MSE (Mean Squared Error) with weighted components
def accuracy_mse(y_true, y_pred):
    # Define indices for different components
    neg_indices = tf.range(start=2, limit=33, delta=3)
    pos_indices = tf.range(start=1, limit=33, delta=3)
    rt_indices = tf.range(start=0, limit=33, delta=3)
    of_indices = tf.range(start=3, limit=33, delta=3)
    # Extract relevant values for negative component
    y_true_neg = tf.gather(y_true, neg_indices, axis=1)
    y_pred_neg = tf.gather(y_pred, neg_indices, axis=1)
    y_pred_neg = tf.clip_by_value(y_pred_neg, -4, 1)
    mse_neg = tf.reduce_mean(tf.square(y_true_neg - y_pred_neg))
    # Extract relevant values for positive component
    y_true_pos = tf.gather(y_true, pos_indices, axis=1)
    y_pred_pos = tf.gather(y_pred, pos_indices, axis=1)
    y_pred_pos = tf.clip_by_value(y_pred_pos, -4, 1)
    mse_pos = tf.reduce_mean(tf.square(y_true_pos - y_pred_pos))
    # Extract relevant values for running time component
    y_true_rt = tf.gather(y_true, rt_indices, axis=1)
    y_pred_rt = tf.gather(y_pred, rt_indices, axis=1)
    mse_rt = tf.reduce_mean(tf.square(y_true_rt - y_pred_rt))
    # Extract relevant values for overall fitness component
    y_true_of = tf.gather(y_true, of_indices, axis=1)
    y_pred_of = tf.gather(y_pred, of_indices, axis=1)
    mse_of = tf.reduce_mean(tf.square(y_true_of - y_pred_of))
    # Calculate weighted MSE
    return mse_neg * 0.47 + mse_pos * 0.47 + mse_of * 0.01 + mse_rt * 0.01

'''
This function takes in y_true (true values) and y_pred (predicted values) as inputs and calculates the tun_l_loss.
It first rounds the predicted values to the nearest integer using tf.round().
It then calculates the total number of positive and negative instances in the true values.
If there are positive instances present, it calculates the false positives (fp) and false negatives (fn) by comparing the true and predicted values.
The false positive rate and false negative rate are computed by dividing fp and fn by the total negative and total positive instances, respectively.
Finally, the tun_l_loss is calculated as a combination of the false positive rate and false negative rate.
If there are no positive instances, the function calculates the tun_l_loss based on the false positive rate only.
'''


# Function to calculate the tun_l_loss
def tun_l_loss(y_true, y_pred):
    y_pred_binary = tf.round(y_pred)
    total_positive = tf.reduce_sum(y_true)
    total_negative = tf.reduce_sum(1 - y_true)
    if total_positive > 0:
        # Calculate false positives and false negatives
        fp = tf.reduce_sum(tf.cast((y_true - y_pred_binary) < 0, tf.float32))
        fn = tf.reduce_sum(tf.cast((y_true - y_pred_binary) > 0, tf.float32))
        # Calculate false positive rate and false negative rate
        fp_rate = tf.divide(fp, total_negative)
        fn_rate = tf.divide(fn, total_positive)
        # Calculate tun_l_loss as a combination of false positive rate and false negative rate
        return 0.5 * fp_rate + 0.5 * fn_rate
    # Calculate tun_l_loss when there are no positive instances
    fp = tf.reduce_sum(tf.cast((y_true - y_pred_binary) < 0, tf.float32))
    return tf.divide(fp, total_negative)


'''
This function builds a neural network model for processing images and statistics data.
It iterates over the datasets, prepares the input, and stores the images and statistics in separate lists.
The function then defines the layers of the neural network, including convolutional layers for image processing, dense layers for combining the image and statistics inputs, and tunning layers for fine-tuning the model.
Finally, it creates the model, compiles it with specified loss functions, and returns the model along with the images and statistics lists.
'''


def build_model(datasets, sizes, gens):
    # Prepare lists to store images and statistics
    images = []
    stats = []
    # Iterate over the datasets
    for disease, values in datasets.items():
        # Prepare input and append to respective lists
        img, stat = prepare_input(values, sizes, gens)
        images.append(img)
        stats.append(stat)
    # Image input layer
    image_input = layers.Input(shape=(sizes["img"], sizes["img"], 1))
    # Convolutional layers for image processing
    img = layers.Conv2D(sizes["img"], (5, 5), activation='relu', kernel_regularizer=regularizers.l2(0.01))(image_input)
    img = layers.MaxPooling2D((2, 2))(img)
    img = layers.Conv2D(sizes["img"], (5, 5), activation='relu', kernel_regularizer=regularizers.l2(0.01))(img)
    img = layers.MaxPooling2D((2, 2))(img)
    img = layers.Conv2D(sizes["img"], (5, 5), activation='relu', kernel_regularizer=regularizers.l2(0.01))(img)
    img = layers.MaxPooling2D((2, 2))(img)
    img = layers.Conv2D(sizes["img"], (5, 5), activation='relu', kernel_regularizer=regularizers.l2(0.01))(img)
    img = layers.MaxPooling2D((2, 2))(img)
    img = layers.Conv2D(sizes["img"], (5, 5), activation='relu', kernel_regularizer=regularizers.l2(0.01))(img)
    img = layers.Flatten()(img)
    img = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(img)
    # Statistics input layer
    stat_input = layers.Input(shape=692647)
    stati = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(stat_input)
    # Combine image and statistics
    combined = layers.concatenate([img, stati])
    combined = layers.Dropout(0.5)(combined)
    # Tunning layers
    tunning = layers.Dense(12, activation='linear', kernel_regularizer=regularizers.l2(0.01))(combined)
    tunning_l = layers.Dense(5, activation='tanh', kernel_regularizer=regularizers.l2(0.01))(combined)
    # Combine the tunning layers with the previous layers
    combined2 = layers.concatenate([combined, tunning, tunning_l])
    combined2 = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(combined2)
    combined2 = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(combined2)
    combined2 = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))(combined2)
    # Performances output layer
    performances = layers.Dense(32, activation='linear', kernel_regularizer=regularizers.l2(0.01))(combined2)
    # Create the model
    model1 = models.Model(inputs=(image_input, stat_input), outputs=[performances, tunning, tunning_l])
    model1.compile(loss=[accuracy_mse, "mse", tun_l_loss], optimizer='adam')
    return model1, images, stats

'''
This function takes a pre-built model along with images, stats, results, tunning, and tunning_l as input.
It converts the images and stats into numpy arrays, defines an early stopping callback, and then fits the model using the input data.
It trains the model for a specified number of epochs with a batch size of 2, using a validation split of 0.2 for validation.
The fitting process also includes the early stopping callback to monitor the loss and stop training if no improvement is observed for a certain number of epochs.
The function returns the training history.
'''



def fit(model, images, stats, results, tunning, tunning_l):
    # Convert images and stats to numpy arrays
    images = np.array(images)
    stats = np.array(stats)
    # Define early stopping callback
    early_stop = callbacks.EarlyStopping(monitor='val_dense_7_loss', mode='min', verbose=1, patience=50)
    # Fit the model using the input data and callbacks
    return model.fit([images, stats], [results, tunning, tunning_l], epochs=1000, batch_size=2, validation_split=0.2, callbacks=[early_stop])

'''
This function is used to classify the output based on specific criteria.
The output is reshaped into a column vector.
If the interval parameter is provided, the output values are rounded to the nearest interval.
If targets is an array, the function finds the target value from the array that is closest to each output value.
The classified values are returned as an array.
'''



def classify(output, interval=0, targets=0):
    # Reshape the output to a column vector
    output = output.reshape(-1, 1)
    if interval != 0:
        # Round the output to the nearest interval
        return (np.round(output / interval) * interval).T[0]
    if isinstance(targets, np.ndarray):
        # Find the target value that is closest to each output value
        return targets[np.argmin(np.abs(output[:, np.newaxis] - targets[np.newaxis, :]),axis=1)]


'''
This function takes in new data, a trained model, a list of gene names (gens), and the size parameter.
It prepares the input by extracting image and statistical features from the new data using the prepare_input function.
Then, it predicts the results using the provided model.
The function returns a list containing the predicted results, tuning results, and tuning results with their respective method names.
'''



def predictMethod(newData, model, gens, size):
    # Prepare input features from new data
    img, stat = prepare_input(newData, size, gens)
    img = np.array([img])
    stat = np.array([stat])
    # Make predictions using the provided model
    pred = model.predict([img, stat])
    # Extract the predicted results, tuning results, and tuning results with method names
    res = pred[0][0]              # Predicted results
    tun_results = pred[1][0]      # Tuning results
    tun_results_l = pred[2][0]    # Tuning results with method names
    # Return the results as a list
    return [res, tun_results, tun_results_l]

'''
This function compares the cross-validation accuracy and the predicted accuracy for different methods.
It takes in a prediction_dict (containing predicted results) and a result_dict_test (containing cross-validation results).
The param parameter specifies the metric to compare (e.g., "neg_accuracy", "pos_accuracy").
The function iterates over the disease names in the prediction dictionary.
For each disease, it creates a DataFrame accuracy_comparison with columns for cross-validation accuracy and predicted accuracy for each method.
It then saves this DataFrame to a CSV file with the disease name and the specified parameter.
The function also prints the accuracy comparison DataFrame and plots a bar chart to visualize the comparison.
The x-axis represents the different methods, and the y-axis represents the accuracy values for the specified parameter.
'''







def compare_prediction_with_true2(prediction_dict, result_dict_test, param):
    # List of method names
    methods = ["LDA", "QDA", "KNN", "Logistic Regression", "Decision Tree", "Random Forest", "SVM", "MLP"]
    for des in prediction_dict:
        # Create a DataFrame to compare the cross-validation accuracy and predicted accuracy
        accuracy_comparison = pd.DataFrame({
            method: {
                'CV': result_dict_test[des].loc[method, param],
                'Prediction': prediction_dict[des].loc[method, param]
            }
            for method in result_dict_test[des].index
        }).T
        accuracy_comparison.index = methods
        # Save the comparison results to a CSV file
        accuracy_comparison.to_csv(f"results/{des}_{param}.csv")
        # Print and plot the accuracy comparison
        print(accuracy_comparison)
        accuracy_comparison.plot(kind='bar', title=des)
        plt.tight_layout()
        plt.ylabel("Accuracy " + param)
        plt.show()

'''
This function splits the input datasets, results, tuning parameters, and tuning parameters with linear transformation into train and test sets.
The test_size parameter specifies the number of datasets to be included in the test set.
The function first shuffles the keys of the datasets randomly to ensure randomness in the split. Then, it selects the first test_size keys as the test set keys, and the remaining keys as the train set keys.
Using these train and test set keys, the function creates dictionaries train_datasets and test_datasets to store the corresponding datasets.
It also retrieves the train and test subsets of the results, tunpar, and tunpar_l dataframes based on the selected keys.
Finally, the function returns the train and test datasets dictionaries, as well as the train and test subsets of the results, tunpar, and tunpar_l dataframes.
'''


def split_datasets(datasets, results, tunpar, tunpar_l, test_size=6):
    # Shuffle the keys of the datasets randomly
    keys = list(datasets.keys())
    np.random.seed(int(time()))
    np.random.shuffle(keys)
    # Select keys for the test set and train set
    test_keys = keys[:test_size]
    train_keys = keys[test_size:]
    # Create train datasets and test datasets dictionaries
    train_datasets = {key: datasets[key] for key in train_keys}
    test_datasets = {key: datasets[key] for key in test_keys}
    # Retrieve train and test results dataframes based on selected keys
    train_results = results.loc[train_keys]
    test_results = results.loc[test_keys]
    # Retrieve train and test tuning parameter dataframes based on selected keys
    train_tun = tunpar.loc[train_keys]
    test_tun = tunpar.loc[test_keys]
    train_tun_l = tunpar_l.loc[train_keys]
    test_tun_l = tunpar_l.loc[test_keys]
    return train_datasets, test_datasets, train_results, test_results, train_tun, test_tun, train_tun_l, test_tun_l


'''
This function evaluates the performance of the model on the test datasets. It takes in the data, results, tunpar, and tunpar_l as input.
The function starts by computing the lengths of the datasets based on the number of rows and the number of positive samples in each dataset.
It then prepares the lengths4 dataframe that stores the total, positive, and negative lengths of the datasets, and saves it to a CSV file.
The tunpar is transformed using transform_tun function based on the computed lengths.
The gene names are retrieved using the getGenes function.
The datasets are split into train and test sets using the split_datasets function.
The results_dict_test dictionary is prepared based on the test results dataframe.
The tunpar and results are scaled using StandardScaler.
The size dictionary is defined to specify the image size and other parameters.
The length dataframe is computed based on the test datasets to determine the lengths for evaluation.
The model is built using the build_model function.
The model is trained using the fit function, and the training history is stored in the history variable.
Predictions are generated for each test dataset using the predictMethod function.
The predictions are stored in perresults, tunresults, and tunresults_l dictionaries.
The perresults and tunresults are converted to dataframes, and the tunresults_l is rounded.
Some transformations and inverse transformations are performed on the scaled results and tuning parameters.
The results_dict_pred dictionary is prepared based on the scaled predictions.
The MSE and bias are computed based on the predictions and test results.
Finally, the function returns the results_dict_pred, results_dict_test, MSE, bias, scaled tuning results, test tuning parameters, tunresults_l, test_tun_l, lengths, and lengths1.
'''


def evaluate(data, results, tunpar, tunpar_l):
    # Compute lengths based on data and tunpar indices
    lengths1 = pd.Series([data[i].shape[0] for i in data.keys()])
    lengths1.index = tunpar.index
    lengths2 = pd.Series([data[i][data[i]["sickness"] > 0].shape[0] for i in data.keys()])
    lengths2.index = tunpar.index
    lengths3 = lengths1 - lengths2
    lengths4 = pd.DataFrame({
        "total": lengths1,
        "positive": lengths2,
        "negative": lengths3
    })
    lengths4.to_csv("lengths.csv")
    # Transform tunpar based on lengths1
    transform_tun(tunpar, lengths1)
    # Retrieve gene names from data
    gens = getGenes(data)
    methods = [col[:-2] for col in results.columns if col.endswith('RT')]
    # Split datasets into train and test sets
    train_datasets, test_datasets, train_results, test_results, train_tun, test_tun, train_tun_l, test_tun_l = split_datasets(data, results, tunpar, tunpar_l)
    # Prepare results_dict_test based on test_results
    results_dict_test = {des: pd.DataFrame({
            'running time': test_results.loc[des, [m+'RT' for m in methods]].values,
            'pos_accuracy': test_results.loc[des, [m+'POSACC' for m in methods]].values,
            'neg_accuracy': test_results.loc[des, [m+'NEGACC' for m in methods]].values,
            'overfitting': test_results.loc[des, [m+'OF' for m in methods]].values},
        index=methods)
        for des in test_results.index}
    # Scale train_tun and train_results using StandardScaler
    tun_scaler = StandardScaler()
    tun_scaler.fit(train_tun)
    train_tun = tun_scaler.transform(train_tun)
    res_scaler = StandardScaler()
    res_scaler.fit(train_results)
    train_results = res_scaler.transform(train_results)
    size = {"img": 140, "conv": 5, "pool": 2}
    # Compute length based on test_datasets
    length = pd.DataFrame({
        "length": {k: np.round(d.shape[0] * 0.2) for k, d in test_datasets.items()},
        "poslength": {k: np.round(np.sum(d["sickness"]) * 0.2) for k, d in test_datasets.items()}
    })
    length["poslength"] = np.min([length["poslength"], length["length"] - 1], axis=0)
    length["neglength"] = length["length"] - length["poslength"]

    # Build the model
    model, images, stat = build_model(train_datasets, size, gens)
    print(model.summary())
    # Fit the model
    history = fit(model, images, stat, train_results, train_tun, train_tun_l)
    perresults = {}
    tunresults = {}
    tunresults_l = {}
    # Generate predictions for each test dataset
    for des in test_datasets.keys():
        predict = predictMethod(test_datasets[des], model, gens, size)
        perresults[des] = predict[0]
        tunresults[des] = predict[1]
        tunresults_l[des] = predict[2]
    # Prepare perresults, tunresults, and tunresults_l dataframes
    perresults = pd.DataFrame(perresults)
    tunresults = pd.DataFrame(tunresults)
    tunresults_l = np.round(pd.DataFrame(tunresults_l)+0.5)
    test_tun_l["svm_gamma_scale"] = 1 - test_tun_l["svm_gamma_scale"]
    tunresults_l = tunresults_l.T+0.5
    tunresults_l.columns = test_tun_l.columns
    tunresults_l["svm_gamma_scale"] = 1 - tunresults_l["svm_gamma_scale"]
    # Inverse transform scaled_per_results and scaled_tun_results
    scaled_per_results = pd.DataFrame(res_scaler.inverse_transform(perresults.T))
    scaled_tun_results = pd.DataFrame(tun_scaler.inverse_transform(tunresults.T))
    scaled_per_results.columns = test_results.columns
    scaled_per_results.index = test_results.index
    scaled_tun_results.columns = test_tun.columns
    scaled_tun_results.index = test_tun.index
    # Detransform scaled_tun_results and test_tun based on lengths1
    detransform_tun(scaled_tun_results, lengths1)
    detransform_tun(test_tun, lengths1)
    # Prepare results_dict_pred based on scaled_per_results
    results_dict_pred = {des: pd.DataFrame({
            'running time': scaled_per_results.loc[des, [m+'RT' for m in methods]].values,
            'pos_accuracy': np.clip(classify(scaled_per_results.loc[des, [m+'POSACC' for m in methods]].values, 1/length.loc[des, "poslength"]), 0, 1),
            'neg_accuracy': np.clip(classify(scaled_per_results.loc[des, [m+'NEGACC' for m in methods]].values, 1/length.loc[des, "neglength"]), 0, 1),
            'overfitting': scaled_per_results.loc[des, [m+'OF' for m in methods]].values},
        index=methods)
        for des in test_results.index}
    # Compute MSE and bias based on results_dict_pred and results_dict_test
    lengths={
      'running time':length["length"],'overfitting':length["length"],'pos_accuracy':length["poslength"],'neg_accuracy':length["neglength"]
    }
    mse = {est: [
        np.mean((df[est] - results_dict_test[des][est])**2) for des, df in results_dict_pred.items()
    ] for est in ['running time', 'pos_accuracy', 'neg_accuracy', 'overfitting']}
    bias = {est: [
        np.mean((df[est] - results_dict_test[des][est])) for des, df in results_dict_pred.items()
    ] for est in ['running time', 'pos_accuracy', 'neg_accuracy', 'overfitting']}
    return [results_dict_pred, results_dict_test, mse, bias, scaled_tun_results, test_tun, tunresults_l, test_tun_l, lengths, lengths1]



'''
This function is responsible for showing the evaluation results of the model. It takes in various evaluation results and visualizes them.
The function first calls the compare_prediction_with_true2 function to compare the predictions with the true values for positive accuracy and negative accuracy.
Then, it creates an estimation dataframe estim containing the MSE and bias values and saves it to a CSV file.
Next, it plots the MSE for each estimation using scatter plots. The x-axis represents the number of observations, and the y-axis represents the Mean Squared Error (MSE).
Finally, it plots the bias for each estimation using scatter plots. The x-axis represents the number of observations, and the y-axis represents the bias.
'''

def show_results(results_dict_pred, results_dict_test, mse, bias, scaled_tun_results, test_tun, tunresults_l, test_tun_l, lengths, lengths1):
    # Compare predictions with true values for positive accuracy
    compare_prediction_with_true2(results_dict_pred, results_dict_test, "pos_accuracy")
    # Compare predictions with true values for negative accuracy
    compare_prediction_with_true2(results_dict_pred, results_dict_test, "neg_accuracy")
    # Prepare estimation dataframe and save it to CSV
    estim = pd.DataFrame({
        "mse": mse,
        "bias": bias
    })
    estim.to_csv("results/performence.csv")
    # Plot MSE for each estimation
    for k, v in mse.items():
        plt.scatter(lengths[k], v)
        plt.title(f"The MSE of {k} vs n")
        plt.ylim(-0.1, 1)
        plt.xlabel("Number of observations")
        plt.ylabel("Mean Squared Error")
        plt.grid()
        plt.show()
    # Plot bias for each estimation
    for k, v in bias.items():
        plt.scatter(lengths[k], v)
        plt.title(f"The BIAS of {k} vs n")
        plt.ylim(-1, 1)
        plt.xlabel("Number of observations")
        plt.ylabel("Bias")
        plt.grid()
        plt.show()

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Available GPUs:", physical_devices)
data=loadData()
results,tunpar,tunpar_l=loadResults()
results_dict_pred,results_dict_test,mse,bias,scaled_tun_results,test_tun,tunresults_l,test_tun_l,lengths,lengths1=evaluate(data,results,tunpar,tunpar_l)
show_results(results_dict_pred,results_dict_test,mse,bias,scaled_tun_results,test_tun,tunresults_l,test_tun_l,lengths,lengths1)
scaled_tun_results["random_forest_n_estimators"]=10**scaled_tun_results["random_forest_n_estimators"]
test_tun["random_forest_n_estimators"]=10**test_tun["random_forest_n_estimators"]
l=lengths1[test_tun.index]
transform_tun(test_tun,l)
transform_tun(scaled_tun_results,l)

scaler=StandardScaler()
scaler.fit(test_tun)
scaled_test_tun,scaled_res_tun=scaler.transform(test_tun),scaler.transform(scaled_tun_results)

dist=(scaled_test_tun-scaled_גres_tun)**2
dist=pd.DataFrame(dist)
dist.index=test_tun.index
dist.columns=test_tun.columns
dist.to_csv("results/tun_mse")