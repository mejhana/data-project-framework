# -*- coding: utf-8 -*-
"""
UTILITY SCRIPT

Task : To provide background information about the train and test datasets

Functions :
    1. train_test_compare
    2. train_test_dist
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def train_test_compare(train_df, test_df):
    """ 
    Comparing the details of train and test datasets
    
    PARAMETERS
        train_df : Training set pandas dataframe (dataframe)
        test_df : Testing set pandas dataframe (dataframe)
        
    OUTPUTS :
        1. Returns a dataframe with the necessary comparisons
        2. Columns - # columns, # instances, 
        3. Identifies the Target feature
        4. Identifies missing values in each data set
        5. Display summary statistics for each data set
    """
    
    # Generating the Comparison table
    data = ["Train Set", "Test Set"]
    columns = [train_df.shape[1], test_df.shape[1]]
    instances = [train_df.shape[0], test_df.shape[0]]
    temp_dict = {
                 "Type_of_Data":data,
                 "Number_of_Columns":columns, 
                 "Number_of_Instances":instances
                }
    print("The Comparison Table :")
    print(pd.DataFrame(temp_dict).T)
    print("------------------------------")
    
    # Identifying the Target feature
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)
    print("\nPotential Target :", train_cols-test_cols)
    print("------------------------------")
    
    # Identifying Missing values
    print("\nMissing Value Count in Train Set :")
    print(train_df.isna().sum())
    print("\nMissing Value Count in Test Set :")
    print(test_df.isna().sum())
    print("------------------------------")
    
    # Displaying Summary Statistics
    print("\nSummary Statistics for Train Set :")
    print(train_df.describe())
    print("\nSummary Statistics for Test Set :")
    print(test_df.describe())
    print("------------------------------")
       
    return None

def train_test_dist(train_df, test_df, shade_train=False, shade_test=False):
    """
    Comparing the Density distribution plots
    of train and test sets
    
    PARAMETERS :
        train_df : Training set pandas dataframe (dataframe)
        test_df : Testing set pandas dataframe (dataframe)
        shade_train : Specifies if the train density plot
                      needs to be shaded (boolean)
        shade_test : Specifies if the test density plot
                      needs to be shaded (boolean)
                      
    OUTPUTS :
        Kinetic Density Plots
    """
    # the descriptor columns (all except Target)
    cols = list(set(train_df.columns).intersection(set(test_df.columns)))
    
    for c in cols:
        sns.kdeplot(train_df[c], label="Train Set", shade=shade_train)
        sns.kdeplot(test_df[c], label="Test Set", shade=shade_test)
        plt.title("Train vs Test Distribution for "+c)
        plt.xlabel("Samples")
        plt.ylabel("Probability")
        plt.show()
        
    return None