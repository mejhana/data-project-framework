# -*- coding: utf-8 -*-
"""
Task : To preprocess the dataset provided based on a series of steps and 
       return the preprocessed dataset
Steps :
    1. Remove the "Identifier Column". Eg: ID, Serial No. etc
    2. 
    3. 
"""

import sys
sys.path.append("../utils/")
import data_cleaner as dclr
from sklearn import preprocessing

def preliminary_preprocessing(df):
    df.columns = dclr.prettify_colnames(df)
    
    return df

def min_max_scaler(features):
    """
    Min Max Scaler
    """
    min_max_scaler = preprocessing.MinMaxScaler()
    scaled = min_max_scaler.fit_transform(features)
    
    return scaled

def scaler(features):
    """
    Choose the scaler required
    """
    
    # change this function call to change the scaler
    return min_max_scaler(features)

def create_descriptor_target(train, target, scale_descr=True):
    """ 
    Splitting the Training set to descriptors and target
    
    PARAMETERS
        train : The Training Dataset (dataframe)
        target : The Target Feature (dataframe)
        scale_descr : Scale the descriptor? (bool) [Scale by deafult]
        
    OUTPUTS :
        1. Returns a tuple : (descriptors, targets)
    """
    # target
    y = train[target].values.astype(float)
    # descriptors
    x = train.drop(["chance_of_admit"], axis=1).values.astype(float)
    if(scale_descr == True):
        x = scaler(x)
    
    return (x,y)


    
    
    

