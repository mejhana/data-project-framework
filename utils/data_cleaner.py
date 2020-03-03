# -*- coding: utf-8 -*-
"""
UTILITY SCRIPT

Task : To perform general data cleaning fucntions where applicable

Functions :
    1. train_test_compare
    2. train_test_dist
"""
import pandas as pd

def prettify_colnames(df):
    """ 
    Cleaning up column names for readbility and ease of accesibility
    
    PARAMETERS
        df : Dataframe to be cleaned (dataframe)
        
    OUTPUTS :
        1. Returns a list of cleaned column names for the given dataframe
    """
    cols = list(df.columns)
    new_cols=[]
    for c in cols:
        new_cols.append(c.strip().lower().replace(" ", "_"))
        
    return new_cols