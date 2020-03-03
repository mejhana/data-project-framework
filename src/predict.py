# -*- coding: utf-8 -*-
"""
Task : To test our algorithms on the test data by loading the pre-trained
       models
"""

import pandas as pd
import joblib

import preprocess as prp
import metrics

TEST_DATA = "../dataset/test.csv"
SOLUTION_DATA = "../dataset/solution.csv"
MODELS = "../models"

def predict():
    test = pd.read_csv(TEST_DATA)
    
    # drop "Serial No."
    test = test.drop("Serial No.", axis=1)
    
    # using our defined preprocessing function
    test = prp.preliminary_preprocessing(test)
    
    # scale the test dataframe like before
    test = prp.scaler(test)
    
    # choose the model you want to use
    model = "linear_regression"
    regr = joblib.load(f"{MODELS}/{model}.pkl")
    
    # use the model to predict
    predictions = regr.predict(test)
    
    # return the predictions (format later if necessary)
    return (predictions)

def evaluate():
    """
    > This function will not ideally be used in contests
    > This is used here only for the purpose of checking the performance of the
      model on the test set initally generated
    """
    solution = pd.read_csv(SOLUTION_DATA)
    actual = solution["target"].values
    submission = predict()
    
    # save the submission file in the output folder
    sub_df = pd.DataFrame({"Chance of Admit":submission})
    sub_df.to_csv("../output/submission.csv")
    print("\nThe Submission file has been saved in ../output/")
    
    return (metrics.root_mean_square_error(actual, submission))

if __name__ == "__main__":
    print("Root Mean Square Error :", evaluate(), "\n")