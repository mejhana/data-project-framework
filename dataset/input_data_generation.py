# -*- coding: utf-8 -*-
"""
Task : To generate the train and test datasets from the main data file
       admission_predict_data.csv
Steps :
    1. pandas sample function is used to randomly select 100 samples
    2. These 100 samples form the test dataset; the rest go to train
    3. The "Chance of Admit " feature in test dataset is stored separately and
       removed of the dataset as it is; saved as solution.csv
    4. train.csv, test.csv and solution.csv are all stored back into the
       dataset folder
"""

import pandas as pd

try:
    df = pd.read_csv("admission_predict_data.csv")
except:
    print("Error while loading main dataset!")

# Split the data into training and testing data
test = df.sample(n=100, random_state=42).reset_index(drop=True)
train = df.drop(list(test.index)).reset_index(drop=True)

print("Train :", train.shape)
print("Test :", test.shape)

# Store Target values of test separately and remove them of the main dataframe
test_target = test["Chance of Admit "].values
test = test.drop("Chance of Admit ", axis=1)

# Make the submission dataframe
# This is the final values with which we have to
# compare our predictions on the test dataset
solution = pd.DataFrame({"target":test_target})

# Store the files back as train.csv, test.csv and submission.csv
train.to_csv("train.csv", index=False)
test.to_csv("test.csv", index=False)
solution.to_csv("solution.csv", index=False)

print("All datasets have been generated!")


                                                                               