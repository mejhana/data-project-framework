# -*- coding: utf-8 -*-
"""
Task : To train our ML algorithms on the data and store the trained models
"""

import pandas as pd
import joblib

import preprocess as prp
import dispatcher
import metrics

TRAIN_DATA = "../dataset/train.csv"

if __name__ == "__main__":
    train = pd.read_csv(TRAIN_DATA)
    
    # drop "Serial No."
    train = train.drop("Serial No.", axis=1)
    
    # using our defined preprocessing function
    train = prp.preliminary_preprocessing(train)
    
    # split into descriptors and target
    (descriptors, target) = prp.create_descriptor_target(train, "chance_of_admit")
    
    # choosing the model
    models_used = []
    metric_scores = []
    
    for model in list(dispatcher.MODELS.keys()):
        regr = dispatcher.MODELS[model]
        regr.fit(descriptors, target)
        predictions = regr.predict(descriptors)
        models_used.append(model)
        metric_scores.append(metrics.root_mean_square_error(target, predictions))
        
        # store the trained model
        joblib.dump(regr, f"../models/{model}.pkl")
        
    result_dict = {"Models":models_used,
                   "Metric Score":metric_scores
                  }
    print("\nMetric Used : Root Mean Square Error\n")
    print(pd.DataFrame(result_dict))
    print("\nAll trained models have been saved in the ../models/ directory!\n")
    
    