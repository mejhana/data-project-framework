# -*- coding: utf-8 -*-
"""
Task : To describe models to be used
"""
from sklearn import linear_model

MODELS = {
    "linear_regression": linear_model.LinearRegression(),
    "ridge_regression": linear_model.Ridge(alpha=0.5),
    "elastic_net": linear_model.ElasticNet(random_state=42, max_iter=500),       
}

