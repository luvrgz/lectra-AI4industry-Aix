#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename: Training_model.py
# @Author: Lucas VERGEZ
# @Time: 24/01/2024 14:39
import pandas as pd
import Data_analysis as data
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


def finito_presto(datas):
    datas.remove_column("net_cutting_time_s")
    datas.remove_column("Unnamed: 0")
    datas.remove_column("marker_name")
    datas.remove_column("activity_classname")
    datas.remove_column("product_ref")
    datas.remove_column("material_name")
    datas.remove_column("execution")
    datas.remove_column("requirement")
    datas.remove_qualitatives(max_values=2)
    X = pd.get_dummies(datas.df)
    return X.astype(dtype="float64")


def train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2)
    regr = MLPRegressor(random_state=1, max_iter=500)
    regr.fit(X_train, y_train)
    # regr.predict(X_test)
    s = regr.score(X_test, y_test)


if __name__ == "__main__":
    datas = data.DataManager()
    datas.load("virga")

    y = datas.df["net_cutting_time_s"].to_numpy()
    x = finito_presto(datas)

    train(x, y)

    print()
