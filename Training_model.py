#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename: Training_model.py
# @Author: Lucas VERGEZ
# @Time: 24/01/2024 14:39
import pandas as pd
import Data_analysis as data
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression


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
    # regr = MLPRegressor(random_state=1, max_iter=500)
    regr = LogisticRegression(max_iter=5000)
    regr.fit(X_train, y_train)
    # regr.predict(X_test)
    s = regr.score(X_test, y_test)


def select_features(x, y):
    # reg = MLPRegressor(random_state=1, max_iter=500)
    model = SelectFromModel(LogisticRegression(), prefit=False)
    model.fit(x, y)
    return model.transform(x)


if __name__ == "__main__":
    # datas = data.DataManager()
    # datas.load("virga")

    name = "VECTOR"

    dftest = pd.read_csv("..\\test_" + name + ".csv", low_memory=False)
    dftrain = pd.read_csv("..\\train_" + name + ".csv", low_memory=False)

    ytest = dftest["net_cutting_time_s"].to_numpy()
    dftest.drop(["net_cutting_time_s"], axis=1, inplace=True)
    xtest = dftest.to_numpy()

    ytrain = dftrain["net_cutting_time_s"].to_numpy()
    dftrain.drop(["net_cutting_time_s"], axis=1, inplace=True)
    xtrain = dftrain.to_numpy()

    regr = MLPRegressor(random_state=1, max_iter=500)
    regr.fit(xtrain, ytrain)
    print(regr.score(xtest, ytest))

    # train(x, y)
    # a = select_features(x, y)
    # VIRGA: 0.9601053739149665
    # VECTOR: 0.9579861621881663

    print()
