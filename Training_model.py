#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename: Training_model.py
# @Author: Lucas VERGEZ
# @Time: 24/01/2024 14:39
from sklearn.preprocessing import LabelBinarizer
import Data_analysis as data


def final_processing(x):
    one_hot_enc = LabelBinarizer()
    one_hot_enc.fit(x)
    return one_hot_enc.transform(x)


if __name__ == "__main__":
    datas = data.DataManager()
    datas.load("virga")

    Y = datas.df["net_cutting_time_s"].to_numpy()

    datas.remove_column("net_cutting_time_s")
    datas.remove_column("Unnamed: 0")
    datas.remove_column("marker_name")
    datas.remove_column("activity_classname")
    datas.remove_column("product_ref")
    datas.remove_column("material_name")
    datas.remove_column("execution")
    datas.remove_column("requirement")
    datas.remove_qualitatives(max_values=2)
    X = final_processing(datas.df.to_numpy())
    print()
