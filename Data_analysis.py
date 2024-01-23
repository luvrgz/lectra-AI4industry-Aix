#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename: Data_analysis.py
# @Author: Lucas VERGEZ
# @Time: 22/01/2024 15:12
import pandas as pd
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def import_data():
    csv_path = "lectra_dataset.csv"
    # df = pd.read_csv(csv_path, low_memory=False)
    with open('..\\LECTRA\\df.pkl', 'rb') as handle:
        # pickle.dump(df, handle)
        a = pickle.load(handle)
        return a


class DataManager:
    def __init__(self, dataframe):
        self.df = dataframe
        self.original_length = len(self.df)  # 794 341
        self.all_masks = dict()

    def save(self, name):
        with open('..\\LECTRA\\' + name + '.pkl', 'wb') as handle:
            pickle.dump(self.df, handle)

    def load(self, name):
        with open('..\\LECTRA\\' + name + '.pkl', 'rb') as handle:
            self.df = pickle.load(handle)
            self.original_length = len(self.df)

    def check_net(self, margin=0):
        """To check: 'gross_cutting_time_s' = 'net_cutting_time_s' + 'total_interruption_time_s ajouter marge'"""
        df_mask = abs(self.df["gross_cutting_time_s"] - self.df["net_cutting_time_s"] - self.df["total_interruption_time_s"]) > margin
        self.all_masks["check_net"] = df_mask
        print("(check_net) Number of deleted: (margin=", margin, ") ", np.count_nonzero(df_mask.values))

    def apply_masks(self):
        merged_mask = None
        for k, mask in enumerate(self.all_masks.values()):
            if k == 0:
                merged_mask = mask
            else:
                merged_mask = merged_mask.where(merged_mask & mask)
        self.df = self.df.where(~merged_mask)

    def add_time_column(self):
        def try_to_pred(row):
            if row["cutting_speed"] != 0:
                return row["perimeter_actual"] / row["cutting_speed"]
            else:
                return 0
        self.df['estimated_time'] = self.df.apply(try_to_pred, axis=1)

    def plot_sns(self, list_of_features, classes=None):
        """Par exemple:
        plot_sns(["perimeter_actual", "cutting_speed", "net_cutting_time_s", "product_model"],
        classes=product_model):"""
        truncated_df = self.df[list_of_features]
        sns.pairplot(truncated_df, hue=classes)
        plt.show()


if __name__ == "__main__":
    df = import_data()
    dm = DataManager(df)
    dm.add_time_column()
    dm.plot_sns(["net_cutting_time_s", "estimated_time"])

    dm.check_net()
    dm.apply_masks()
    print()


