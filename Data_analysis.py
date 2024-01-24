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


def import_data(csv_path):
    """To convert CSV data into pickle object"""
    df = pd.read_csv(csv_path, low_memory=False)
    with open('..\\df.pkl', 'wb') as handle:
        pickle.dump(df, handle)


class DataManager:
    def __init__(self, dataframe=None):
        self.df = dataframe
        self.all_line_masks = dict()
        self.all_column_masks = dict()

    def save(self, name):
        with open('..\\' + name + '.pkl', 'wb') as handle:
            pickle.dump(self.df, handle)

    def load(self, name):
        with open('..\\' + name + '.pkl', 'rb') as handle:
            self.df = pickle.load(handle)

    def check_net(self, margin=0):
        """To check: 'gross_cutting_time_s' = 'net_cutting_time_s' + 'total_interruption_time_s ajouter marge'
        Il faut delete les True"""
        df_mask = abs(self.df["gross_cutting_time_s"] - self.df["net_cutting_time_s"] - self.df["total_interruption_time_s"]) > margin
        self.all_line_masks["check_net"] = df_mask
        print("(check_net) Number of deleted: (margin=", margin, ") ", np.count_nonzero(df_mask.values))

    def nan_market(self):
        """Il faut delete les True"""
        df_mask = self.df["market"].isnull()
        self.all_line_masks["nan_market"] = df_mask
        print("(nan_market) Number of deleted: ", np.count_nonzero(df_mask.values))

    def delete_model(self, name):
        """name="VECTOR" or "VIRGA"
        Il faut delete les True"""
        df_mask = self.df["production_line"] == name
        self.all_line_masks["model_" + name] = df_mask
        print("(delete_model) Number of deleted: ", np.count_nonzero(df_mask.values))

    def keep_interval(self, column, min_value, max_value, show=False):
        """Il faut delete les True"""
        if show:
            sns.histplot(data=dm.df, x=column)
            plt.show()
        df_mask_max = self.df[column] > max_value
        df_mask_min = self.df[column] < min_value
        df_mask = np.logical_or.reduce([df_mask_min, df_mask_max])
        self.all_line_masks["interval_" + column] = df_mask
        print("(keep_interval", column, ") Number of deleted: ", np.count_nonzero(df_mask))

    def remove_lines_duplicata(self):
        # Suppression des colonnes avec des valeurs identiques
        self.df.drop_duplicates(keep="first", inplace=True)

    def apply_line_masks(self):
        self.df = self.df[~np.logical_or.reduce(list(self.all_line_masks.values()))]
        self.all_line_masks = dict()

    def find_nan_column(self, max_empty=0.0):
        """max_empty est entre 0-->supprimé si une case vide et 1-->rien n'est supprimé"""
        number_of_nan = self.df.isna().sum()
        column_mask = number_of_nan.apply(lambda x: x / len(self.df) > max_empty)
        self.all_column_masks["nan_column"] = column_mask
        print("(find_nan_column) Number of deleted: ", np.count_nonzero(column_mask))

    def apply_column_masks(self):
        a = np.logical_or.reduce([mask.values for mask in self.all_column_masks.values()])
        column_indexs_to_delete = np.where(a)
        self.df.drop(self.df.columns[column_indexs_to_delete], axis=1, inplace=True)
        self.all_column_masks = dict()

    def remove_column(self, column):
        self.df.drop([column], axis=1, inplace=True)

    def plot_sns(self, list_of_features, classes=None):
        """Par exemple:
        plot_sns(["perimeter_actual", "cutting_speed", "net_cutting_time_s", "product_model"],
        classes=product_model):"""
        truncated_df = self.df[list_of_features]
        sns.pairplot(truncated_df, hue=classes)
        plt.show()


if __name__ == "__main__":
    # IMPORT
    csv_path = "lectra_dataset.csv"
    import_data(csv_path)
    dm = DataManager()
    dm.load("df")

    # LINES PREPROCESSING
    dm.delete_model("VECTOR")
    dm.check_net(margin=0)
    dm.keep_interval("net_cutting_time_s", 5, 3600)
    dm.keep_interval("nb_interruptions", -1, 12)
    dm.nan_market()
    dm.apply_line_masks()
    dm.remove_lines_duplicata()

    # COLUMN PREPROCESSING
    dm.find_nan_column(max_empty=0.005)
    dm.apply_column_masks()
    dm.remove_column("serial_number")
    dm.remove_column("start_date")
    dm.remove_column("end_date")

    # EXPORT
    dm.df.to_csv('VIRGA.csv')
    print()


