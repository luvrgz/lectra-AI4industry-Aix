#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename: Data_analysis.py
# @Author: Lucas VERGEZ
# @Time: 22/01/2024 15:12
import pandas as pd
import pickle
import seaborn as sns


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

    def check_net(self):
        """To check: 'gross_cutting_time_s' = 'net_cutting_time_s' + 'total_interruption_time_s'"""
        df_mask = self.df["gross_cutting_time_s"] != self.df["net_cutting_time_s"] + self.df["total_interruption_time_s"]
        return df_mask


if __name__ == "__main__":
    df = import_data()
    dm = DataManager(df)
    mask1 = dm.check_net()
    print()


