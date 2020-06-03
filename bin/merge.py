#!/usr/bin/env python3

import pandas as pd
import numpy as np
import pickle

"""

	FILES IN: 
                'tenNN_items.pickle'
                'threeNN_shops.pickle'
                'main_data_feature_eng_2.pickle'
                'main_data_feature_eng_3.pickle'
                'main_data_feature_eng_4.pickle'
                'main_data_feature_eng_5.pickle'

	FILES OUT: 
		'train_0.pickle'
                'train_1.pickle'
                'train_2.pickle'
 """


def merge_dataframes(d1, d2, cols):
    merged = pd.merge(d1, d2, on=cols, how="left")
    return merged


def get_train_0():
    return


def get_train_1():
    return


def get_train_2():
    return


def main():
    tenNN_items    = pd.read_pickle("tenNN_items.pickle")
    threeNN_shops  = pd.read_pickle("threeNN_shops.pickle")
    
    main_data_2    = pd.read_pickle("main_data_feature_eng_2.pickle")
    main_data_3    = pd.read_pickle("main_data_feature_eng_3.pickle")
    main_data_4    = pd.read_pickle("main_data_feature_eng_4.pickle")
    main_data_5    = pd.read_pickle("main_data_feature_eng_5.pickle")

    cols = ["date_block_num", "shop_id", "item_id", "item_cnt_month", "item_category_id"]
    main_data_merged = merge_dataframes(main_data_2, main_data_3, cols)
    main_data_merged = merge_dataframes(main_data_merged, main_data_4, cols)
    main_data_merged = merge_dataframes(main_data_merged, main_data_5, cols)

    #main_data_merged = merge_dataframes(main_data_merged, tenNN_items, ["item_id"])
    #main_data_merged = merge_dataframes(main_data_merged, threeNN_shops, ["shop_id"])

    print(main_data_merged.columns)
    print(main_data_merged.shape)


if __name__ == "__main__":
    main()
