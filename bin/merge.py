#!/usr/bin/env python3

import gc
import pickle
import pandas as pd

"""
	FILES IN: 
                'tenNN_items.pickle'
                'threeNN_shops.pickle'
                'main_data_feature_eng_2.pickle'
                'main_data_feature_eng_3.pickle'
                'main_data_feature_eng_4.pickle'
                'main_data_feature_eng_5.pickle'

	FILES OUT: 
		'train_group_0.pickle'
                'train_group_1.pickle'
                'train_group_2.pickle'
                'test_group_0.pickle'
                'test_group_1.pickle'
                'test_group_2.pickle'
                'main_data_feature_eng_all.pickle'
 """


def merge_dataframes(d1, d2, cols):
    return pd.merge(d1, d2, on=cols, how="left")


def get_train_0(data):
    return data.loc[data["item_seniority"] == 0, :]


def get_train_1(data):
    return data.loc[data["item_seniority"] == 1, :]


def get_train_2(data):
    return data.loc[data["item_seniority"] == 2, :]


def main():
    gc.enable()
    #tenNN_items    = pd.read_pickle("tenNN_items.pickle")
    #threeNN_shops  = pd.read_pickle("threeNN_shops.pickle")
    
    main_data_2    = pd.read_pickle("main_data_feature_eng_2.pickle")
    main_data_3    = pd.read_pickle("main_data_feature_eng_3.pickle")
    main_data_4    = pd.read_pickle("main_data_feature_eng_4.pickle")
    main_data_5    = pd.read_pickle("main_data_feature_eng_5.pickle")
    
    #merge all dataframes with features to one dataframe
    cols = ["date_block_num", "shop_id", "item_id", "item_cnt_month", "item_category_id"]
    main_data_merged = merge_dataframes(main_data_2, main_data_3, cols)
    main_data_2 = None
    main_data_3 = None
    
    main_data_merged = merge_dataframes(main_data_merged, main_data_4, cols)
    main_data_4 = None
    
    main_data_merged = merge_dataframes(main_data_merged, main_data_5, cols)
    main_data_5 = None

    #main_data_merged = merge_dataframes(main_data_merged, tenNN_items, ["item_id"])
    #main_data_merged = merge_dataframes(main_data_merged, threeNN_shops, ["shop_id"])

    #split dataframe to groups based on seniority
    group_0 = get_train_0(main_data_merged)
    group_1 = get_train_1(main_data_merged)
    group_2 = get_train_2(main_data_merged)

    train_group_0 = group_0[group_0.date_block_num < 34]
    train_group_1 = group_1[group_1.date_block_num < 34]
    train_group_2 = group_2[group_2.date_block_num < 34]

    test_group_0 = group_0[group_0.date_block_num == 34]
    test_group_1 = group_1[group_1.date_block_num == 34]
    ttest_group_2 = group_2[group_2.date_block_num == 34]
    
    #save output
    pickle.dump(train_group_0, open("train_group_0.pickle", "wb"), protocol=4)
    pickle.dump(train_group_1, open("train_group_1.pickle", "wb"), protocol=4)
    pickle.dump(train_group_2, open("train_group_2.pickle", "wb"), protocol=4)
    pickle.dump(test_group_0, open("test_group_0.pickle", "wb"), protocol=4)
    pickle.dump(test_group_1, open("test_group_1.pickle", "wb"), protocol=4)
    pickle.dump(test_group_2, open("test_group_2.pickle", "wb"), protocol=4)
    pickle.dump(main_data_merged, open("main_data_feature_eng_all.pickle", "wb"), protocol=4)


if __name__ == "__main__":
    main()
