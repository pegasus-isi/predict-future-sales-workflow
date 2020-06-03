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
 """


def merge_all_features_and_output(data, cols):
    extra_data = pd.read_pickle("main_data_feature_eng_2.pickle")
    data = pd.merge(data, extra_data, on=cols, how="left")
    extra_data = pd.read_pickle("main_data_feature_eng_3.pickle")
    data = pd.merge(data, extra_data, on=cols, how="left")
    extra_data = pd.read_pickle("main_data_feature_eng_5.pickle")
    data = pd.merge(data, extra_data, on=cols, how="left")
    extra_data = None
    
    return data


def get_train_0(data):
    return data.loc[data["item_seniority"] == 0, :]


def get_train_1(data):
    return data.loc[data["item_seniority"] == 1, :]


def get_train_2(data):
    return data.loc[data["item_seniority"] == 2, :]


def main():
    gc.enable()
    
    main_data_seniority = pd.read_pickle("main_data_feature_eng_4.pickle")
    
    #split dataframe to groups based on seniority
    train_group_0 = get_train_0(main_data_seniority)
    train_group_1 = get_train_1(main_data_seniority)
    train_group_2 = get_train_2(main_data_seniority)
    main_data_seniority = None

    #merge all features for all seniorities
    cols = ["date_block_num", "shop_id", "item_id", "item_cnt_month", "item_category_id"]
    train_group_0 = merge_all_features_and_output(train_group_0, cols)
    train_group_1 = merge_all_features_and_output(train_group_1, cols)
    train_group_2 = merge_all_features_and_output(train_group_2, cols)
    
    #save output
    pickle.dump(train_group_0, open("train_group_0.pickle", "wb"), protocol=4)
    pickle.dump(train_group_1, open("train_group_1.pickle", "wb"), protocol=4)
    pickle.dump(train_group_2, open("train_group_2.pickle", "wb"), protocol=4)


if __name__ == "__main__":
    main()
