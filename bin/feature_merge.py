#!/usr/bin/env python3

import gc
import json
import pickle
import pandas as pd
from argparse import ArgumentParser

"""
	FILES IN: 
                'tenNN_items.pickle'
                'threeNN_shops.pickle'
                'merged_features.json'
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
                'merged_features_output.json'
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


def sort_columns(columns_dict, columns_unsorted):
    sorting_map = [ columns_dict[col] for col in columns_unsorted ]
    columns_sorted = [col for _,col in sorted(zip(sorting_map,columns_unsorted))]
    return columns_sorted


def create_columns_dict(columns, mandatory):
    columns_dict = {}
    for i in range(len(mandatory)):
        columns_dict[mandatory[i]] = i
    for i in range(len(columns)):
        if not columns[i] in mandatory:
            columns_dict[columns[i]] = i + len(mandatory)

    return columns_dict


def fiter_merged_columns(data, columns_new, columns_mandatory):
    #keep columns specified in config file
    columns_current = list(data.columns)
    columns_dict = create_columns_dict(columns_current, columns_mandatory)

    for col in columns_mandatory:
        if not col in columns_new:
            columns_new.append(col)

    columns_kept = sort_columns(columns_dict, list(set(columns_current) & set(columns_new)))
    columns_dropped = sort_columns(columns_dict, list((set(columns_current) ^ set(columns_new)) & set(columns_current)))
    columns_not_found = list((set(columns_current) ^ set(columns_new)) & set(columns_new))
    
    print(f"Columns not found: {columns_not_found}")
    print(f"Columns dropped: {columns_dropped}")
    print(f"Columns kept: {columns_kept}")
    
    return data[columns_kept]


def main():
    gc.enable()
    
    parser = ArgumentParser(description="Merge all generated features")
    parser.add_argument("--cols", metavar="STR", type=str, default="", help="JSON file with columns to keep after merging", required=False)
    args = parser.parse_args()

    columns_mandatory = ["date_block_num", "shop_id", "item_id", "item_cnt_month", "item_category_id", "item_seniority"]

    tenNN_items    = pd.read_pickle("tenNN_items.pickle")
    threeNN_shops  = pd.read_pickle("threeNN_shops.pickle")
    
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

    main_data_merged = merge_dataframes(main_data_merged, tenNN_items, ["item_id"])
    tenNN_items = None

    main_data_merged = merge_dataframes(main_data_merged, threeNN_shops, ["shop_id"])
    threeNN_shops = None

    #filter columns to keep only the ones specified in the config
    if args.cols:
        columns_new = json.load(open(args.cols, "r"))["columns"]
        main_data_merged = fiter_merged_columns(main_data_merged, columns_new, columns_mandatory)
    
    #split dataframe to groups based on seniority
    group_0 = get_train_0(main_data_merged)
    train_group_0 = group_0[group_0.date_block_num < 34]
    test_group_0 = group_0[group_0.date_block_num == 34]
    group_0 = None
    
    group_1 = get_train_1(main_data_merged)
    train_group_1 = group_1[group_1.date_block_num < 34]
    test_group_1 = group_1[group_1.date_block_num == 34]
    group_1 = None

    group_2 = get_train_2(main_data_merged)
    train_group_2 = group_2[group_2.date_block_num < 34]
    test_group_2 = group_2[group_2.date_block_num == 34]
    group_2 = None
    
    #save output
    pickle.dump(train_group_0, open("train_group_0.pickle", "wb"), protocol=4)
    pickle.dump(train_group_1, open("train_group_1.pickle", "wb"), protocol=4)
    pickle.dump(train_group_2, open("train_group_2.pickle", "wb"), protocol=4)
    pickle.dump(test_group_0, open("test_group_0.pickle", "wb"), protocol=4)
    pickle.dump(test_group_1, open("test_group_1.pickle", "wb"), protocol=4)
    pickle.dump(test_group_2, open("test_group_2.pickle", "wb"), protocol=4)
    pickle.dump(main_data_merged, open("main_data_feature_eng_all.pickle", "wb"), protocol=4)
    json.dump({"columns": list(main_data_merged.columns)}, open("merged_features_output.json", "w"), indent=2)


if __name__ == "__main__":
    main()
