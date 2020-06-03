#!/usr/bin/env python3

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

"""
Preprocesses data for the Future Sales Predictions

	FILES IN: 
		"sales_train.csv",
		"items.csv",
		"item_categories.csv",
		"test.csv",
		"shops.csv"

	FILES OUT: 
		'items_preprocessed.pickle',
		'shops_preprocessed.pickle',
		'categories_preprocessed.pickle',
		'sales_train_preprocessed.pickle',
		'test_preprocessed.pickle'

 """

# -----------------           HELPER  FUNCTIONS       -------------------------


def remove_outliers(trainset):
    trainset = trainset[(trainset.item_price < 100000 )& (trainset.item_cnt_day < 1000)]
    trainset = trainset[trainset.item_price > 0].reset_index(drop = True)
    trainset.loc[trainset.item_cnt_day < 1, "item_cnt_day"] = 0	
    return trainset


def replace_shop_ids(dataset,ids_pairs,id_replaced):	
    for id_pair in ids_pairs:
        dataset.loc[dataset[id_replaced] == id_pair[0], id_replaced]  = id_pair[1]
    return dataset


def drop_rows_by_index(dataframe, rows_ids):
    for row in rows_ids:
        dataframe.drop(row, axis=0, inplace=True)
    return dataframe


def drop_rows_by_col_val(dataframe, col_name, ids):
    for i in ids:
        dataframe.drop(dataframe.loc[dataframe[col_name] == i].index, axis = 0, inplace = True)
    return dataframe


# -----------------        PREPROCESSING FUNCTIONS       -------------------------

# deletes 7 entries from the stores
def preprocess_shops(shops):	
    shops_ids_to_drop = [0,1,11,40,9,20,33]
    shops             = drop_rows_by_index(shops,shops_ids_to_drop )
    return shops


# deletes 4 unneeded categories
def preprocess_categories(categories, categories_ids_drop):
    categories = drop_rows_by_index(categories, categories_ids_drop)
    return categories


# deletes 4 items
def preprocess_items(items,categories_ids_drop):
    items = drop_rows_by_index(items, categories_ids_drop)
    return items


# 18953 rows of undeeded data are removed
def preprocess_trainset(trainset, shop_ids_pairs, categories_ids_drop, items):
    trainset = remove_outliers(trainset)
    trainset = replace_shop_ids(trainset, shop_ids_pairs,"shop_id")
    shops_ids_drop = [9,20,33]
    trainset = drop_rows_by_col_val(trainset,"shop_id",shops_ids_drop)

    for i in categories_ids_drop:
        trainset.drop(trainset.loc[(trainset['item_id']
	        .map(items['item_category_id'])== i)].index.values,axis = 0,inplace = True)
    
    return trainset


def preprocess_testset(testset,shop_ids_pairs):
    testset = replace_shop_ids(testset, shop_ids_pairs,"shop_id")
    return testset


def main():
    # Read in the data for the analysis
    sales_trainset   = pd.read_csv("sales_train.csv")
    items            = pd.read_csv("items.csv")
    categories       = pd.read_csv("item_categories.csv")
    testset          = pd.read_csv("test.csv")
    shops            = pd.read_csv("shops.csv")
       
    # Replace store ids of duplicated stores
    shop_ids_pairs      = [[0,57], [1,58], [11,10],[40,39]]
    categories_ids_drop = [8,80,81,82]

    sales_trainset = preprocess_trainset(sales_trainset, shop_ids_pairs, categories_ids_drop, items)
    testset        = preprocess_testset(testset, shop_ids_pairs)
    shops          = preprocess_shops(shops)
    categories     = preprocess_categories(categories, categories_ids_drop)
    items          = preprocess_items(items, categories_ids_drop)

    pickle.dump(testset, open('test_preprocessed.pickle', 'wb'), protocol = 4)
    pickle.dump(shops, open('shops_preprocessed.pickle', 'wb'), protocol = 4)
    pickle.dump(items, open('items_preprocessed.pickle', 'wb'), protocol = 4)
    pickle.dump(categories, open('categories_preprocessed.pickle', 'wb'), protocol = 4)
    pickle.dump(sales_trainset, open('sales_train_preprocessed.pickle', 'wb'), protocol = 4)


if __name__ == "__main__":
    main()
