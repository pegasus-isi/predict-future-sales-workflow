import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from IPython import embed
"""
Preprocesses data for the Future Sales Predictions

	FILES IN: "sales_train.csv",
	          "items.csv",
	          "item_categories.csv",
			  "test.csv",
			  "shops_translated_new.csv"

	FILES OUT: 'items_preprocessed.pickle',
	           'shops_preprocessed.pickle',
			   'categories_preprocessed.pickle',
			   'train_preprocessed.pickle',
			   'test_preprocessed.pickle'

 """

def remove_outliners(trainset):
	
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


def preprocess_shops(shops):
	
	#Rename columns
	shops = shops.rename(columns = {"shop_name_translated":"shop_name"})
	shops_ids_to_drop = [0,1,11,40,9,20,33]
	shops             = drop_rows_by_index(shops,shops_ids_to_drop )
	#Create new categories based on shop' s names
	shops["city"]     = shops.shop_name.str.split(" ").map( lambda x: x[0] )
	shops["category"] = shops.shop_name.str.split(" ").map( lambda x: x[1].lower() )

	common_categories = []
	for cat in shops.category.unique():
		if len(shops[shops.category == cat]) > 4:
			common_categories.append(cat)

	shops.category         = shops.category.apply( lambda x: x if (x in common_categories) else "etc" )	
	shops["shop_category"] = LabelEncoder().fit_transform( shops.category )
	shops["shop_city"]     = LabelEncoder().fit_transform( shops.city )
	shops                  = shops[["shop_id", "shop_category", "shop_city"]]

	return shops

def preprocess_trainset(trainset,shop_ids_pairs):

	trainset = remove_outliners(trainset)
	trainset = replace_shop_ids(trainset, shop_ids_pairs,"shop_id")
	shop_ids_drop_train = [9,20,33]
	trainset = drop_rows_by_col_val(trainset,"shop_id",shop_ids_drop_train)
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
    shops            = pd.read_csv("shops_translated_new.csv")
       
    # Replace store ids of duplicated stores
    shop_ids_pairs = [[0,57], [1,58], [11,10],[40,39]]

    
    sales_trainset = preprocess_trainset(sales_trainset, shop_ids_pairs)
    testset        = preprocess_testset(testset,shop_ids_pairs)
    shops          = preprocess_shops(shops)







if __name__ == "__main__":
    main()