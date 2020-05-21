import pandas as pd
import numpy as np
import pickle
from itertools import product
from IPython import embed

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
		'train_preprocessed.pickle',
		'test_preprocessed.pickle'

 """

# -----------------           HELPER  FUNCTIONS       -------------------------

def dataframe_setup(train):
	
	data_matrix = []
	cols        = ["date_block_num", "shop_id", "item_id"]
	for i in range(34):
		sales = train[train.date_block_num == i]
		data_matrix.append( np.array(list( product( [i], sales.shop_id.unique(), sales.item_id.unique() ) ), dtype = np.int16) )

	data_matrix                   = pd.DataFrame( np.vstack(data_matrix), columns = cols )
	data_matrix["date_block_num"] = data_matrix["date_block_num"].astype(np.int8)
	data_matrixmatrix["shop_id"]  = data_matrix["shop_id"].astype(np.int8)
	data_matrix["item_id"]        = data_matrix["item_id"].astype(np.int16)
	data_matrix.sort_values( cols, inplace = True )

	return data_matrix

def add_date_block(test, data):
	test["date_block_num"] = 34
	test["date_block_num"] = test["date_block_num"].astype(np.int8)
	test["shop_id"] = test.shop_id.astype(np.int8)
	test["item_id"] = test.item_id.astype(np.int16)
	data = pd.concat([data, test.drop(["ID"],axis = 1)], ignore_index=True, sort=False, keys=cols)
	data.fillna( 0, inplace = True )
	return test, data


def monthly_sales_count(train, data):

	group         = train.groupby( ["date_block_num", "shop_id", "item_id"] ).agg( {"item_cnt_day": ["sum"]} )
	group.columns = ["item_cnt_month"]
	group.reset_index( inplace = True)
	data = pd.merge( data, group, on = cols, how = "left" )
	data["item_cnt_month"] = data["item_cnt_month"].fillna(0).clip(0,20).astype(np.float16)

	return data

def main():

    # Read in the data for the analysis
	items      = pd.read_pickle('items_preprocessed.pickle')
	categories = pd.read_pickle('categories_preprocessed.pickle')
	shops      = pd.read_pickle('shops_preprocessed.pickle')
	train      = pd.read_pickle('sales_train_preprocessed.pickle')
	test       = pd.read_pickle('test_preprocessed.pickle')

	main_data  = dataframe_setup(train)
	main_data  = monthly_sales_count(train, mian_data)
	train["revenue"] = train["item_cnt_day"] * train["item_price"]

	test, main_data = add_date_block(train,main_data)



if __name__ == "__main__":
    main()