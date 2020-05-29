import pandas as pd
import numpy as np
import pickle
from itertools import product
from IPython import embed

"""


	FILES IN: 
		'items_preprocessed.pickle'
		'categories_preprocessed.pickle'
		'shops_preprocessed.pickle'
		'sales_train_preprocessed.pickle'
		'test_preprocessed.pickle'

	FILES OUT: 
		'main_data_feature_eng_1.pickle'


 """

# -----------------           HELPER  FUNCTIONS       -------------------------

'''
Extends the training dataframe by adding entries for sales for each month 
of every item an shop combination. 
'''
def dataframe_setup(train,cols):

    data_matrix = []
    for i in range(34):
        sales = train[train.date_block_num == i]
        data_matrix.append( np.array(list( product( [i], sales.shop_id.unique(), sales.item_id.unique() ) ), dtype = np.int16) )

    data_matrix  = pd.DataFrame( np.vstack(data_matrix), columns = cols )
    data_matrix.sort_values( cols, inplace = True )

    return data_matrix

def add_date_block(test, data):
    test["date_block_num"] = 34
    test["date_block_num"] = test["date_block_num"].astype(np.int8)
    return test, data


def monthly_sales_count(train, data,cols):
    group         = train.groupby( cols ).agg( {"item_cnt_day": ["sum"]} )
    group.columns = ["item_cnt_month"]
    group.reset_index( inplace = True)
    data = pd.merge( data, group, on = cols, how = "left" )
    data["item_cnt_month"] = data["item_cnt_month"].fillna(0).clip(0,20).astype(np.float16)

    return data
'''
Creates final version of the main dataframe (in terms of rows)
to which later features (columns) are added 
'''
def merge_dataframes(data, test,cols,categories, items):
    data = pd.concat([data, test.drop(["ID"],axis = 1)], ignore_index=True, sort=False, keys=cols)
    data.fillna( 0, inplace = True )
    data = pd.merge( data, items, on = ["item_id"], how = "left")
    data = pd.merge( data, categories, on = ["item_category_id"], how = "left" )
    data.drop(columns= ["item_name","item_category_name"], inplace = True)

    return data

def main():
    items      = pd.read_pickle('items_preprocessed.pickle')
    categories = pd.read_pickle('categories_preprocessed.pickle')
    train      = pd.read_pickle('sales_train_preprocessed.pickle')
    test       = pd.read_pickle('test_preprocessed.pickle')
    
    cols            = ["date_block_num", "shop_id", "item_id"]
    main_data       = dataframe_setup(train, cols)
    main_data       = monthly_sales_count(train, main_data,cols)
    test, main_data = add_date_block(test,main_data)
    main_data       = merge_dataframes(main_data, test, cols,categories,items)
    pickle.dump(main_data, open('main_data_feature_eng_1.pickle', 'wb'), protocol = 4)


if __name__ == "__main__":
    main()