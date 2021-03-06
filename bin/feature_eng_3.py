#!/usr/bin/env python3

import pickle
import pandas as pd

"""
	FILES IN: 
		'main_data_feature_eng_1.pickle'
		'categories_feature_eng_0.pickle'
		'shops_feature_eng_0.pickle'
		'items_feature_eng_0.pickle'

	FILES OUT: 
		'main_data_feature_eng_3.pickle'

 """

def add_lag_feature(df, lags, main_cols, cols):
    for col in cols:
        current_cols = main_cols + [col]
        tmp = df[current_cols]
        for i in lags:
            shifted = tmp.copy()
            shifted.columns = main_cols + [ col + "_lag_"+str(i)]
            shifted.date_block_num = shifted.date_block_num + i
            df = pd.merge(df, shifted, on=main_cols, how="left")
    return df


def create_lag_features_from_aggregations(df, main_cols, group_by_date, aggregation_op_dict, result_feature, lags):
    group = df.groupby( group_by_date).agg(aggregation_op_dict )
    group.columns = result_feature
    group.reset_index(inplace = True)
    df = pd.merge(df, group, on = group_by_date, how = "left")
    df = add_lag_feature( df, lags, main_cols, result_feature )
    df.drop( result_feature, axis = 1, inplace = True )
    
    return df


def create_all_cat_shop_item_lags(main_data, categories, shops, items):
    aggregation_mean_dict   = {"item_cnt_month": ["mean"]}
    main_cols               = ["date_block_num", "shop_id", "item_id"]
    items.drop(columns=["item_name_p2_id", "item_name_p3_id", "item_category_id"], inplace = True)
    main_data = pd.merge(main_data, items, on=["item_id"], how="left")
    main_data = pd.merge(main_data, categories, on=["item_category_id"], how="left")
    main_data = pd.merge(main_data, shops, on=["shop_id"], how ="left")
	
    group_by_date_shop_broad_cat   = ["date_block_num", "shop_id", "item_broader_category_id"]
    group_by_date_shop_type        = ["date_block_num", "shop_location_id"]
    group_by_date_item_shop_type   = ["date_block_num", "item_id", "shop_location_id"]

    avg_item_shop_location_item_feature = ["date_item_city_avg_item_cnt"]
    avg_item_shop_location_feature = ["date_city_avg_item_cnt"]	
    avg_item_broad_cat_feature     = ["date_shop_broad_avg_item_cnt"]
	
    lags_10   = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    lags_1    = [1]

    # LAG on sales of broader category sales per SHOP
    main_data = create_lag_features_from_aggregations(main_data,main_cols,group_by_date_shop_broad_cat,aggregation_mean_dict, avg_item_broad_cat_feature, lags_1)
    # LAG on sales of by month per SHOP location (city)
    main_data = create_lag_features_from_aggregations(main_data,main_cols,group_by_date_shop_type, aggregation_mean_dict,avg_item_shop_location_feature, lags_1)
    # LAG on sales of each item by month per SHOP location (city)
    main_data = create_lag_features_from_aggregations(main_data,main_cols,group_by_date_item_shop_type,aggregation_mean_dict,avg_item_shop_location_item_feature, lags_1)
    
    return main_data 


def create_mean_feature(data, group_features, mean_feature_name, aggregation_op_dict):
    new_df = data.groupby(group_features).agg(aggregation_op_dict)
    new_df.columns = mean_feature_name 
    data = pd.merge(data, new_df, on=group_features, how="left")
    
    return data


def create_lag_feature_drop(df, lags, main_cols, cols):
    for col in cols:
        current_cols = main_cols + [col]
        tmp = df[current_cols]
        for i in lags:
            shifted = tmp.copy()
            shifted.columns = main_cols + [ col + "_lag_"+str(i)]
            shifted.date_block_num = shifted.date_block_num + i
            df = pd.merge(df, shifted, on=main_cols, how ="left")
    df.drop( cols, axis = 1, inplace = True )
    
    return df


def create_all_mean_lags(main_data):
    
    aggregation_mean_dict     = {"item_cnt_month" : ["mean"] }
    main_cols                 = ["date_block_num", "shop_id", "item_id"]   
    lags_1                    = [1]    
    
    group_by_shopcat_item     = ["date_block_num", "shop_category_id", "item_id"]     
    mean_feature_shopcat_item = ["month_shopcat_item_cnt_mean"]   
    main_data  = create_mean_feature(main_data ,group_by_shopcat_item ,mean_feature_shopcat_item ,aggregation_mean_dict)
    main_data  = create_lag_feature_drop(main_data, lags_1, main_cols, mean_feature_shopcat_item  )

    group_by_shopcat     = ["date_block_num", "shop_category_id"]
    mean_feature_shopcat = ["month_shopcat_cnt_mean"]
    main_data = create_mean_feature(main_data, group_by_shopcat, mean_feature_shopcat, aggregation_mean_dict)
    main_data = create_lag_feature_drop( main_data, lags_1, main_cols,  mean_feature_shopcat)
 
    group_by_shop_itemcat = ["date_block_num", "shop_id", "item_category_id"]
    mean_feature_shop_cat = ["month_shop_cat_cnt_mean"]
    main_data = create_mean_feature(main_data,group_by_shop_itemcat, mean_feature_shop_cat,aggregation_mean_dict )
    main_data = create_lag_feature_drop( main_data, lags_1, main_cols, mean_feature_shop_cat )
    
    group_by_shop_gencatshop     = ["date_block_num", "shop_id", "item_general_category_id"]
    mean_feature_shop_gencatshop = ["month_shop_shopcat_cnt_mean"]
    main_data = create_mean_feature(main_data, group_by_shop_gencatshop, mean_feature_shop_gencatshop ,aggregation_mean_dict )
    main_data = create_lag_feature_drop( main_data, lags_1, main_cols, mean_feature_shop_gencatshop )

    group_by_month_itemcat = ["date_block_num", "item_category_id"]
    mean_feature_shop_cat  = ["month_cat_cnt_mean"]
    main_data = create_mean_feature(main_data,group_by_month_itemcat, mean_feature_shop_cat, aggregation_mean_dict )
    main_data = create_lag_feature_drop( main_data, lags_1, main_cols,mean_feature_shop_cat )
    
    group_by_month_broad_itemcat = ["date_block_num", "item_broader_category_id"]
    mean_feature_broad_itemcat   = ["month_broad_itemcat_cnt_mean"]
    main_data = create_mean_feature(main_data,group_by_month_broad_itemcat, mean_feature_broad_itemcat, aggregation_mean_dict )
    main_data = create_lag_feature_drop(main_data, lags_1, main_cols, mean_feature_broad_itemcat) 
    
    
    group_by_month_gen_itemcat     = ["date_block_num", "item_general_category_id"]
    mean_feature_month_gen_itemcat = ["month_type_cnt_mean"]
    main_data = create_mean_feature(main_data, group_by_month_gen_itemcat, mean_feature_month_gen_itemcat,aggregation_mean_dict )
    main_data = create_lag_feature_drop(main_data, lags_1, main_cols,mean_feature_month_gen_itemcat)
    
    return main_data


def main():
	train      = pd.read_pickle("main_data_feature_eng_1.pickle")
	categories = pd.read_pickle("categories_feature_eng_0.pickle")
	shops      = pd.read_pickle("shops_feature_eng_0.pickle")
	items      = pd.read_pickle("items_feature_eng_0.pickle")

	main_data  = create_all_cat_shop_item_lags(train, categories, shops, items)
	main_data  = create_all_mean_lags(main_data)

	pickle.dump(main_data, open("main_data_feature_eng_3.pickle", "wb"), protocol=4)


if __name__ == "__main__":
    main()
