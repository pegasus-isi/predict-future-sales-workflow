#!/usr/bin/env python3

import pickle
import pandas as pd

"""
	FILES IN: 
                'sales_train.csv'
                'items_feature_eng_0.pickle'
                'shops_feature_eng_0.pickle'
                'main_data_feature_eng_1.pickle'
                'categories_feature_eng_0.pickle'

	FILES OUT: 
                'main_data_feature_eng_5.pickle'
 """

def create_price_lag_feature(main_data, feature_name, group_by_features, train, aggragation_opt_dict):   
    group = train.groupby(group_by_features).agg(aggragation_opt_dict)
    group.columns = feature_name
    group.reset_index(inplace=True)   
    main_data = main_data.merge(group, on=group_by_features, how="left")    
    
    return main_data


def add_lag_feature(df, lags, main_cols, cols):
    for col in cols:
        current_cols = main_cols + [col]
        tmp = df[current_cols]
        for i in lags:
            shifted = tmp.copy()
            shifted.columns = main_cols + [col + "_lag_" + str(i)]
            shifted.date_block_num = shifted.date_block_num + i
            df = pd.merge(df, shifted, on=main_cols, how ="left")    
    
    return df


def create_single_delta_lag(flag, matrix, lags, features_to_drop, delta_feature_name, base_feature_name):
    def select_trends1(row) :
        for i in lags:
            if row["delta_price_lag_" + str(i)]:
                return row["delta_price_lag_" + str(i)]
        return 0

    def select_trends2(row) :
        for i in lags:
            if row["delta_shop_price_lag_" + str(i)]:
                return row["delta_shop_price_lag_" + str(i)]
        return 0

    if flag == 1:
        matrix[delta_feature_name] = matrix.apply(select_trends1, axis=1)
    else:
        matrix[delta_feature_name] = matrix.apply(select_trends2, axis=1)        
    
    matrix[delta_feature_name].fillna(0, inplace=True)

    for i in lags:
        features_to_drop.append(base_feature_name + "_lag_" + str(i))
        features_to_drop.append(delta_feature_name + "_" + str(i))

    matrix.drop(features_to_drop, axis=1, inplace=True)
    
    return matrix


def create_delta_lag_features(flag, matrix, lags, main_cols, delta_feature_name, base_feature_name, overave_feature_name):   
    matrix = add_lag_feature(matrix, lags, main_cols, [base_feature_name])
    
    for i in lags:
        matrix[delta_feature_name + "_" + str(i)] = \
        (matrix[base_feature_name + "_lag_" + str(i)] - matrix[overave_feature_name]) / matrix[overave_feature_name]
    
    features_to_drop = [overave_feature_name, base_feature_name]
    matrix           = create_single_delta_lag(flag, matrix, lags, features_to_drop, delta_feature_name, base_feature_name)    
    
    return matrix


def create_all_delta_features(main_data):
    lags_3 = [1, 2, 3]
    lags_1 = [1]
    main_cols  = ["date_block_num", "shop_id", "item_id"]

    main_data = create_delta_lag_features(1, main_data, lags_3, main_cols, "delta_price_lag", "date_item_avg_item_price", "item_avg_item_price")    
    main_data = create_delta_lag_features(2, main_data, lags_3, main_cols, "delta_shop_price_lag", "date_shop_item_avg_item_price", "shop_item_avg_item_price") 
    main_data["delta_revenue"] = (main_data["date_shop_revenue"] - main_data["shop_avg_revenue"]) / main_data["shop_avg_revenue"]
    main_data                  = add_lag_feature(main_data, lags_1, main_cols, ["delta_revenue"])
    main_data.drop(["date_shop_revenue", "shop_avg_revenue", "delta_revenue"], axis=1, inplace=True)
    
    return main_data


def create_all_price_related_lags(main_data, train):
    
    train["revenue"]          = train["item_cnt_day"] * train["item_price"]

    lags_3     = [1, 2, 3]
    lags_1     = [1]
    
    agg_dict_itemp_mean      = {"item_price": ["mean"]}   
    agg_dict_revenue_sum     = {"revenue": ["sum"]}     
    agg_dict_date_mean       = {"date_block_num":["mean"]}      
    
    group_by_item            = ["item_id"]
    group_by_shop            = ["shop_id"]
    group_by_month_item      = ["date_block_num","item_id"]
    group_by_month_shop      = ["date_block_num","shop_id"]
    group_by_shop_item       = ["shop_id", "item_id"]   
    group_by_month_shop_item = ["date_block_num", "shop_id", "item_id"]    
    
    price_feature_item_ave    = ["item_avg_item_price"]
    price_feature_month_ave   = ["date_item_avg_item_price"]
    price_feature_shop_item   = ["shop_item_avg_item_price"]
    revenue_feature_date_shop = ["date_shop_revenue"]    
    revenue_feature_shop      = ["shop_avg_revenue"]    
    price_feature_month_shop_item = ["date_shop_item_avg_item_price"]
    
    main_data = create_price_lag_feature(main_data, price_feature_item_ave, group_by_item, train, agg_dict_itemp_mean)
    main_data = create_price_lag_feature(main_data, price_feature_month_ave, group_by_month_item, train, agg_dict_itemp_mean)       
    main_data = create_price_lag_feature(main_data, revenue_feature_date_shop, group_by_month_shop, train, agg_dict_revenue_sum)    
    main_data = create_price_lag_feature(main_data, revenue_feature_shop, group_by_shop, train, agg_dict_date_mean)     
    main_data = create_price_lag_feature(main_data, price_feature_shop_item, group_by_shop_item, train, agg_dict_itemp_mean)
    main_data = create_price_lag_feature(main_data, price_feature_month_shop_item, group_by_month_shop_item, train, agg_dict_itemp_mean)  
         
    return main_data
    
def main():
    org_train  = pd.read_csv("sales_train.csv")
    items      = pd.read_pickle("items_feature_eng_0.pickle")
    shops      = pd.read_pickle("shops_feature_eng_0.pickle")
    categories = pd.read_pickle("categories_feature_eng_0.pickle")
    main_data  = pd.read_pickle("main_data_feature_eng_1.pickle")

    main_data = create_all_price_related_lags(main_data, org_train)
    main_data = create_all_delta_features(main_data)
    pickle.dump(main_data, open("main_data_feature_eng_5.pickle", "wb"), protocol=4)


if __name__ == "__main__":
    main()
