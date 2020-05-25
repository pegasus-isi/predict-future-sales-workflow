import pandas as pd
import numpy as np
import pickle

"""
Creates 5 basic lag features for the data:
LAG ON ITEM COUNT                     "item_cnt_month"
LAG ON ITEMS AVERAGE COUNT            "date_avg_item_cnt"	
LAG ON Specific ITEM AVERAGE COUNT    "date_item_avg_item_cnt"
LAG on sales of any items per SHOP    "date_shop_avg_item_cnt"
LAG on sales of SPEC items per SHOP   "date_shop_item_avg_item_cnt"

	
	FILES IN: 
		'main_data_feature_eng_1.pickle'

	FILES OUT: 
		'main_data_feature_eng_2.pickle'

 """


# -----------------           HELPER  FUNCTIONS       -------------------------

def add_lag_feature( df,lags,main_cols, cols ):
    for col in cols:
        current_cols = main_cols + [col]
        tmp = df[current_cols]
        for i in lags:
            shifted = tmp.copy()
            shifted.columns = main_cols + [ col + "_lag_"+str(i)]
            shifted.date_block_num = shifted.date_block_num + i
            df = pd.merge(df, shifted, on=main_cols, how ='left')
    return df


def create_lag_features_from_aggregations(df,main_cols,group_by_date,aggregation_op_dict,result_feature,lags):
    
    group = df.groupby( group_by_date).agg(aggregation_op_dict )
    group.columns = result_feature
    group.reset_index(inplace = True)
    df = pd.merge(df, group, on = group_by_date, how = "left")
    df = add_lag_feature( df, lags, main_cols, result_feature )
    df.drop( result_feature, axis = 1, inplace = True )
    
    return df

# LAG ON ITEM COUNT

def create_all_basic_item_lags(main_data):

	aggregation_mean_dict   = {"item_cnt_month" : ["mean"] }
	main_cols               = ["date_block_num", "shop_id","item_id"]
	
	group_by_date           = ["date_block_num"]	
	group_by_date_item      = ["date_block_num",'item_id']	
	group_by_date_shop      = ["date_block_num",'shop_id']
	group_by_date_shop_item = ["date_block_num",'shop_id', "item_id"]	

	item_cnt_feature      = ["item_cnt_month"]
	avg_item_feature      = ["date_avg_item_cnt"]	
	avg_spec_item_feature = ['date_item_avg_item_cnt']	
	avg_shop_sell_feature = ["date_shop_avg_item_cnt"]
	avg_spec_item_shop_feature = ["date_shop_item_avg_item_cnt"]
	
	lags_10   = [1,2,3,4,5,6,7,8,9,10]
	lags_1    = [1]

	# LAG ON ITEM COUNT
	main_data = add_lag_feature(main_data, lags_10, main_cols, item_cnt_feature ) 
	# LAG ON ITEMS AVERAGE COUNT	
	main_data = create_lag_features_from_aggregations(main_data, main_cols, group_by_date, aggregation_mean_dict, avg_item_feature, lags_1)
	# LAG ON Specific ITEM AVERAGE COUNT
	main_data = create_lag_features_from_aggregations(main_data, main_cols, group_by_date_item,aggregation_mean_dict,avg_spec_item_feature,lags_10)
	# LAG on sales of any items per SHOP
	main_data = create_lag_features_from_aggregations(main_data, main_cols, group_by_date_shop,aggregation_mean_dict,avg_shop_sell_feature,lags_10)
	# LAG on sales of SPEC items per SHOP
	main_data = create_lag_features_from_aggregations(main_data, main_cols, group_by_date_shop_item,aggregation_mean_dict,avg_spec_item_shop_feature,lags_10)
	return main_data 

def main():
	train      = pd.read_pickle('main_data_feature_eng_1.pickle')
	main_data  = create_all_basic_item_lags(train)
	pickle.dump(main_data, open('main_data_feature_eng_2.pickle', 'wb'), protocol = 4)


if __name__ == "__main__":
    main()