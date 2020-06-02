#!/usr/bin/env python3
import pandas as pd
import pickle
import numpy as np
from IPython import embed
"""
Creates holiday features with 3 lags from future months.
	
	FILES IN: 
                'holidays.csv'
		'main_data_feature_eng_1.pickle'

	FILES OUT: 
		'main_data_feature_eng_4.pickle'

"""

# -----------------           HELPER  FUNCTIONS       -------------------------

def group_holidays_by_month(holidays):
    holidays_by_month = holidays[["date_block_num", "holiday_date"]].groupby(["date_block_num"]).count().reset_index()
    holidays_by_month = holidays_by_month.rename(columns={"holiday_date": "holidays_cnt_month"})
    
    return holidays_by_month


def create_holiday_lag_features(holidays, lags):
    for i in lags:
        tmp = holidays[["date_block_num", "holidays_cnt_month"]].copy()
        tmp["date_block_num"] = tmp["date_block_num"].apply(lambda x: x-i)
        tmp.columns = ["date_block_num", "holidays_cnt_month_fut_" + str(i)]
        holidays = pd.merge(holidays, tmp, on=["date_block_num"], how="left")
    
    holidays = holidays.fillna(0)

    return holidays


def create_first_sale_features(main_data):
    main_data["month"] = main_data["date_block_num"] % 12
    days = pd.Series([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    main_data["item_shop_first_sale"] = main_data["date_block_num"] - main_data.groupby(["item_id", "shop_id"])["date_block_num"].transform("min")
    main_data["item_first_sale"]      = main_data["date_block_num"] - main_data.groupby(["item_id"])["date_block_num"].transform("min")

    return main_data    


def create_all_holiday_features(train, holidays):
    lags_3              = [1, 2, 3]
    holidays_by_month   = group_holidays_by_month(holidays)
    holidays_with_lags  = create_holiday_lag_features(holidays_by_month, lags_3)
    main_data = pd.merge(train, holidays_with_lags, on=["date_block_num"], how="left")

    #TODO: maybe we need to fillna again after the merge....
    
    return main_data


def create_featrues_last_sale_shop(matrix, train):
    tmp_list = []
    for mid in matrix["date_block_num"].unique():
        tmp = train.loc[train["date_block_num"] < mid, ["date_block_num", "shop_id", "item_id"]].groupby(["shop_id", "item_id"]).last().rename({"date_block_num": "item_month_id_of_last_sale_in_shop"}, axis=1).astype(np.int16)
        tmp["date_block_num"] = mid
        tmp.reset_index(inplace=True)
        tmp_list.append(tmp) 
    tmp = pd.concat(tmp_list)

    matrix = matrix.join(tmp.set_index(["date_block_num", "shop_id", "item_id"]), on=["date_block_num", "shop_id", "item_id"])
    # time since last sale in shop
    matrix["item_months_since_last_sale_in_shop"] = matrix["date_block_num"] - matrix["item_month_id_of_last_sale_in_shop"]

    return matrix


def create_featrues_month_last_sale(matrix, train):
    tmp_list = []
    for mid in matrix["date_block_num"].unique():
        tmp = train.loc[train["date_block_num"] < mid,["date_block_num", "item_id"]].groupby("item_id").last().rename({"date_block_num": "item_month_id_of_last_sale"}, axis=1)
        tmp["date_block_num"] = mid
        tmp.reset_index(inplace=True)
        tmp_list.append(tmp)      
    tmp = pd.concat(tmp_list)

    matrix = matrix.join(tmp.set_index(["date_block_num", "item_id"]), on=["date_block_num", "item_id"])
    # downcast dtype (int not possible due to NaN values)
    matrix["item_month_id_of_last_sale"] = matrix["item_month_id_of_last_sale"].astype(np.float16)
    # time since last sale over all shops
    matrix["item_months_since_last_sale"] = matrix["date_block_num"] - matrix["item_month_id_of_last_sale"]
    # downcast dtype (int not possible due to NaN values)
    matrix["item_month_id_of_last_sale"] = matrix["item_month_id_of_last_sale"].astype(np.float16)
    # time since last sale over all shops
    matrix["item_months_since_last_sale"] = matrix["date_block_num"] - matrix["item_month_id_of_last_sale"]

    return matrix


def create_opening_relase_related_features(matrix, train):
    matrix["shop_months_since_opening"] = matrix["date_block_num"] - matrix["shop_id"].map(matrix[["date_block_num", "shop_id"]].groupby("shop_id").min()["date_block_num"])
    matrix["shop_opening"] = (matrix["shop_months_since_opening"] == 0)
    matrix["item_month_id_of_release"] = matrix["item_id"].map(matrix[["date_block_num", "item_id"]].groupby("item_id").min()["date_block_num"])
    matrix["item_month_of_release"] = matrix["item_month_id_of_release"] % 12 + 1
    matrix["item_months_since_release"] = matrix["date_block_num"] - matrix["item_month_id_of_release"]
    matrix["item_months_since_release"].clip(0, 12, inplace=True)  
    matrix["item_new"] = (matrix["item_months_since_release"] == 0)
    # month where the item has first been sold in shop
    group = train[["date_block_num", "shop_id", "item_id"]].groupby(["shop_id", "item_id"]).min().rename({"date_block_num": "item_month_id_of_first_sale_in_shop"}, axis=1)
    matrix = pd.merge(matrix, group, on=["shop_id", "item_id"], how="left")
    matrix["item_month_of_first_sale_in_shop"] = matrix["item_month_id_of_first_sale_in_shop"] % 12 + 1

    # number of months since the item has been released in this shop
    matrix["item_months_since_first_sale_in_shop"] = matrix["date_block_num"] - matrix["item_month_id_of_first_sale_in_shop"]
    matrix["item_months_since_first_sale_in_shop"].clip(0, 12, inplace=True)  # group together items sold for more than a year in the shop

    # whether the item has already been sold in this shop before or not
    matrix["item_never_sold_in_shop_before"] =~ (matrix["item_months_since_first_sale_in_shop"] > 0)

    # set month of release and number of months since the item has been released in this shop to -1 for all items never sold in shop before (remove info from future)
    matrix.loc[matrix["item_never_sold_in_shop_before"],"item_months_since_first_sale_in_shop"] =- 1
    matrix.loc[matrix["item_never_sold_in_shop_before"],"item_month_id_of_first_sale_in_shop"] =- 1
    matrix.loc[matrix["item_never_sold_in_shop_before"],"item_month_of_first_sale_in_shop"] =- 1

    matrix["item_never_sold_in_shop_before_but_not_new"] = ((1 - matrix["item_new"]) * matrix["item_never_sold_in_shop_before"]).astype(bool)
    matrix["item_seniority"] = (2 - matrix["item_new"].astype(int) - matrix["item_never_sold_in_shop_before"].astype(int)).astype(np.int8)
    matrix["item_sold_in_shop"] = (matrix["item_cnt_month"] > 0)
    return matrix


def main():
    holidays    = pd.read_csv("holidays.csv")
    train       = pd.read_pickle("main_data_feature_eng_1.pickle")
    main_data   = create_all_holiday_features(train, holidays)
    main_data   = create_first_sale_features(main_data)
    main_data   = create_opening_relase_related_features(main_data, train)
    main_data   = create_featrues_last_sale_shop(main_data, train)
    mian_data   = create_featrues_month_last_sale(main_data, train)
    pickle.dump(main_data, open("main_data_feature_eng_4.pickle", "wb"), protocol=4)


if __name__ == "__main__":
    main()
