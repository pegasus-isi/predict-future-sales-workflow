#!/usr/bin/env python3
import pandas as pd
import pickle

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
    holidays_by_month = holidays[["date_block_num","holiday_date"]].groupby(["date_block_num"]).count().reset_index()
    holidays_by_month = holidays_by_month.rename(columns={"holiday_date": "holidays_cnt_month"})
    
    return holidays_by_month


def create_holiday_lag_features(holidays, lags):
    for i in lags:
        tmp = holidays[["date_block_num", "holidays_cnt_month"]].copy()
        tmp["date_block_num"] = tmp["date_block_num"].apply(lambda x: x-i)
        tmp.columns = ["date_block_num", "holidays_cnt_month_fut_"+str(i)]
        holidays = pd.merge(holidays, tmp, on=["date_block_num"], how="left")
    
    holidays = holidays.fillna(0)

    return holidays


def create_first_sale_features(main_data):
    main_data["month"] = main_data["date_block_num"] % 12
    days = pd.Series([31,28,31,30,31,30,31,31,30,31,30,31])
    main_data["item_shop_first_sale"]= main_data["date_block_num"] - main_data.groupby(["item_id", "shop_id"])["date_block_num"].transform("min")
    main_data["item_first_sale"]     = main_data["date_block_num"] - main_data.groupby(["item_id"])["date_block_num"].transform("min")

    return main_data    


def create_all_holiday_features(train, holidays):
    lags_3              = [1,2,3]
    holidays_by_month   = group_holidays_by_month(holidays)
    holidays_with_lags  = create_holiday_lag_features(holidays_by_month, lags_3)
    main_data = pd.merge(train, holidays_with_lags, on=["date_block_num"], how="left")

    #TODO: maybe we need to fillna again after the merge....
    
    return main_data


def main():
    holidays    = pd.read_csv("holidays.csv")
    train       = pd.read_pickle("main_data_feature_eng_1.pickle")
    main_data   = create_holiday_lags(train, holidays)
    main_data   = create_first_sale_features(main_data)
    pickle.dump(main_data, open("main_data_feature_eng_4.pickle", "wb"), protocol = 4)


if __name__ == "__main__":
    main()
