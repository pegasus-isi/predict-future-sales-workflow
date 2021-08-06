#!/usr/bin/env python3

import gc
import json
import pickle
import pandas as pd

"""
	FILES IN: 
                'test.csv'
                'test_group_0_predictions.pickle'
                'test_group_1_predictions.pickle'
                'test_group_2_predictions.pickle'

	FILES OUT:
                'item_predictions.csv'
 """


def main():
    gc.enable()
    
    #load test data
    test = pd.read_csv("test.csv")
    
    #load predictions 
    predict_group_0 = pd.read_pickle("test_group_0_predictions.pickle")
    predict_group_1 = pd.read_pickle("test_group_1_predictions.pickle")
    predict_group_2 = pd.read_pickle("test_group_2_predictions.pickle")

    #concat to a single dataframe
    predictions = pd.concat([predict_group_0, predict_group_1, predict_group_2], axis=0, sort=False)


    #join predictions
    test = test.join(predictions.set_index(["shop_id", "item_id"]), on=["shop_id", "item_id"])

    #prepare output
    final_output = pd.DataFrame({
        "ID": test["ID"], 
        "item_cnt_month": test["prediction"]
    })

    #save predictions to csv
    final_output.to_csv("item_predictions.csv", index=False)

if __name__ == "__main__":
    main()
