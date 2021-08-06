#!/usr/bin/env python3

import pickle
import json
import pandas as pd
from argparse import ArgumentParser

"""
Produces predictions.
	
	FILES IN: 
            'main_data_feature_eng_all.pickle'
            'xgboost_model_params.json'

	FILES OUT: 
            '{prefix}_predictions.pickle'
"""

# -----------------           HELPER  FUNCTIONS       -------------------------

def create_predictions(filename, column_filter, model):
    test_data = pd.read_pickle(filename)
    test_data = test_data[column_filter]

    X_test = test_data[test_data["date_block_num"] == 34].drop(["item_cnt_month"], axis=1)
    Y_test = model.predict(X_test).clip(0, 20)

    Y_pred = data.loc[data['date_block_num']==34, ['shop_id','item_id']]
    Y_pred['prediction'] = Y_test

    return Y_pred


def main():
    global tree_method, gpu_id, early_stopping_rounds

    parser = ArgumentParser(description="XGBoost Model Predict")
    parser.add_argument("--file", metavar="STR", type=str, help="File with preprocessed TEST data", required=True)
    parser.add_argument("--params", metavar="STR", type=str, help="JSON configuarion file containing xgboost params and list of columns", required=True)
    parser.add_argument("--model", metavar="STR", type=str, help="File containing model trained with XGBoost", required=True)

    args = parser.parse_args()

    model = pickle.load(args.model)
    params = json.load(open(args.params, 'r'))
    predictions = create_predictions(args.file, params["columns"], model)
    
    prefix = args.file[:args.file.find(".")]
    pickle.dump(predictions, open(f"{prefix}_predictions.pickle", "wb"), protocol=4)
    
if __name__ == "__main__":
    main()
