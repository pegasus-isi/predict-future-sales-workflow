#!/usr/bin/env python3

import pickle
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from argparse import ArgumentParser

plt.style.use('ggplot')

"""
Tunes hyperparameres for a set of features.
	
	FILES IN: 
            'main_data_feature_eng_all.pickle'
	FILES OUT: 
            '{prefix}_feature_importance.pdf'
            '{prefix}_model.pickle'
"""

# -----------------           HELPER  FUNCTIONS       -------------------------

X_train = None
Y_train = None
X_valid = None
Y_valid = None
gpu_id = None
tree_method = None
early_stopping_rounds = None

def train(params):
    # Instantiate the classifier
    model = xgb.XGBRegressor(
                    n_estimators=1000,
                    max_depth=int(params["max_depth"]),
                    learning_rate=0.1,
                    gamma=params["gamma"],
                    min_child_weight=params["min_child_weight"],
                    subsample=params["subsample"],
                    colsample_bytree=params["colsample_bytree"],
                    reg_lambda=params["reg_lambda"],
                    objective="reg:squarederror",
                    tree_method=tree_method,
                    gpu_id=gpu_id,
                    random_state=42
            )

    # Fit the classsifier
    model.fit(
        X_train,
        Y_train,
        eval_metric="rmse",
        eval_set=[(X_train, Y_train), (X_valid, Y_valid)],
        early_stopping_rounds=early_stopping_rounds,
        verbose=False
    )

    return model


def load_data(filename, validation_months, column_filter):
    global X_train, X_valid, Y_train, Y_valid

    train_data = pd.read_pickle(filename)
    train_data = train_data[column_filter]

    X_train = train_data[train_data.date_block_num < (34 - validation_months)].drop(["item_cnt_month"], axis=1)
    Y_train = train_data[train_data.date_block_num < (34 - validation_months)]["item_cnt_month"]
    X_valid = train_data[train_data.date_block_num == (34 - validation_months)].drop(["item_cnt_month"], axis=1)
    Y_valid = train_data[train_data.date_block_num == (34 - validation_months)]["item_cnt_month"]
    
    return


def plot_feature_importance(prefix, model):
    plt.ioff()
    fig, ax = plt.subplots(1,1,figsize=(16,8))
    xgb.plot_importance(booster=model, ax=ax)
    plt.savefig(f"{prefix}_feature_importance.pdf")
    plt.close(fig)

    return
    

def main():
    global tree_method, gpu_id, early_stopping_rounds

    parser = ArgumentParser(description="XGBoost Model Train")
    parser.add_argument("--file", metavar="STR", type=str, help="File with training and validation data", required=True)
    parser.add_argument("--params", metavar="STR", type=str, help="JSON configuarion file containing xgboost params and list of columns", required=True)
    parser.add_argument("--early_stopping_rounds", metavar="INT", type=int, default=5, help="XGBoost early stopping rounds", required=False)
    parser.add_argument("--validation_months", metavar="INT", type=int, default=1, help="Number of trialing months for validation", required=False)
    parser.add_argument("--tree_method", metavar="STR", type=str, default="hist", help="XGBoost tree method", choices=["hist", "gpu_hist"], required=False)
    parser.add_argument("--gpu_id", metavar="INT", type=int, default=0, help="XGBoost target gpu id", required=False)
    parser.add_argument("--output", metavar="STR", type=str, default="best_parameters.json", help="Output file", required=False)

    args = parser.parse_args()

    gpu_id = args.gpu_id
    tree_method = args.tree_method
    early_stopping_rounds = args.early_stopping_rounds
    
    params = json.load(open(args.params, 'r'))
    load_data(args.file, args.validation_months, params["columns"])

    model = train(params["best_config"])
    
    prefix = args.file[:args.file.find(".")]

    plot_feature_importance(prefix, model)
    pickle.dump(model, open(f"{prefix}_model.pickle", "wb"), protocol=4)
    
if __name__ == "__main__":
    main()
