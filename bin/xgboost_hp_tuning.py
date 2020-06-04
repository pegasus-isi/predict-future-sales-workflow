#!/usr/bin/env python3

import pickle
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from argparse import ArgumentParser
from sklearn.metrics import mean_squared_error
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

"""
Tunes hyperparameres for a set of features.
	
	FILES IN: 
            'main_data_feature_eng_all.pickle'
	FILES OUT: 
            'parameter_stats.out'
"""

# -----------------           HELPER  FUNCTIONS       -------------------------

X_train = None
Y_train = None
X_valid = None
Y_valid = None
early_stopping_rounds = None

def hyper_objective(space):
    # Instantiate the classifier
    model = xgb.XGBRegressor(
                    n_estimators=1000,
                    max_depth=int(space["max_depth"]),
                    learning_rate=0.1,
                    gamma=space["gamma"],
                    min_child_weight=space["min_child_weight"],
                    subsample=space["subsample"],
                    colsample_bytree=space["colsample_bytree"],
                    reg_lambda=space["reg_lambda"],
                    objective="reg:squarederror",
                    tree_method="gpu_hist",
                    gpu_id=0,
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

    # Predict on Cross Validation data
    Y_pred = model.predict(X_valid)

    # Calculate Root Mean Squared Error of Validation Data
    rms_error = np.sqrt(mean_squared_error(Y_valid, Y_pred))

    return {"loss": rms_error, "status": STATUS_OK }


def run_trials(config, total_trials):
    space = {
        "max_depth": hp.quniform("x_max_depth", config["max_depth"]["low"], config["max_depth"]["high"], 1),
        "min_child_weight": hp.quniform("x_min_child_weight", config["min_child_weight"]["low"], config["min_child_weight"]["high"], 1),
        "subsample": hp.uniform("x_subsample", config["subsample"]["low"], config["subsample"]["high"]),
        "gamma": hp.uniform("x_gamma", config["gamma"]["low"], config["gamma"]["high"]),
        "colsample_bytree": hp.uniform("x_colsample_bytree", config["colsample_bytree"]["low"], config["colsample_bytree"]["high"]),
        "reg_lambda": hp.uniform("x_reg_lambda", config["reg_lambda"]["low"], config["reg_lambda"]["high"]),
    }

    trials = Trials()
    best_config = fmin( fn=hyper_objective,
                        space=space,
                        algo=tpe.suggest,
                        max_evals=total_trials,
                        trials=trials
                    )

    return best_config


def load_data(filename, validation_months):
    global X_train, X_valid, Y_train, Y_valid

    train_data = pd.read_pickle(filename)
    X_train = train_data[train_data.date_block_num < (34 - validation_months)].drop(["item_cnt_month"], axis=1)
    Y_train = train_data[train_data.date_block_num < (34 - validation_months)]["item_cnt_month"]
    X_valid = train_data[train_data.date_block_num == (34 - validation_months)].drop(["item_cnt_month"], axis=1)
    Y_valid = train_data[train_data.date_block_num == (34 - validation_months)]["item_cnt_month"]
    return


def main():
    parser = ArgumentParser(description="XGBoost Hyperparameter Tuning based on Hyperopt")
    parser.add_argument("-f", "--file", metavar="STR", type=str, help="File with training and validation data", required=True)
    parser.add_argument("-c", "--conf", metavar="STR", type=str, help="JSON configuarion file containing trial space", required=True)
    parser.add_argument("-t", "--trials", metavar="INT", type=int, default=10, help="Total number of trials", required=False)
    parser.add_argument("-e", "--early_stopping_rounds", metavar="INT", type=int, default=20, help="XGBoost early stopping rounds", required=False)
    parser.add_argument("-m", "--validation_months", metavar="INT", type=int, default=1, help="Number of trialing months for validation", required=False)
    parser.add_argument("-o", "--output", metavar="STR", type=str, default="best_parameters.json", help="Output file", required=False)
    #TODO: Accept a filter for columns to drop or keep
    #TODO: Add argument for gpu (with gpu id maybe?)

    args = parser.parse_args()
    config = json.load(open(args.conf, 'r'))
    early_stopping_rounds = args.early_stopping_rounds

    load_data(args.file, args.validation_months)

    best_config = run_trials(config, args.trials)
    json.dump(best_config, open(args.output, 'w'), indent=2)
    
if __name__ == "__main__":
    main()
