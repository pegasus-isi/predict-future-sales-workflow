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
gpu_id = None
tree_method = None
early_stopping_rounds = None

def hyper_objective(params):
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

    # Predict on Cross Validation data
    Y_pred = model.predict(X_valid)

    # Calculate Root Mean Squared Error of Validation Data
    rms_error = np.sqrt(mean_squared_error(Y_valid, Y_pred))

    return {"loss": rms_error, "status": STATUS_OK }


def run_trials(config, max_trials):
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
                        max_evals=max_trials,
                        trials=trials,
                        show_progressbar=False
                    )
    
    return trials.best_trial["result"]["loss"], best_config


def load_data(filename, validation_months, column_filter):
    global X_train, X_valid, Y_train, Y_valid

    train_data = pd.read_pickle(filename)
    if column_filter != [-1]:
        train_data_columns = list(train_data.columns)
        train_data_columns_filtered = [train_data_columns[i] for i in column_filter]
        train_data = train_data[train_data_columns_filtered]
    else:
        train_data_columns_filtered = list(train_data.columns)

    X_train = train_data[train_data.date_block_num < (34 - validation_months)].drop(["item_cnt_month"], axis=1)
    Y_train = train_data[train_data.date_block_num < (34 - validation_months)]["item_cnt_month"]
    X_valid = train_data[train_data.date_block_num == (34 - validation_months)].drop(["item_cnt_month"], axis=1)
    Y_valid = train_data[train_data.date_block_num == (34 - validation_months)]["item_cnt_month"]
    
    return train_data_columns_filtered


def main():
    global tree_method, gpu_id, early_stopping_rounds

    parser = ArgumentParser(description="XGBoost Hyperparameter Tuning based on Hyperopt")
    parser.add_argument("--file", metavar="STR", type=str, help="File with training and validation data", required=True)
    parser.add_argument("--space", metavar="STR", type=str, help="JSON configuarion file containing trial space", required=True)
    parser.add_argument("--trials", metavar="INT", type=int, default=10, help="Total number of trials", required=False)
    parser.add_argument("--early_stopping_rounds", metavar="INT", type=int, default=20, help="XGBoost early stopping rounds", required=False)
    parser.add_argument("--validation_months", metavar="INT", type=int, default=1, help="Number of trialing months for validation", required=False)
    parser.add_argument("--tree_method", metavar="STR", type=str, default="hist", help="XGBoost tree method", choices=["hist", "gpu_hist"], required=False)
    parser.add_argument("--gpu_id", metavar="INT", type=int, default=0, help="XGBoost target gpu id", required=False)
    parser.add_argument("--col_filter", metavar="INT", type=int, default=[-1], nargs="+", help="Column filter", required=False)
    parser.add_argument("--output", metavar="STR", type=str, default="best_parameters.json", help="Output file", required=False)

    args = parser.parse_args()

    gpu_id = args.gpu_id
    tree_method = args.tree_method
    early_stopping_rounds = args.early_stopping_rounds
    
    space = json.load(open(args.space, 'r'))

    column_filter = args.col_filter
    train_data_columns = load_data(args.file, args.validation_months, column_filter)

    best_loss, best_config = run_trials(space, args.trials)
    
    best_params={"best_loss": best_loss, "best_config": best_config, "columns": train_data_columns}
    json.dump(best_params, open(args.output, 'w'), indent=2)
    
if __name__ == "__main__":
    main()
