#!/usr/bin/env python3

import os
import json
from argparse import ArgumentParser

"""
Finds best hyperparameters for a group of records.
	
	FILES IN: 
            '{prefix}*.json'
	FILES OUT: 
            'best_parameters.json'
"""

# -----------------           HELPER  FUNCTIONS       -------------------------
def find_best_params(directory, prefix):
    best_params = None

    for filename in os.listdir(directory):
        if filename.startswith(prefix) and filename.endswith(".json"):
            temp = json.load(open(filename, "r"))
            if best_params is None:
                best_params = temp
            elif best_params["best_loss"] > temp["best_loss"]:
                best_params = temp

    return best_params


def main():
    parser = ArgumentParser(description="Pick best combination of features and parameters for XGBoost")
    parser.add_argument("--prefix", metavar="STR", type=str, help="Prefix for hyperparameter tuning JSON files", required=True)
    parser.add_argument("--dir", metavar="STR", type=str, default=".", help="Directory that files are located in", required=False)
    parser.add_argument("--output", metavar="STR", type=str, default="best_parameters.json", help="Output file", required=False)

    args = parser.parse_args()

    best_params = find_best_params(args.dir, args.prefix)
    json.dump(best_params, open(args.output, "w"), indent=2)
    
if __name__ == "__main__":
    main()
