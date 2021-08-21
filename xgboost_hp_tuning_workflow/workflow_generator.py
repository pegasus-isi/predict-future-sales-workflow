#!/usr/bin/env python3

import os
import json
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser
from itertools import combinations

logging.basicConfig(level=logging.DEBUG)

# --- Import Pegasus API -------------------------------------------------------
from Pegasus.api import *

class predict_future_sales_workflow:
    wf = None
    sc = None
    tc = None
    rc = None
    props = None
    
    daxfile = None
    wf_name = None
    wf_dir = None
    
    is_root_wf = None
    output_single = None
    
    xgb_trials = None
    xgb_early_stopping = None
    xgb_tree_method = None
    xgb_default_cols = None
    xgb_hp_tuning_conf = None

    
    # --- Init ---------------------------------------------------------------------
    def __init__(self, is_root_wf, daxfile, output_single, xgb_args):
        self.daxfile = daxfile
        self.output_single = output_single
        self.xgb_trials = xgb_args["xgb_trials"]
        self.xgb_early_stopping = xgb_args["xgb_early_stopping"]
        self.xgb_tree_method = xgb_args["xgb_tree_method"]
        self.xgb_feat_lens = xgb_args["xgb_feat_lens"]
        self.xgb_data_file = xgb_args["xgb_data_file"]
        self.xgb_cols_file = xgb_args["xgb_cols_file"]
        self.xgb_default_cols = xgb_args["xgb_default_cols"]
        self.xgb_hp_tuning_conf = xgb_args["xgb_hp_tuning_conf"]
        self.is_root_wf = is_root_wf
        
        self.wf_dir = Path(__file__).parent.resolve()
        
        ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
        self.wf_name = "predict-sales-wf-hp-tuning-%s" % ts


    # --- Write files in directory -------------------------------------------------
    def write(self):
        if self.output_single:
            if self.sc: self.wf.add_site_catalog(self.sc)
            if self.rc: self.wf.add_replica_catalog(self.rc)
            if self.tc: self.wf.add_transformation_catalog(self.tc)
        else:
            if self.sc: self.sc.write()
            if self.rc: self.rc.write()
            if self.tc: self.tc.write()

        if self.is_root_wf:
            self.props.write()

        with open(self.daxfile, "w+") as f:
            self.wf.write(f)


    # --- Configuration (Pegasus Properties) ---------------------------------------
    def create_pegasus_properties(self):
        self.props = Properties()

        self.props["pegasus.monitord.encoding"] = "json"                                                                    
        self.props["pegasus.catalog.workflow.amqp.url"] = "amqp://friend:donatedata@msgs.pegasus.isi.edu:5672/prod/workflows"
    
    
    # --- Site Catalog -------------------------------------------------------------
    def create_sites_catalog(self):
        self.sc = SiteCatalog()

        shared_scratch_dir = os.path.join(self.wf_dir, "scratch")
        local_storage_dir = os.path.join(self.wf_dir, "output")

        local = Site("local")\
                    .add_directories(
                        Directory(Directory.SHARED_SCRATCH, shared_scratch_dir)
                            .add_file_servers(FileServer("file://" + shared_scratch_dir, Operation.ALL)),
                        
                        Directory(Directory.LOCAL_STORAGE, local_storage_dir)
                            .add_file_servers(FileServer("file://" + local_storage_dir, Operation.ALL))
                    )

        condorpool = Site("condorpool")\
                        .add_pegasus_profile(style="condor")\
                        .add_condor_profile(universe="vanilla")\
                        .add_profiles(Namespace.PEGASUS, key="data.configuration", value="condorio")

        self.sc.add_sites(local, condorpool)


    # --- Transformation Catalog (Executables and Containers) ----------------------
    def create_transformation_catalog(self):
        self.tc = TransformationCatalog()
        
        predict_sales_container = Container("predict_sales_container",
            Container.SINGULARITY,
            image="docker:///papajim/predict_sales_container:latest",
            image_site="dockerhub"
        )

        # Add the xgboost hyperparameter tuning executable
        xgboost_hp_tuning = Transformation("xgboost_hp_tuning", site="condorpool", pfn=os.path.join(self.wf_dir, "bin/xgboost_hp_tuning.py"), is_stageable=True, container=predict_sales_container)\
                                        .add_pegasus_profile(cores="16")

        # Find best params from xgboost hp tuning
        xgboost_best_params = Transformation("xgboost_best_params", site="condorpool", pfn=os.path.join(self.wf_dir, "bin/xgboost_best_params.py"), is_stageable=True, container=predict_sales_container)
        
        if self.xgb_tree_method == "gpu_hist":
            xgboost_hp_tuning.add_pegasus_profile(gpus="1")
        
        self.tc.add_containers(predict_sales_container)
        self.tc.add_transformations(xgboost_hp_tuning, xgboost_best_params)


    # --- Replica Catalog ----------------------------------------------------------
    def create_replica_catalog(self):
        self.rc = ReplicaCatalog()
        
        for f in self.input_files:
            self.rc.add_replica("local", f, os.path.join(self.wf_dir, "data", f))
        for f in self.config_files:
            self.rc.add_replica("local", f, os.path.join(self.wf_dir, "config", f))


    # --- Find position of the mandatory columns in the column list ------------------
    def find_mandatory_col_positions(self):
        columns = json.load(open(self.xgb_cols_file, "r"))["columns"]
        mandatory_col_pos = [columns.index(x) for x in self.xgb_default_cols]
        return (mandatory_col_pos, len(columns))


    # --- Workflow -------------------------------------------------------------------
    def create_workflow(self):
        self.wf = Workflow(self.wf_name, infer_dependencies=True)

        # --- Create hyperparameter tuning jobs and add to the dag ----------------------
        hash_alg = hashlib.new("md5")
        xgboost_data_file = File(self.xgb_data_file)
        xgboost_hp_tuning_space = File(self.xgb_hp_tuning_conf)
        params_prefix = self.xgb_data_file[:str(self.xgb_data_file).find(".")]
        params_name = "{0}_hp_params.json".format(params_prefix)
        xgboost_params_out = File(params_name)
        if self.xgb_feat_lens == [-1]:
            xgboost_hp_tuning_job = Job("xgboost_hp_tuning")\
                                        .add_args("--file", xgboost_data_file, "--space", xgboost_hp_tuning_space, "--trials", self.xgb_trials, "--early_stopping_rounds", self.xgb_early_stopping, "--tree_method", self.xgb_tree_method, "--output", xgboost_params_out)\
                                        .add_inputs(xgboost_data_file, xgboost_hp_tuning_space)\
                                        .add_outputs(xgboost_params_out, stage_out=True, register_replica=True)

            self.wf.add_jobs(xgboost_hp_tuning_job)
        else:
            xgboost_hp_tuning_outputs = []
            (xgboost_mandatory_col_pos, xgboost_col_len) = self.find_mandatory_col_positions()
            iter_indexes = [i for i in range(xgboost_col_len) if not i in xgboost_mandatory_col_pos]
            for feat_len in self.xgb_feat_lens:
                feat_combinations = combinations(iter_indexes, feat_len - len(xgboost_mandatory_col_pos))
                for feat_combination in feat_combinations:
                    new_features = xgboost_mandatory_col_pos + list(feat_combination)
                    new_features = list(map(str, new_features))
                    temp_params_name = "{0}_hp_params_{1}.json".format(params_prefix, "_".join(new_features))
                    temp_params_md5 = "{0}_hp_params_{1}.json".format(params_prefix, hashlib.md5("_".join(new_features).encode()).hexdigest())
                    temp_xgboost_params_out = File(temp_params_md5)\
                                                .add_metadata(original_name=temp_params_name, features="_".join(new_features))
                    
                    xgboost_hp_tuning_outputs.append(temp_xgboost_params_out)
            
                    xgboost_hp_tuning_job = Job("xgboost_hp_tuning")\
                                                .add_args("--file", xgboost_data_file, "--space", xgboost_hp_tuning_space, "--trials", self.xgb_trials, "--early_stopping_rounds", self.xgb_early_stopping, "--tree_method", self.xgb_tree_method, "--output", xgboost_params_out, "--col_filter", ",".join(new_features))\
                                                .add_inputs(xgboost_data_file, xgboost_hp_tuning_space)\
                                                .add_outputs(temp_xgboost_params_out, stage_out=True, register_replica=True)
                    
                    self.wf.add_jobs(xgboost_hp_tuning_job)
            
            xgboost_best_params_job = Job("xgboost_best_params")\
                                        .add_args("--prefix", params_prefix, "--output", xgboost_params_out)\
                                        .add_outputs(xgboost_params_out, stage_out=True, register_replica=True)
            
            for xgboost_hp_tuning_output in xgboost_hp_tuning_outputs:                            
                xgboost_best_params_job.add_inputs(xgboost_hp_tuning_output)

            self.wf.add_jobs(xgboost_best_params_job)


def xgb_feat_len_check(xgb_feat_len):
    if xgb_feat_len[0] < 0:
        xgb_feat_len[0] = -1
    elif xgb_feat_len[0] < 5:
        xgb_feat_len[0] = 5

    if xgb_feat_len[1] < xgb_feat_len[0]:
        xgb_feat_len[1] = xgb_feat_len[0]

    return xgb_feat_len


def main():
    parser = ArgumentParser(description="Pegasus Workflow for Kaggle's Future Sales Predictiong Competition")
    parser.add_argument("--xgb_data_file", metavar="STR", type=str, default="main_data_feature_eng_all.pickle", help="Data file", required=False)
    parser.add_argument("--xgb_cols_file", metavar="STR", type=str, default="merged_features_output.json", help="Columns file", required=False)
    parser.add_argument("--xgb_trials", metavar="INT", type=int, default=5, help="Max trials for XGBoost hyperparameter tuning", required=False)
    parser.add_argument("--xgb_early_stopping", metavar="INT", type=int, default=5, help="XGBoost early stopping rounds", required=False)
    parser.add_argument("--xgb_tree_method", metavar="STR", type=str, default="hist", choices=["hist", "gpu_hist"], help="XGBoost hist type", required=False)
    parser.add_argument("--xgb_feat_len", metavar="INT", type=int, nargs=2, default=[-1, -1], help="Train XGBoost by including features between [LEN_MIN, LEN_MAX], LEN_MIN>=5", required=False)
    parser.add_argument("--xgb_default_cols", metavar="STR", type=str, nargs="+", default=["date_block_num", "shop_id", "item_id", "item_cnt_month", "item_category_id"], help="Columns to always use in hp tuning", required=False)
    parser.add_argument("--xgb_hp_tuning_conf", metavar="STR", type=str, default="xgboost_hp_tuning_space.json", help="JSON file describing hp tuning space", required=False)
    parser.add_argument("--is_root_wf", action="store_true", help="Create the workflow as a root worfklow", required=False)
    parser.add_argument("--output_single", action="store_true", help="Output Pegasus configuration in a single yaml file", required=False)
    parser.add_argument("--output", metavar="STR", type=str, default="workflow.yml", help="Output file", required=False)

    args = parser.parse_args()
    args.xgb_feat_len = xgb_feat_len_check(args.xgb_feat_len)
    number_of_features = [i for i in range(args.xgb_feat_len[0], args.xgb_feat_len[1]+1)]

    xgb_args = {
        "xgb_trials": args.xgb_trials,
        "xgb_early_stopping": args.xgb_early_stopping,
        "xgb_tree_method": args.xgb_tree_method,
        "xgb_feat_lens": number_of_features,
        "xgb_data_file": args.xgb_data_file,
        "xgb_cols_file": args.xgb_cols_file,
        "xgb_default_cols": args.xgb_default_cols,
        "xgb_hp_tuning_conf": args.xgb_hp_tuning_conf
    }

    workflow = predict_future_sales_workflow(args.is_root_wf, args.output, args.output_single, xgb_args)

    if args.is_root_wf:
        workflow.create_pegasus_properties()
        workflow.create_sites_catalog()
        workflow.create_replica_catalog()
        workflow.create_transformation_catalog()
    
    workflow.create_workflow()

    workflow.write()


if __name__ == "__main__":
    main()
