#!/usr/bin/env python3
import os
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
    output_multiple = None
    
    xgb_trials = None
    xgb_early_stopping = None
    xgb_tree_method = None
    xgb_default_cols = None
    xgb_hp_tuning_conf = None

    
    # --- Init ---------------------------------------------------------------------
    def __init__(self, is_root_wf, daxfile, output_multiple, xgb_args):
        self.daxfile = daxfile
        self.output_multiple = output_multiple
        self.xgb_trials = xgb_args.xgb_trials
        self.xgb_early_stopping = xgb_args.xgb_early_stopping
        self.xgb_tree_method = xgb_args.xgb_tree_method
        self.xgb_feat_lens = xgb_args.xgb_feat_lens
        self.xgb_default_cols = xgb_args.xgb_default_cols
        self.xgb_hp_tuning_conf = xgb_args.xgb_hp_tuning_conf
        self.is_root_wf = is_root_wf
        
        self.wf_dir = Path(__file__).parent.resolve()
        
        ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
        self.wf_name = "predict-sales-wf-hp-tuning-%s" % ts


    # --- Write files in directory -------------------------------------------------
    def write(self):
        if self.output_multiple:
            self.sc.write()
            self.rc.write()
            self.tc.write()
        else:
            self.wf.add_site_catalog(self.sc)
            self.wf.add_replica_catalog(self.rc)
            self.wf.add_transformation_catalog(self.tc)

        if self.is_root_wf:
            self.props.write()
        self.wf.write()


    # --- Configuration (Pegasus Properties) ---------------------------------------
    def create_pegasus_properties(self):
        self.props = Properties()

        #props["pegasus.data.configuration"] = "condorio"
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

        # Add the xgboost hyperparameter tuning executable
        xgboost_hp_tuning = Transformation("xgboost_hp_tuning", site="condorpool", pfn=os.path.join(self.wf_dir, "bin/xgboost_hp_tuning.py"), is_stageable=True)

        # Add the xgboost model creation executable
        xgboost_model = Transformation("xgboost_best_params", site="condorpool", pfn=os.path.join(self.wf_dir, "bin/xgboost_best_params.py"), is_stageable=True)
        
        self.tc.add_transformations(xgboost_hp_tuning, xgboost_best_params)


    # --- Replica Catalog ----------------------------------------------------------
    def create_replica_catalog(self):
        self.rc = ReplicaCatalog()
        
        for f in self.input_files:
            self.rc.add_replica("local", f, os.path.join(self.wf_dir, "data", f))
        for f in self.config_files:
            self.rc.add_replica("local", f, os.path.join(self.wf_dir, "config", f))


    # --- Workflow -------------------------------------------------------------------
    def create_workflow(self):
        self.wf = Workflow(self.wf_name, infer_dependencies=True)

        # --- Create hyperparameter tuning jobs and add to the dag ----------------------
        # --- This should become a subworkflow that takes as input the merged features --
        hash_alg = hashlib.new("md5")
        xgboost_hp_tuning_outputs = []
        xgboost_mandatory_cols = [0, 1, 2, 3, 4] # first 5 columns of the merged table
        xgboost_hp_tuning_space = File("xgboost_hp_tuning_space.json")

        if self.xgb_feat_lens == [-1]:
            for group in train_groups:
                param_name = "{0}_hp_params.json".format(group.lfn[:str(group.lfn).find(".")])
                params_out = File(param_name)
                xgboost_hp_tuning_outputs.append(params_out)
                xgboost_hp_tuning_job = Job("xgboost_hp_tuning")\
                                            .add_args("--file", group, "--space", xgboost_hp_tuning_space, "--trials", self.xgb_trials, "--early_stopping_rounds", self.xgb_early_stopping, "--tree_method", self.xgb_tree_method, "--output", params_out)\
                                            .add_inputs(group, xgboost_hp_tuning_space)\
                                            .add_outputs(params_out, stage_out=True, register_replica=True)\
                                            .add_pegasus_profile(cores="16")

                self.wf.add_jobs(xgboost_hp_tuning_job)
        else:
            for feat_len in self.xgb_feat_lens:
                feat_combinations = 
                for group in train_groups:
                xgboost_hp_tuning_outputs.append(params_out)


def xgb_feat_len_check(xgb_feat_len):
    if xgb_feat_len[0] < -1:
        xgb_feat_len[0] = -1
    elif xgb_feat_len[0] < 5:
        xgb_feat_len[0] = 5

    if xgb_feat_len[1] < -1:
        xgb_feat_len[1] = -1
    elif xgb_feat_len[1] < xgb_feat_len[0]:
        xgb_feat_len[1] = xgb_feat_len[0]


def main():
    parser = ArgumentParser(description="Pegasus Workflow for Kaggle's Future Sales Predictiong Competition")
    parser.add_argument("--xgb_data_file", metavar="STR", type=str, nargs=1, default="main_data_feature_eng_all.pickle", help="Data file", required=False)
    parser.add_argument("--xgb_cols_file", metavar="STR", type=str, nargs=1, default="merged_features_output.json", help="Columns file", required=False)
    parser.add_argument("--xgb_trials", metavar="INT", type=int, nargs=1, default=5, help="Max trials for XGBoost hyperparameter tuning", required=False)
    parser.add_argument("--xgb_early_stopping", metavar="INT", type=int, nargs=1, default=5, help="XGBoost early stopping rounds", required=False)
    parser.add_argument("--xgb_tree_method", metavar="STR", type=str, nargs=1, default="hist", choices=["hist", "gpu_hist"], help="XGBoost hist type", required=False)
    parser.add_argument("--xgb_feat_len", metavar="INT", type=int, nargs=2, default=[-1, -1], help="Train XGBoost by including features between [LEN_MIN, LEN_MAX], LEN_MIN>=5", required=False)
    parser.add_argument("--xgb_default_cols", metavar="STR", type=str, nargs=+, default=["date_block_num", "shop_id", "item_id", "item_cnt_month", "item_category_id"], help="Columns to always use in hp tuning", required=False)
    parser.add_argument("--xgb_hp_tuning_conf", metavar="STR", type=str, nargs=1, default="xgboost_hp_tuning_space.json", help="JSON file describing hp tuning space", required=False)
    #parser.add_argument("--xgb_feat_list", metavar="STR", type=str, nargs=+, help="Train XGBoost with the given list of features", required=False)
    parser.add_argument("--is_root_wf", action="store_true", help="Create the workflow as a root worfklow", required=False)
    parser.add_argument("--output_multiple", action="store_true", help="Output Pegasus configuration in multiple files", required=False)
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
        "xgb_hp_tuning_conf:" args.xgb_hp_tuning_conf
    }

    workflow = predict_future_sales_workflow(args.is_root_wf, rgs.output, args.output_multiple, xgb_args)

    if args.is_root_wf:
        workflow.create_pegasus_properties()
    workflow.create_sites_catalog()
    workflow.create_transformation_catalog()
    workflow.create_replica_catalog()
    workflow.create_workflow()

    workflow.write()


if __name__ == "__main__":
    main()
