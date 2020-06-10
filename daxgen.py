#!/usr/bin/env python3
import os
import logging
import hashlib
from datetime import datetime
from argparse import ArgumentParser

from pathlib import Path
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
    xgb_trials = None
    xgb_early_stopping = None
    xgb_tree_method = None
    
    output_multiple = None

    input_files = ["test.csv","items.csv","items_translated.csv","shops.csv","holidays.csv","sales_train.csv","item_categories.csv"]
    config_files = ["xgboost_hp_tuning_space.json"]

    
    # --- Init ------------------------------------------------------------------
    def __init__(self, daxfile, output_multiple, xgb_trials, xgb_early_stopping, xgb_tree_method):
        self.daxfile = daxfile
        self.output_multiple = output_multiple
        self.xgb_trials = xgb_trials
        self.xgb_early_stopping = xgb_early_stopping
        self.xgb_tree_method = xgb_tree_method
        
        self.wf_dir = Path(__file__).parent.resolve()
        
        ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
        self.wf_name = "predict-sales-workflow-%s" % ts


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

        # Add the eda executable
        eda = Transformation("eda", site="condorpool", pfn=os.path.join(self.wf_dir, "bin/EDA.py"), is_stageable=True)

        # Add the proprocess executable
        preprocess = Transformation("preprocess", site="condorpool", pfn=os.path.join(self.wf_dir, "bin/preprocess.py"), is_stageable=True)

        # Add the nlp executable
        nlp = Transformation("nlp", site="condorpool", pfn=os.path.join(self.wf_dir, "bin/NLP.py"), is_stageable=True)

        # Add the feature_eng_0 executable
        feature_eng_0 = Transformation("feature_eng_0", site="condorpool", pfn=os.path.join(self.wf_dir, "bin/feature_eng_0.py"), is_stageable=True)

        # Add the feature_eng_1 executable
        feature_eng_1 = Transformation("feature_eng_1", site="condorpool", pfn=os.path.join(self.wf_dir, "bin/feature_eng_1.py"), is_stageable=True)

        # Add the feature_eng_2 executable
        feature_eng_2 = Transformation("feature_eng_2", site="condorpool", pfn=os.path.join(self.wf_dir, "bin/feature_eng_2.py"), is_stageable=True)

        # Add the feature_eng_3 executable
        feature_eng_3 = Transformation("feature_eng_3", site="condorpool", pfn=os.path.join(self.wf_dir, "bin/feature_eng_3.py"), is_stageable=True)

        # Add the feature_eng_4 executable
        feature_eng_4 = Transformation("feature_eng_4", site="condorpool", pfn=os.path.join(self.wf_dir, "bin/feature_eng_4.py"), is_stageable=True)

        # Add the feature_eng_5 executable
        feature_eng_5 = Transformation("feature_eng_5", site="condorpool", pfn=os.path.join(self.wf_dir, "bin/feature_eng_5.py"), is_stageable=True)

        # Add the merge executable
        merge = Transformation("merge", site="condorpool", pfn=os.path.join(self.wf_dir, "bin/merge.py"), is_stageable=True)

        # Add the xgboost hyperparameter tuning executable
        xgboost_hp_tuning = Transformation("xgboost_hp_tuning", site="condorpool", pfn=os.path.join(self.wf_dir, "bin/xgboost_hp_tuning.py"), is_stageable=True)

        # Add the xgboost model creation executable
        xgboost_model = Transformation("xgboost_model", site="condorpool", pfn=os.path.join(self.wf_dir, "bin/xgboost_model.py"), is_stageable=True)
        
        self.tc.add_transformations(eda, nlp, preprocess, feature_eng_0, feature_eng_1, feature_eng_2, feature_eng_3, feature_eng_4, feature_eng_5, merge, xgboost_hp_tuning, xgboost_model)


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
        
        test = File("test.csv")
        items = File("items.csv")
        items_translated = File("items_translated.csv")
        shops = File("shops.csv")
        holidays = File("holidays.csv")
        sales_train = File("sales_train.csv")
        item_categories = File("item_categories.csv")

        # --- Add EDA Job ----------------------------------------------------------------
        eda_output = File("EDA.pdf")
        eda_job = Job("eda")\
                    .add_inputs(items, item_categories, shops, sales_train)\
                    .add_outputs(eda_output, stage_out=True, register_replica=True)

        # --- Add NLP Job ----------------------------------------------------------------
        shops_nlp = File("shops_nlp.pickle")
        items_nlp = File("items_nlp.pickle")
        tenNN_items = File("tenNN_items.pickle")
        threeNN_shops = File("threeNN_shops.pickle")
        items_clusters = File("items_clusters.pickle")
        nlp_job = Job("nlp")\
                    .add_inputs(items_translated, item_categories, shops)\
                    .add_outputs(shops_nlp, items_nlp, tenNN_items, threeNN_shops, items_clusters, stage_out=True, register_replica=True)\
                    .add_pegasus_profile(cores="16")

        # --- Add Preprocess Job ---------------------------------------------------------
        test_preprocessed = File("test_preprocessed.pickle")
        items_preprocessed = File("items_preprocessed.pickle")
        shops_preprocessed = File("shops_preprocessed.pickle")
        sales_train_preprocessed = File("sales_train_preprocessed.pickle")
        categories_preprocessed = File("categories_preprocessed.pickle")

        preprocess_job = Job("preprocess")\
                            .add_inputs(items, item_categories, shops, sales_train, test)\
                            .add_outputs(test_preprocessed, items_preprocessed, shops_preprocessed, sales_train_preprocessed, categories_preprocessed, stage_out=True, register_replica=True)

        # --- Add feature_eng_0 Job -------------------------------------------------------
        items_feature_eng_0 = File("items_feature_eng_0.pickle")
        shops_feature_eng_0 = File("shops_feature_eng_0.pickle")
        categories_feature_eng_0 = File("categories_feature_eng_0.pickle")

        feature_eng_0_job = Job("feature_eng_0")\
                                .add_inputs(items_preprocessed, shops_preprocessed, categories_preprocessed)\
                                .add_outputs(items_feature_eng_0, shops_feature_eng_0, categories_feature_eng_0, stage_out=True, register_replica=True)

        # --- Add feature_eng_1 Job -------------------------------------------------------
        main_data_feature_eng_1 = File("main_data_feature_eng_1.pickle")

        feature_eng_1_job = Job("feature_eng_1")\
                                .add_inputs(items_preprocessed, categories_preprocessed, sales_train_preprocessed, test_preprocessed)\
                                .add_outputs(main_data_feature_eng_1, stage_out=True, register_replica=True)

        # --- Add feature_eng_2 Job -------------------------------------------------------
        main_data_feature_eng_2 = File("main_data_feature_eng_2.pickle")

        feature_eng_2_job = Job("feature_eng_2")\
                                .add_inputs(main_data_feature_eng_1)\
                                .add_outputs(main_data_feature_eng_2, stage_out=True, register_replica=True)

        # --- Add feature_eng_3 Job -------------------------------------------------------
        main_data_feature_eng_3 = File("main_data_feature_eng_3.pickle")

        feature_eng_3_job = Job("feature_eng_3")\
                                .add_inputs(items_feature_eng_0, shops_feature_eng_0, categories_feature_eng_0, main_data_feature_eng_1)\
                                .add_outputs(main_data_feature_eng_3, stage_out=True, register_replica=True)

        # --- Add feature_eng_4 Job -------------------------------------------------------
        main_data_feature_eng_4 = File("main_data_feature_eng_4.pickle")

        feature_eng_4_job = Job("feature_eng_4")\
                                .add_inputs(holidays, main_data_feature_eng_1)\
                                .add_outputs(main_data_feature_eng_4, stage_out=True, register_replica=True)

        # --- Add feature_eng_5 Job -------------------------------------------------------
        main_data_feature_eng_5 = File("main_data_feature_eng_5.pickle")

        feature_eng_5_job = Job("feature_eng_5")\
                                .add_inputs(items_feature_eng_0, shops_feature_eng_0, categories_feature_eng_0, sales_train, main_data_feature_eng_1)\
                                .add_outputs(main_data_feature_eng_5, stage_out=True, register_replica=True)

        # --- Add merge Job ---------------------------------------------------------------
        train_group_0 = File("train_group_0.pickle")
        test_group_0 = File("test_group_0.pickle")
        train_group_1 = File("train_group_1.pickle")
        test_group_1 = File("test_group_1.pickle")
        train_group_2 = File("train_group_2.pickle")
        test_group_2 = File("test_group_2.pickle")
        main_data_feature_eng_all = File("main_data_feature_eng_all.pickle")

        train_groups = [train_group_0, train_group_1]#, train_group_2]
        test_groups = [test_group_0, test_group_1, test_group_2] 

        merge_job = Job("merge")\
                        .add_inputs(tenNN_items, threeNN_shops, main_data_feature_eng_2, main_data_feature_eng_3, main_data_feature_eng_4, main_data_feature_eng_5)\
                        .add_outputs(train_group_0, train_group_1, train_group_2, test_group_0, test_group_1, test_group_2, main_data_feature_eng_all, stage_out=True, register_replica=True)


        # --- Add Jobs to the Workflow dag -----------------------------------------------
        self.wf.add_jobs(eda_job, nlp_job, preprocess_job, feature_eng_0_job, feature_eng_1_job, feature_eng_2_job, feature_eng_3_job, feature_eng_4_job, feature_eng_5_job, merge_job)


        hash_alg = hashlib.new("md5")
        xgboost_hp_tuning_outputs = []
        xgboost_hp_tuning_space = File("xgboost_hp_tuning_space.json")
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


def main():
    parser = ArgumentParser(description="Pegasus Workflow for Kaggle's Future Sales Predictiong Competition")
    parser.add_argument("--xgb_trials", metavar="INT", type=int, nargs=1, default=5, help="Max trials for XGBoost hyperparameter tuning", required=False)
    parser.add_argument("--xgb_early_stopping", metavar="INT", type=int, nargs=1, default=5, help="XGBoost early stopping rounds", required=False)
    parser.add_argument("--xgb_tree_method", metavar="STR", type=str, nargs=1, default="hist", help="Max trials for XGBoost hyperparameter tuning", required=False)
    #parser.add_argument("--xgb_feat_len", metavar="INT", type=int, nargs=2, help="Train XGBoost by including features between [LEN_MIN, LEN_MAX]", required=False)
    #parser.add_argument("--xgb_feat_list", metavar="STR", type=int, nargs=2, help="Train XGBoost with the given list of features", required=False)
    parser.add_argument("--output_multiple", action="store_true", help="Output Pegasus configuration in multiple files", required=False)
    parser.add_argument("--output", metavar="STR", type=str, default="workflow.yml", help="Output file", required=False)

    args = parser.parse_args()
    
    #number_of_features = [i for i in range(args.xgb_feat_len[0], args.xgb_feat_len[1])]

    workflow = predict_future_sales_workflow(args.output, args.output_multiple, args.xgb_trials, args.xgb_early_stopping, args.xgb_tree_method)

    workflow.create_pegasus_properties()
    workflow.create_sites_catalog()
    workflow.create_transformation_catalog()
    workflow.create_replica_catalog()
    workflow.create_workflow()

    workflow.write()


if __name__ == "__main__":
    main()
