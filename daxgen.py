#!/usr/bin/env python3
import os
import logging
import sys
from datetime import datetime

from pathlib import Path
logging.basicConfig(level=logging.DEBUG)

# --- Import Pegasus API -------------------------------------------------------
from Pegasus.api import *

# --- Configuration (Pegasus Properties) ---------------------------------------
props = Properties()

#props["pegasus.data.configuration"] = "condorio"
props["pegasus.monitord.encoding"] = "json"                                                                    
props["pegasus.catalog.workflow.amqp.url"] = "amqp://friend:donatedata@msgs.pegasus.isi.edu:5672/prod/workflows"

props.write()

# --- Site Catalog -------------------------------------------------------------
sc = SiteCatalog()

shared_scratch_dir = "${PWD}/scratch"
local_storage_dir = "${PWD}/output"

local = Site("local")\
                .add_directories(
                    Directory(Directory.SHAREDSCRATCH, shared_scratch_dir)
                        .add_file_servers(FileServer("file://" + shared_scratch_dir, Operation.ALL)),
                    
                    Directory(Directory.LOCALSTORAGE, local_storage_dir)
                        .add_file_servers(FileServer("file://" + local_storage_dir, Operation.ALL))
                )

condorpool = Site("condorpool")\
                .add_pegasus_profile(style="condor")\
                .add_condor_profile(universe="vanilla")\
                .add_profiles(Namespace.PEGASUS, key="data.configuration", value="condorio")

sc.add_sites(local, condorpool)
sc.write()

# --- Transformation Catalog (Executables and Containers) ----------------------
tc = TransformationCatalog()

# Add the eda executable
eda = Transformation(
                "eda",
                site="condorpool",
                pfn="${PWD}/bin/EDA.py",
                is_stageable=True
            )

# Add the proprocess executable
preprocess = Transformation(
                "preprocess",
                site="condorpool",
                pfn="${PWD}/bin/preprocess.py",
                is_stageable=True
            )

# Add the nlp executable
nlp = Transformation(
                "nlp",
                site="condorpool",
                pfn="${PWD}/bin/NLP.py",
                is_stageable=True
            )

# Add the feature_eng_0 executable
feature_eng_0 = Transformation(
                "feature_eng_0",
                site="condorpool",
                pfn="${PWD}/bin/feature_eng_0.py",
                is_stageable=True
            )

# Add the feature_eng_1 executable
feature_eng_1 = Transformation(
                "feature_eng_1",
                site="condorpool",
                pfn="${PWD}/bin/feature_eng_1.py",
                is_stageable=True
            )

# Add the feature_eng_2 executable
feature_eng_2 = Transformation(
                "feature_eng_2",
                site="condorpool",
                pfn="${PWD}/bin/feature_eng_2.py",
                is_stageable=True
            )

# Add the feature_eng_3 executable
feature_eng_3 = Transformation(
                "feature_eng_3",
                site="condorpool",
                pfn="${PWD}/bin/feature_eng_3.py",
                is_stageable=True
            )

# Add the feature_eng_4 executable
feature_eng_4 = Transformation(
                "feature_eng_4",
                site="condorpool",
                pfn="${PWD}/bin/feature_eng_4.py",
                is_stageable=True
            )

# Add the feature_eng_5 executable
feature_eng_5 = Transformation(
                "feature_eng_5",
                site="condorpool",
                pfn="${PWD}/bin/feature_eng_5.py",
                is_stageable=True
            )

# Add the merge executable
merge = Transformation(
                "merge",
                site="condorpool",
                pfn="${PWD}/bin/merge.py",
                is_stageable=True
            )

# Add the xgboost executable
xgboost_hp_tuning = Transformation(
                "xgboost_hp_tuning",
                site="condorpool",
                pfn="${PWD}/bin/xgboost_hp_tuning.py",
                is_stageable=True
            )

tc.add_transformations(eda, nlp, preprocess, feature_eng_0, feature_eng_1, feature_eng_2, feature_eng_3, feature_eng_4, feature_eng_5, merge, xgboost_hp_tuning)
tc.write()

# --- Replica Catalog ----------------------------------------------------------
test = File("test.csv")
items = File("items.csv")
items_translated = File("items_translated.csv")
shops = File("shops.csv")
holidays = File("holidays.csv")
sales_train = File("sales_train.csv")
item_categories = File("item_categories.csv")

rc = ReplicaCatalog()\
        .add_replica("local", "items.csv", str(Path(__file__).parent.resolve() / "data/items.csv"))\
        .add_replica("local", "items_translated.csv", str(Path(__file__).parent.resolve() / "data/items_translated.csv"))\
        .add_replica("local", "item_categories.csv", str(Path(__file__).parent.resolve() / "data/item_categories.csv"))\
        .add_replica("local", "shops.csv", str(Path(__file__).parent.resolve() / "data/shops.csv"))\
        .add_replica("local", "sales_train.csv", str(Path(__file__).parent.resolve() / "data/sales_train.csv"))\
        .add_replica("local", "test.csv", str(Path(__file__).parent.resolve() / "data/test.csv"))\
        .add_replica("local", "holidays.csv", str(Path(__file__).parent.resolve() / "data/holidays.csv"))\
        .add_replica("local", "xgboost_hp_tuning_space.json", str(Path(__file__).parent.resolve() / "config/xgboost_hp_tuning_space.json"))\
        .write()

# --- Workflow -------------------------------------------------------------------
ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
wf = Workflow(("predict-sales-workflow-%s" % ts), infer_dependencies=True)


# --- Add EDA Job ----------------------------------------------------------------
eda_output = File("EDA.pdf")
eda_job = Job(eda)\
            .add_inputs(items, item_categories, shops, sales_train)\
            .add_outputs(eda_output)

# --- Add NLP Job ----------------------------------------------------------------
shops_nlp = File("shops_nlp.pickle")
items_nlp = File("items_nlp.pickle")
tenNN_items = File("tenNN_items.pickle")
threeNN_shops = File("threeNN_shops.pickle")
items_clusters = File("items_clusters.pickle")
nlp_job = Job(nlp)\
            .add_inputs(items_translated, item_categories, shops)\
            .add_outputs(shops_nlp, items_nlp, tenNN_items, threeNN_shops, items_clusters)

# --- Add Preprocess Job ---------------------------------------------------------
test_preprocessed = File("test_preprocessed.pickle")
items_preprocessed = File("items_preprocessed.pickle")
shops_preprocessed = File("shops_preprocessed.pickle")
sales_train_preprocessed = File("sales_train_preprocessed.pickle")
categories_preprocessed = File("categories_preprocessed.pickle")

preprocess_job = Job(preprocess)\
                    .add_inputs(items, item_categories, shops, sales_train, test)\
                    .add_outputs(test_preprocessed, items_preprocessed, shops_preprocessed, sales_train_preprocessed, categories_preprocessed)

# --- Add feature_eng_0 Job -------------------------------------------------------
items_feature_eng_0 = File("items_feature_eng_0.pickle")
shops_feature_eng_0 = File("shops_feature_eng_0.pickle")
categories_feature_eng_0 = File("categories_feature_eng_0.pickle")

feature_eng_0_job = Job(feature_eng_0)\
                        .add_inputs(items_preprocessed, shops_preprocessed, categories_preprocessed)\
                        .add_outputs(items_feature_eng_0, shops_feature_eng_0, categories_feature_eng_0)

# --- Add feature_eng_1 Job -------------------------------------------------------
main_data_feature_eng_1 = File("main_data_feature_eng_1.pickle")

feature_eng_1_job = Job(feature_eng_1)\
                        .add_inputs(items_preprocessed, shops_preprocessed, categories_preprocessed, sales_train_preprocessed, test_preprocessed)\
                        .add_outputs(main_data_feature_eng_1)

# --- Add feature_eng_2 Job -------------------------------------------------------
main_data_feature_eng_2 = File("main_data_feature_eng_2.pickle")

feature_eng_2_job = Job(feature_eng_2)\
                        .add_inputs(main_data_feature_eng_1)\
                        .add_outputs(main_data_feature_eng_2)

# --- Add feature_eng_3 Job -------------------------------------------------------
main_data_feature_eng_3 = File("main_data_feature_eng_3.pickle")

feature_eng_3_job = Job(feature_eng_3)\
                        .add_inputs(items_feature_eng_0, shops_feature_eng_0, categories_feature_eng_0, main_data_feature_eng_1)\
                        .add_outputs(main_data_feature_eng_3)

# --- Add feature_eng_4 Job -------------------------------------------------------
main_data_feature_eng_4 = File("main_data_feature_eng_4.pickle")

feature_eng_4_job = Job(feature_eng_4)\
                        .add_inputs(main_data_feature_eng_1)\
                        .add_outputs(main_data_feature_eng_4)

# --- Add feature_eng_5 Job -------------------------------------------------------
main_data_feature_eng_5 = File("main_data_feature_eng_5.pickle")

feature_eng_5_job = Job(feature_eng_5)\
                        .add_inputs(main_data_feature_eng_1)\
                        .add_outputs(main_data_feature_eng_5)

# --- Add merge Job ---------------------------------------------------------------
train_group_0 = File("train_group_0.pickle")
test_group_0 = File("test_group_0.pickle")
train_group_1 = File("train_group_1.pickle")
test_group_1 = File("test_group_1.pickle")
train_group_2 = File("train_group_2.pickle")
test_group_2 = File("test_group_2.pickle")
main_data_feature_eng_all = File("main_data_feature_eng_all.pickle")

train_groups = [train_group_0, train_group_1, train_group_2]
test_groups = [test_group_0, test_group_1, test_group_2] 

merge_job = Job(merge)\
                .add_inputs(tenNN_items, threeNN_shops, main_data_feature_eng_2, main_data_feature_eng_3, main_data_feature_eng_4, main_data_feature_eng_5)\
                .add_outputs(train_group_0, train_group_1, train_group_2, test_group_0, test_group_1, test_group_2, main_data_feature_eng_all)


# --- Add Jobs to the Workflow dag -----------------------------------------------
wf.add_jobs(eda_job, nlp_job, preprocess_job, feature_eng_0_job, feature_eng_1_job, feature_eng_2_job, feature_eng_3_job, feature_eng_4_job, feature_eng_5_job, merge_job)


xgboost_hp_tuning_space = File("xgboost_hp_tuning_space.json")
xgboost_hp_tuning_outputs = []
for group in train_groups:
    param_name = str(group)[:str(group).find(".")+1] + "_hp_params.json"
    params_out = File(param_name)
    xgboost_hp_tuning_outputs.append(params_out)
    xgboost_hp_tuning_job = Job(xgboost_hp_tuning)\
                                .add_args("-f", group, "-c", xgboost_hp_tuning_space, "-t", 100, "-e", 20, "-o", params_out)\
                                .add_inputs(group, xgboost_hp_tuning_space)\
                                .add_outputs(params_out)

    wf.add_jobs(xgboost_hp_tuning_job)

wf.write()
