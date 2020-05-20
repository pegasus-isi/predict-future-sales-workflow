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
                pfn=str(Path(__file__).parent.resolve() / "bin/EDA.py"),
                is_stageable=True
            )

# Add the proprocess executable
preprocess = Transformation(
                "preprocess",
                site="condorpool",
                pfn=str(Path(__file__).parent.resolve() / "bin/preprocess.py"),
                is_stageable=True
            )

tc.add_transformations(eda, preprocess)
tc.write()

# --- Replica Catalog ----------------------------------------------------------
test = File("test.csv")
items = File("items.csv")
shops = File("shops.csv")
holidays = File("holidays.csv")
sales_train = File("sales_train.csv")
item_categories = File("item_categories.csv")

rc = ReplicaCatalog()\
        .add_replica("local", "items.csv", str(Path(__file__).parent.resolve() / "data/items.csv"))\
        .add_replica("local", "item_categories.csv", str(Path(__file__).parent.resolve() / "data/item_categories.csv"))\
        .add_replica("local", "shops.csv", str(Path(__file__).parent.resolve() / "data/shops.csv"))\
        .add_replica("local", "sales_train.csv", str(Path(__file__).parent.resolve() / "data/sales_train.csv"))\
        .add_replica("local", "test.csv", str(Path(__file__).parent.resolve() / "data/test.csv"))\
        .add_replica("local", "holidays.csv", str(Path(__file__).parent.resolve() / "data/holidays.csv"))\
        .write()

# --- Workflow -------------------------------------------------------------------
ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
wf = Workflow(("predict-sales-workflow-%s" % ts), infer_dependencies=True)


# --- Add EDA Job ----------------------------------------------------------------
eda_output = File("EDA.pdf")
eda_job = Job(eda)\
            .add_inputs(items,item_categories, shops, sales_train)\
            .add_outputs(eda_output)

# --- Add Preprocess Job ---------------------------------------------------------
test_preprocessed = File("test_preprocessed.pickle")
items_preprocessed = File("items_preprocessed.pickle")
shops_preprocessed = File("shops_preprocessed.pickle")
sales_train_preprocessed = File("sales_train_preprocessed.pickle")
item_categories_preprocessed = File("item_categories_preprocessed.pickle")

preprocess_job = Job(preprocess)\
                    .add_inputs(items,item_categories, shops, sales_train, test)\
                    .add_outputs(test_preprocessed, items_preprocessed, shops_preprocessed, sales_train_preprocessed, item_categories_preprocessed)

# --- Add Jobs to the Workflow dag -----------------------------------------------
wf.add_jobs(eda_job, preprocess_job)

wf.write()
