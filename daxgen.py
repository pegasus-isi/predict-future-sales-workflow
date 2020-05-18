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
                pfn="{PWD}/bin/eda.py",
                is_stageable=True
            )

# Add the proprocess executable
preprocess = Transformation(
                "preprocess",
                site="condorpool",
                pfn="{PWD}/bin/preprocess.py",
                is_stageable=True
            )

tc.add_transformations(eda, preprocess)
tc.write()

# --- Replica Catalog ----------------------------------------------------------
items = File("items.csv")
item_categories = File("item_categories.csv")
shops = File("shops.csv")
sales_train = File("sales_train.csv")
test = File("test.csv")
holidays = File("holidays.csv")

rc = ReplicaCatalog()\
        .add_replica("local", "items.csv", str(Path(__file__).parent.resolve() / "data/items_translated.csv"))\
        .add_replica("local", "item_categories.csv", str(Path(__file__).parent.resolve() / "data/item_categories_translated.csv"))\
        .add_replica("local", "shops.csv", str(Path(__file__).parent.resolve() / "data/shops_translated.csv"))\
        .add_replica("local", "sales_train.csv", str(Path(__file__).parent.resolve() / "data/sales_train.csv"))\
        .add_replica("local", "test.csv", str(Path(__file__).parent.resolve() / "data/test.csv"))\
        .add_replica("local", "holidays.csv", str(Path(__file__).parent.resolve() / "data/holidays.csv"))\
        .write()
