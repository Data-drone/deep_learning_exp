# Databricks notebook source
# MAGIC %md
# MAGIC # Testing Running notebooks

# COMMAND ----------

#dbutils.fs.mkdirs('/dbfs/user/brian.law/lightning_fashion_mnist/checkpoints')

# COMMAND ----------

# MAGIC %sh 
# MAGIC 
# MAGIC python fashion_mnist_main.py

# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC $DB_DRIVER_IP

# COMMAND ----------

from sparkdl import HorovodRunner
import horovod.torch as hvd
from fashion_mnist_main import main_hvd
from pl_bolts.datamodules import FashionMNISTDataModule
from models.fashion_mnist_basic import LitFashionMNIST
import os

# COMMAND ----------

# MLFlow workaround
#DEMO_SCOPE_TOKEN_NAME = "TOKEN"
databricks_host = dbutils.secrets.get(scope="scaling_dl", key="host_workspace")
databricks_token = dbutils.secrets.get(scope="scaling_dl", key="host_token")
os.environ['DATABRICKS_HOST'] = databricks_host
os.environ['DATABRICKS_TOKEN'] = databricks_token

# setup env
data_path = '/dbfs/user/brian.law/data/fashionMNIST'

# COMMAND ----------

# setup the experiment

# greater than one worker can result in things hanging 0 or 1 works
data_module = FashionMNISTDataModule(data_dir=data_path, num_workers=4)

# initialize model
model = LitFashionMNIST(*data_module.size(), data_module.num_classes)


# COMMAND ----------

# set np to number of gpus
# Horovod Runner doesn't work when trying to distribute the local files too in repo mode
#hr = HorovodRunner(np=1, driver_log_verbosity='all')
#model = hr.run(main_hvd, mlflow_db_host=databricks_host, mlflow_db_token=databricks_token)

import horovod.spark 

# set to the number of workers * ?num gpu per worker?
num_processes = 4

model = horovod.spark.run(main_hvd, args=(databricks_host, databricks_token, data_module, model,), num_proc=4, verbose=2)

# COMMAND ----------

