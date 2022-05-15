# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## Training and packaging a Tensorflow 2.x model with Model Hub
# MAGIC 
# MAGIC - based on: https://www.tensorflow.org/text/tutorials/classify_text_with_bert
# MAGIC - Sentiment Analysis Model

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Extra libraries not in DB ML Image
# MAGIC - tensorflow-text
# MAGIC - tf-models-official

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Load Libs

# COMMAND ----------

import os

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer
from petastorm.spark import SparkDatasetConverter, make_spark_converter

import matplotlib.pyplot as plt

tf.get_logger().setLevel('DEBUG')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Config Variables

# COMMAND ----------

### Setting up MLFlow variables for Horovod
databricks_mlflow_host = 'https://e2-demo-tokyo.cloud.databricks.com'
databricks_notebook_token = dbutils.notebook.entry_point.getDbutils().getContext().apiToken().get()



# COMMAND ----------

# Extra Dirs setup for DB
log_dir = '/dbfs/Users/brian.law@databricks.com/tf_log_dirs'
dataset_dir = '/dbfs/user/brian.law/data/'

## Extra Cache Path for Petastorm
cache_pathing = 'user/brian.law/pt_cache_1/'
local_cache_path = os.path.join('/dbfs', cache_pathing)
cache_dir = 'file://' + local_cache_path

# COMMAND ----------

## Clean up and create cache
dbutils.fs.rm(local_cache_path, True)
dbutils.fs.mkdirs(local_cache_path)

# COMMAND ----------

# MAGIC %run ./utils/aclimdb_utils

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Tensorboard Setup

# COMMAND ----------

# MAGIC %load_ext tensorboard
# MAGIC %tensorboard --logdir $log_dir

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Sentiment Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Dataset Loading

# COMMAND ----------

# MAGIC %run ./dataloaders/aclimdb_dataloaders

# COMMAND ----------

train_ds, val_ds, test_ds, size_train, size_val, size_test = get_petastorm_dataset(cache_dir=cache_dir, partitions=4)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Loading Pretrained Model
# MAGIC 
# MAGIC - Can choose any of the models in map_name_to_handle as long as it first in mem (this is single GPU example)
# MAGIC - Each model has an associated preprocess

# COMMAND ----------

# MAGIC %run ./models/aclimdb_models

# COMMAND ----------


