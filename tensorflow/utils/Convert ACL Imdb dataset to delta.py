# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Load Libs
# MAGIC 
# MAGIC This code loads up the unzipped acl IMDB Dataset and writes it out into a delta table

# COMMAND ----------

import os
import matplotlib.pyplot as plt

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Review DataSet on dbfs

# COMMAND ----------

# Extra Dirs setup for DB
data_dir = 'user/brian.law/data/'
uri_data_dir = 'dbfs:/' + data_dir

local_dataset_dir = os.path.join('dbfs', data_dir)
local_aclImdb_dir = os.path.join(os.path.dirname(local_dataset_dir), 'aclImdb')

uri_aclImdb_dir = os.path.join(uri_data_dir, 'aclImdb')
uri_aclImdb_dir

# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC cat /dbfs/user/brian.law/data/aclImdb/README

# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC ls /dbfs/user/brian.law/data/aclImdb/test
# MAGIC #cat /dbfs/user/brian.law/data/aclImdb/train/neg/0_3.txt
# MAGIC #cat /dbfs/user/brian.law/data/aclImdb/train/urls_unsup.txt

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Writing Data to Delta table 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC We assume that the default dataset has already been unzipped into object store

# COMMAND ----------

## Spark Reader ##
from pyspark.sql import functions as F 

def process_aclImdb(input_frame):
  
  """
  For reading the sample textfiles from aclImdb dataset
  """
  
  result_frame = input_frame \
                  .withColumn("filepath", F.input_file_name()) \
                  .withColumn("filepath_split", F.split(F.col("filepath"), "/") ) \
                  .select(F.element_at("filepath_split", -1).alias("filename_type"),
                          F.col("filepath"), F.col("value")) \
                  .withColumn("filename", F.split(F.col("filename_type"), "\\.")[0]) \
                  .withColumn("extension", F.split(F.col("filename_type"), "\\.")[1]) \
                  .withColumn("sample_id", F.split(F.col("filename"), "_")[0]) \
                  .withColumn("sample_label", F.split(F.col("filename"), "_")[1]) \
                  .select("sample_id", "value", "sample_label", "filepath", "filename", "extension")
  
  return result_frame

# COMMAND ----------

train_neg = os.path.join(uri_aclImdb_dir, "train", "neg")
train_pos = os.path.join(uri_aclImdb_dir, "train", "pos")
test_neg = os.path.join(uri_aclImdb_dir, "test", "neg")
test_pos = os.path.join(uri_aclImdb_dir, "test", "pos")

train_negative = spark.read.text(train_neg)
train_positive = spark.read.text(train_pos)
test_negative = spark.read.text(test_neg)
test_positive = spark.read.text(test_pos)

train_negative = process_aclImdb(train_negative) \
                  .withColumn("set", F.lit("train")) \
                  .withColumn("label", F.lit(0))
train_positive = process_aclImdb(train_positive) \
                  .withColumn("set", F.lit("train")) \
                  .withColumn("label", F.lit(1))
test_negative = process_aclImdb(test_negative) \
                  .withColumn("set", F.lit("test")) \
                  .withColumn("label", F.lit(0))
test_positive = process_aclImdb(test_positive) \
                  .withColumn("set", F.lit("test")) \
                  .withColumn("label", F.lit(1))

# COMMAND ----------

aclImdb_frame = train_negative \
                  .union(train_positive) \
                  .union(test_negative) \
                  .union(test_positive)

# COMMAND ----------

## merged_frame
display(aclImdb_frame)

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC CREATE DATABASE IF NOT EXISTS brian_petastorm_datasets

# COMMAND ----------

aclImdb_frame.write.format("delta").mode("overwrite").option("mergeSchema", "true").saveAsTable("brian_petastorm_datasets.aclImdb_label")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Review the dataset post transform into Delta Table

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT * FROM brian_petastorm_datasets.aclImdb_label

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC OPTIMIZE brian_petastorm_datasets.aclImdb_label

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Petastorm testing to return valid datasets for tf

# COMMAND ----------

from petastorm.spark import SparkDatasetConverter, make_spark_converter

spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, 'file:///dbfs/user/brian.law/pt_cache')

# COMMAND ----------

train_frame = spark.sql("select value, sample_label as `label` from brian_petastorm_datasets.aclImdb_label \
                        where `set` = 'train'")

df_train, df_val = train_frame.randomSplit([0.8,0.2], seed=12345)

#df_train.repartition(4)
#df_val.repartition(4)

# COMMAND ----------


