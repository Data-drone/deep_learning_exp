# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ACL Imdb dataset from tf hub
# MAGIC 
# MAGIC In order to make components more interchangeable we need to make sure that we return the same results from different dataformat dataloaders

# COMMAND ----------

def get_petastorm_dataset(cache_dir: str, partitions: int=4):
  """
  
  This Dataloader assumes that the dataset has been converted to Delta table already
  
  The Delta Table Schema is:
  root
   |-- sample_id: string (nullable = true)
   |-- value: string (nullable = true)
   |-- sample_label: string (nullable = true)
   |-- filepath: string (nullable = true)
   |-- filename: string (nullable = true)
   |-- extension: string (nullable = true)
   |-- set: string (nullable = true)
   |-- label: integer (nullable = true)
  
  See: TBD to Load and convert the aclImdb dataset from the tf sample dataset lib
  
  Args:
    cache_dir: Cache Directory for Peatstorm
    partitions: Num Partitions for Petastorm partitions need to match num horovod threads / gpus (TO CHECK)
  
  Returns:
    df_train: spark df of training data
    df_val: spark df of val data
    size_train: size of the training dataset for use in batch step calcs
    size_val: size of the val dataset for use in validation batch step calcs 
    
  """
  
  from petastorm.spark import SparkDatasetConverter, make_spark_converter

  spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, cache_dir)

  train_frame = spark.sql("select value, `label` \
                        from brian_petastorm_datasets.aclImdb_label \
                        where `set` = 'train'")
  
  df_test = spark.sql("select value, `label` \
                        from brian_petastorm_datasets.aclImdb_label \
                        where `set` = 'test'")

  df_train, df_val = train_frame.randomSplit([0.8,0.2], seed=12345)

  df_train.repartition(partitions)
  df_val.repartition(partitions)
  df_test.repartition(partitions)

  size_train = df_train.count()
  size_val = df_val.count()
  size_test = df_test.count()
  
  return df_train, df_val, df_test, size_train, size_val, size_test

# COMMAND ----------

def get_raw_dataset(object_store_dir:str, batch_size:int=32, shards:int=1, rank:int=0):
  
  """
  Gets the raw unzipped datafiles
  We don't do the unzipping here because using the default keras utils can be slow on cloud
  
  Args:
    object_store_dir: File path for to where the dataset will be eg:
                      '/dbfs/user/{uesrname}/data/'
    batch_size: Batch size for the tf datasets
    shards: Total number of shards default is 1 needed for Horovod
    rank: Current shard number for this batch default 0
  
  """
  
  import tensorflow as tf
  import os
  seed = 42
  
  # Autotune the buffer size
  AUTOTUNE = tf.data.AUTOTUNE
  
  url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'

  dataset = tf.keras.utils.get_file('aclImdb_v1.tar.gz', url,
                                  untar=False, cache_dir=object_store_dir,
                                  cache_subdir='')
  
  unzip_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
  
  raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    os.path.join(unzip_dir, 'train'),
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed)
  
  class_names = raw_train_ds.class_names
  train_ds = raw_train_ds.shard(num_shards=shards, index=rank).cache().prefetch(buffer_size=AUTOTUNE)
  
  val_ds = tf.keras.utils.text_dataset_from_directory(
    os.path.join(unzip_dir, 'train'),
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed)

  val_ds = val_ds.shard(num_shards=shards, index=rank).cache().prefetch(buffer_size=AUTOTUNE)
  
  test_ds = tf.keras.utils.text_dataset_from_directory(
    os.path.join(unzip_dir, 'test'),
    batch_size=batch_size)

  test_ds = test_ds.shard(num_shards=shards, index=rank).cache().prefetch(buffer_size=AUTOTUNE)
  
  size_train = len(train_ds) * batch_size
  size_val = len(val_ds) * batch_size
  size_test = len(test_ds) * batch_size
  
  # df_train, df_val, size_train, size_val
  return train_ds, val_ds, test_ds, size_train, size_val, size_test
