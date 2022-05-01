# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # AclImdb Dataloader Tests
# MAGIC 
# MAGIC We need to check that each dataloader returns the same things.
# MAGIC So they both return tuples with two objects the text and the label
# MAGIC The label is a numeric and the text is a string

# COMMAND ----------

# MAGIC %run ./dataloaders/aclimdb_dataloaders

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Regular TF

# COMMAND ----------

dataset_dir = '/dbfs/user/brian.law/data/'
train_ds, val_ds, size_train, size_val = get_raw_dataset(dataset_dir)

# COMMAND ----------

for text_batch, label_batch in train_ds.take(1):
  for i in range(3):
    print(f'Review: {text_batch.numpy()[i]}')
    label = label_batch.numpy()[i]
    print(f'Label : {label}') 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Petastorm Table

# COMMAND ----------

from petastorm.spark import make_spark_converter

# COMMAND ----------

cache_path = '/dbfs/user/brian.law/pt_cache/'
cache_dir = 'file://' + cache_path
dbutils.fs.rm(cache_path, True)
dbutils.fs.mkdirs(cache_path)
batch_size = 32

train_ds_pt, val_ds_pt, size_train_pt, size_val_pt = get_petastorm_dataset(cache_dir)

# COMMAND ----------

converter_train = make_spark_converter(train_ds_pt)

with converter_train.make_tf_dataset(batch_size=batch_size, workers_count=10) as pt_train:
  
  # this is how we can remap and or convert the values as needed
  # pt_train = pt_train.map(lambda x: (x[0],tf.strings.to_number(x[1],
  #                          out_type=tf.dtyles.int32)))
  
  pt_train = pt_train.map(lambda x: (x[0],x[1]))
  
  for text_batch, label_batch in pt_train.take(3):
    for i in range(3):
      print(f'Review: {text_batch[i]}')
      print(f'Label : {label_batch[i]}') 

# COMMAND ----------


