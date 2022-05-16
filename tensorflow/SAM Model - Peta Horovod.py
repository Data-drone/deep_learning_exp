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

## This doesn't always work
#databricks_notebook_token = dbutils.notebook.entry_point.getDbutils().getContext().apiToken().get()
databricks_notebook_token = dbutils.secrets.get(scope="scaling_dl", key="host_workspace")

## Run Name
mlflow_experiment_name = 224704298431727
mlflow_run_name = 'Horovod Petastorm Cluster Run'

## This will be used to make sure the tf logs get separated out properly
tf_log_prefix = 'horo_peta_cluster_'

# COMMAND ----------

# Extra Dirs setup for DB
log_dir = '/dbfs/Users/brian.law@databricks.com/flowers_tf_log_dirs'
dataset_dir = '/dbfs/user/brian.law/data/'
temp_path = '/local_disk0/tmp' # this just for temp storing model

## Extra Cache Path for Petastorm
cache_pathing = 'user/brian.law/pt_cache_1/'
local_cache_path = os.path.join('/dbfs', cache_pathing)
cache_dir = 'file://' + local_cache_path

## Cluster Config
single_node = False
total_num_gpus = 2
petastorm_workers = 10 

## Model Settings
epochs = 5
batch_size = 64
init_lr = 3e-5

global_batch_size = batch_size * total_num_gpus

# COMMAND ----------

### parsing to make sure horovod runner np set right
if single_node:
  hr_setting = -total_num_gpus
else:
  hr_setting = total_num_gpus

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

train_ds, val_ds, test_ds, size_train, size_val, size_test = get_petastorm_dataset(cache_dir=cache_dir, partitions=total_num_gpus*4)

# COMMAND ----------

# When using Horovod and Petastorm, we can initialise the spark converter first before pushing it in
peta_train_ds = make_spark_converter(train_ds)
peta_val_ds = make_spark_converter(val_ds)

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

# MAGIC %md
# MAGIC 
# MAGIC ### Setup Main Training Loop

# COMMAND ----------

def train_hvd(mlflow_run_key:str):
  
  import os
  import horovod.tensorflow as hvd
  import tensorflow as tf
  import mlflow
  import tensorflow_text
  from math import floor
  
  ## Setup MLFlow for Horovod
  os.environ['DATABRICKS_HOST'] = databricks_mlflow_host
  os.environ['DATABRICKS_TOKEN'] = databricks_notebook_token
  os.environ['MLFLOW_RUN_ID'] = mlflow_run_key
  
  hvd.init()
  
  # Pin GPU to be used to process local rank (one GPU per process)
  # These steps are skipped on a CPU cluster
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
  if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
  
  mlflow.tensorflow.autolog(log_models=False)
  
  classifier_model = build_tf_raw_model()

  # These are specifically needed for the 
  steps_per_epoch = floor(size_train / batch_size)
  num_train_steps = steps_per_epoch * epochs
  num_warmup_steps = int(0.1*num_train_steps)

  # Compile model
  loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  metrics = tf.metrics.BinaryAccuracy()

  parallel_lr = init_lr*hvd.size()
  optimizer = optimization.create_optimizer(init_lr=parallel_lr,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')
  
  # Add Horovod
  optimizer = hvd.DistributedOptimizer(optimizer)

  classifier_model.compile(optimizer=optimizer,
                         loss=loss,
                         metrics=metrics)
  
  run_log = os.path.join(log_dir, tf_log_prefix+mlflow_run_key)
  
  callbacks = [
    hvd.keras.callbacks.BroadcastGlobalVariablesCallback(0),
  ]
  
  
  
  with peta_train_ds.make_tf_dataset(batch_size=batch_size, workers_count=petastorm_workers,
                                    cur_shard=hvd.rank(), shard_count=hvd.size()) as trainset, \
    peta_val_ds.make_tf_dataset(batch_size=batch_size, workers_count=petastorm_workers,
                                    cur_shard=hvd.rank(), shard_count=hvd.size()) as valset: 
  
    
    # Fix up the formatting for the dataloaders
    trainset = trainset.map(lambda x: (x[0], x[1]))
    valset = valset.map(lambda x: (x[0], x[1]))
  
    hvd_steps_per_epoch = len(peta_train_ds) // global_batch_size
    val_steps = len(peta_val_ds) // global_batch_size
  
    if hvd.rank() == 0:
      with mlflow.start_run() as run:
        mlflow.tensorflow.autolog(log_models=False)
      
        tensorboard_stats = tf.keras.callbacks.TensorBoard(log_dir=run_log, histogram_freq=1,
                                                          update_freq=10, profile_batch='40,60')
        
        
        ## TODO CheckPointing
        
        callbacks.append([tensorboard_stats])
      
        # unless we specify extra then mlflow will just log these
        from mlflow.models.signature import ModelSignature
        from mlflow.types.schema import Schema, ColSpec

        # extra pip requirements for our specific example
        extra_reqs = [
          f"tensorflow-text==2.8.*",
          f"tf-models-official==2.7.0"
        ] 

        # for the input signature field
        input_examples = [
          'this is such an amazing movie!',  # this is the same sentence tried earlier
          'The movie was great!',
          'The movie was meh.',
          'The movie was okish.',
          'The movie was terrible...'
        ]

        # lets manually spec input and output schema

        input_schema = Schema([
          ColSpec("string", "An Input Sentence to Evaluate")
        ])

        output_schema = Schema([ColSpec("float", "The Sentiment Score")])

        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        
  
      ## If steps are defined that it will crash
        history = classifier_model.fit(x=trainset, 
                       validation_data=valset,
                       validation_steps=val_steps,
                       steps_per_epoch=hvd_steps_per_epoch,          
                       epochs=epochs,
                       callbacks=callbacks,
                       verbose=1, workers=2)
      
        dataset_name = 'imdb'
        
        saved_model_path = os.path.join(temp_path, '{}_bert'.format(dataset_name.replace('/', '_')))
        
        ## Autolog seems temperamental in logging this particular model maybe the tensorflow model hub and text
        ## has issues? Or the dependencies are weird?
        classifier_model.save(saved_model_path, include_optimizer=False)
        
        mlflow.tensorflow.log_model(tf_saved_model_dir=saved_model_path,
                            tf_meta_graph_tags=None,
                            tf_signature_def_key='serving_default', # default from tf official model
                            artifact_path='model', # model is default for mlflow in order to link to UI
                            signature=signature,
                            input_example=input_examples,
                            extra_pip_requirements=extra_reqs)
    
    else:
      classifier_model.fit(x=trainset, 
                       validation_data=valset,
                       validation_steps=val_steps,
                       steps_per_epoch=hvd_steps_per_epoch,          
                       epochs=epochs,
                       callbacks=callbacks,
                       verbose=1, workers=2)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Horovod Execution

# COMMAND ----------

from sparkdl import HorovodRunner

import mlflow

mlflow.start_run(experiment_id=mlflow_experiment_name, run_name=mlflow_run_name)
run_id = mlflow.active_run().info.run_id
mlflow.end_run()

hr = HorovodRunner(np=hr_setting, driver_log_verbosity='all')
hr.run(train_hvd, mlflow_run_key=run_id)

# COMMAND ----------


