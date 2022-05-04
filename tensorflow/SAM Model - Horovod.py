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

import matplotlib.pyplot as plt

tf.get_logger().setLevel('ERROR')

# COMMAND ----------

# MLFlow workaround - For Horovod
#DEMO_SCOPE_TOKEN_NAME = "TOKEN"
databricks_host = dbutils.secrets.get(scope="scaling_dl", key="host_workspace")
databricks_token = dbutils.secrets.get(scope="scaling_dl", key="host_token")

# Extra Dirs setup for DB
log_dir = '/dbfs/Users/brian.law@databricks.com/tf_log_dirs'
dataset_dir = '/dbfs/user/brian.law/data/'
temp_path = '/dbfs/user/brian.law/tmp'

# COMMAND ----------

# MAGIC %run ./utils/aclimdb_utils

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Tensorboard Setup

# COMMAND ----------

# MAGIC %load_ext tensorboard
# MAGIC experiment_log_dir = log_dir

# COMMAND ----------

# MAGIC %tensorboard --logdir $experiment_log_dir

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Sentiment Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC In order to leverage Horovod, we need to move all components of the main training loop into one train function that will be distributed amongst the workers. Horovod will handle the moving around of weights and collecting of metrics

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Dataset Loading

# COMMAND ----------

# MAGIC %run ./dataloaders/aclimdb_dataloaders

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

# Defining Optimizer

epochs = 5
batch_size = 64
init_lr = 3e-5

# COMMAND ----------

def train_hvd(run_key: str):
  
  # imports to be run on the workers
  import tensorflow as tf
  import horovod.tensorflow as hvd
  import mlflow
  import tensorflow_hub as hub
  import tensorflow_text as text
  from official.nlp import optimization  # to create AdamW optimizer
  from math import floor
  import os
  
  os.environ['DATABRICKS_HOST'] = databricks_host
  os.environ['DATABRICKS_TOKEN'] = databricks_token
  os.environ['MLFLOW_RUN_ID'] = run_key
  
  hvd.init()
  
  # Pin GPU to be used to process local rank (one GPU per process)
  # These steps are skipped on a CPU cluster
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
  if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
  
  mlflow.tensorflow.autolog(log_models=False)
  
  train_ds, val_ds, test_ds, size_train, size_val, size_test = get_raw_dataset(dataset_dir,
                                                                                 batch_size=1, 
                                                                                 shards=hvd.size(), 
                                                                                 rank=hvd.rank())

  classifier_model = build_tf_raw_model()
  
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
  
  callbacks = [
    hvd.keras.callbacks.BroadcastGlobalVariablesCallback(0),
  ]
  
  if hvd.rank() == 0:
    with mlflow.start_run() as run:
      mlflow.tensorflow.autolog(log_models=False)
      
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
      history = classifier_model.fit(x=train_ds, 
                       validation_data=val_ds,
                       validation_steps=floor(size_val / batch_size),
                       steps_per_epoch=steps_per_epoch,          
                       epochs=epochs,
                       batch_size=batch_size,
                       callbacks=callbacks,
                       verbose=2,
                      workers=2)
      
      dataset_name = 'imdb'
      
      ### Needs to be modified in order to log to s3 path
      saved_model_path = os.path.join(temp_path, '{}_bert'.format(dataset_name.replace('/', '_'))) 

      classifier_model.save(saved_model_path, include_optimizer=False)
      
      mlflow.tensorflow.log_model(tf_saved_model_dir=saved_model_path,
                            tf_meta_graph_tags=None,
                            tf_signature_def_key='serving_default', # default from tf official model
                            artifact_path='model', # model is default for mlflow in order to link to UI
                            signature=signature,
                            input_example=input_examples,
                            extra_pip_requirements=extra_reqs)

      fig = accuracy_and_loss_plots(history)
      mlflow.log_figure(fig, 'training_perf.png')
      
      loss, accuracy = classifier_model.evaluate(test_ds)
      
      mlflow.end_run()

  else:
    classifier_model.fit(x=train_ds, 
                       validation_data=val_ds,
                       validation_steps=floor(size_val / batch_size),
                       steps_per_epoch=steps_per_epoch,          
                       epochs=epochs,
                       batch_size=batch_size,
                       callbacks=callbacks,
                       verbose=2,
                      workers=2)
    
    # we need to shard the datasets
    # we will do batching at the model level hence batch_size is set to 1
  

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC HorovodRunner automatically pickles variables and distributes them along with the main function.
# MAGIC Using something like horovod.spark.runner requires that everything be packaged up and specified by the user instead

# COMMAND ----------

# This crashes at Epoch 5???
### Needed to make sure that we have the batch set properly perhaps?
## Setting batch_size partially fixed but still getting issues with the dataloader and all_gather


# COMMAND ----------

run_id = 'e60b44d6c0b340e8be7481fb3d87d534'

# COMMAND ----------

mlflow.end_run()

# COMMAND ----------

from sparkdl import HorovodRunner

import mlflow

mlflow.start_run(experiment_id='224704298431727', run_name='HorovodDist')  
run_id = mlflow.active_run().info.run_id
mlflow.end_run()

# , checkpoint_path=checkpoint_path, learning_rate=learning_rate
hr = HorovodRunner(np=2, driver_log_verbosity='all')
hr.run(train_hvd, run_key=run_id)
