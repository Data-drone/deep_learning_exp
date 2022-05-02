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

tf.get_logger().setLevel('DEBUG')

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
# MAGIC experiment_log_dir = log_dir

# COMMAND ----------

# MAGIC %tensorboard --logdir $experiment_log_dir

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

# MAGIC %md
# MAGIC 
# MAGIC ### Loading Pretrained Model
# MAGIC 
# MAGIC - Can choose any of the models in map_name_to_handle as long as it first in mem (this is single GPU example)
# MAGIC - Each model has an associated preprocess

# COMMAND ----------

# MAGIC %run ./models/aclimdb_models

# COMMAND ----------

classifier_model = build_tf_raw_model()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Validating Model Works

# COMMAND ----------

text_test = ['this is such an amazing movie!']

bert_raw_result = classifier_model(tf.constant(text_test))
print(tf.sigmoid(bert_raw_result))

# COMMAND ----------

# requires pydot and graphviz - installed on cluster level
tf.keras.utils.plot_model(classifier_model)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Model Training

# COMMAND ----------

## Adding extra MLflow steps
# due to extra tf requirements we will manually call autolog and turn off the artifact logging
import mlflow

# COMMAND ----------

# Defining Optimizer

epochs = 3
batch_size = 128

steps_per_epoch = size_train // batch_size 
num_train_steps = steps_per_epoch * epochs
num_warmup_steps = int(0.1*num_train_steps)

init_lr = 3e-5

# COMMAND ----------

# Compile model
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
metrics = tf.metrics.BinaryAccuracy()

optimizer = optimization.create_optimizer(init_lr=init_lr,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')

classifier_model.compile(optimizer=optimizer,
                         loss=loss,
                         metrics=metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Advanced MLFlow Logging
# MAGIC 
# MAGIC We need to set these up to include with the main training loop

# COMMAND ----------

# unless we specify extra then mlflow will just log these
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

mlflow.tensorflow.get_default_pip_requirements()

# COMMAND ----------

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

# COMMAND ----------

import datetime
# Main training Loop
from petastorm.spark import make_spark_converter

with mlflow.start_run(experiment_id='224704298431727') as run:
  
  mlflow.tensorflow.autolog(log_models=False)
  
  ## Adding Extra logging
  run_log_dir = os.path.join(log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
  #debug_dir = os.path.join(run_log_dir, 'debug')

  tf.debugging.experimental.enable_dump_debug_info(
    run_log_dir,
    tensor_debug_mode="FULL_HEALTH",
    circular_buffer_size=-1)
  
  # didn't seem to work?
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=run_log_dir, histogram_freq=1, update_freq=1)
  
  ### Extra code for Petastorm
  converter_train = make_spark_converter(train_ds)
  converter_val = make_spark_converter(val_ds)
  
  with converter_train.make_tf_dataset(batch_size=batch_size, workers_count=10) as train_pt_ds, \
    converter_val.make_tf_dataset(batch_size=batch_size, workers_count=10) as val_pt_ds:
    
    # The dataset that comes out might not be:
    # -- the right format
    # -- the right ordering
    train_iter = train_pt_ds.map(lambda x: (x[0], x[1]))
    val_iter = val_pt_ds.map(lambda x: (x[0], x[1]))
  
    ### End Extra Code for Petastorm
  
    history = classifier_model.fit(x=train_iter,
                                 validation_data=val_iter,
                                 validation_steps=size_val // batch_size,
                                 epochs=epochs,
                                 steps_per_epoch=steps_per_epoch,
                                 batch_size=batch_size,
                                 callbacks=[tensorboard_callback])
  
  # we need to first save out the model to a temp folder then we can log it
  dataset_name = 'imdb'
  saved_model_path = './{}_bert'.format(dataset_name.replace('/', '_'))

  #classifier_model.save(saved_model_path, include_optimizer=False)
  
  # try manually speccing some things in log model
  # tf_signature_def_key was from trial and error seems like
  # meta_graph_tags is none by design 
  # classifier_model automatically sets this
  
  #mlflow.tensorflow.log_model(tf_saved_model_dir=saved_model_path,
  #                          tf_meta_graph_tags=None,
  #                          tf_signature_def_key='serving_default', # default from tf official model
  #                          artifact_path='model', # model is default for mlflow in order to link to UI
  #                          signature=signature,
  #                          input_example=input_examples,
  #                          extra_pip_requirements=extra_reqs)

  fig = accuracy_and_loss_plots(history)
  mlflow.log_figure(fig, 'training_perf.png')
  

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Model Evaluation
# MAGIC 
# MAGIC -- option in training steps

# COMMAND ----------

converter_test = make_spark_converter(test_ds)

with converter_test.make_tf_dataset(batch_size=batch_size, workers_count=10) as test_pt_ds:
  
  test_iter = test_pt_ds.map(lambda x: (x[0], x[1]))
  
  # Note if the steps isn't defined then it will loop infinitely
  loss, accuracy = classifier_model.evaluate(test_iter,
                                            steps=size_test // batch_size)

  print(f'Loss: {loss}')
  print(f'Accuracy: {accuracy}')

# COMMAND ----------

mlflow.end_run()

# COMMAND ----------

### clean up
clean_up = False

if clean_up:
  from numba import cuda
  
  cuda.select_device(0)
  cuda.close()
  

# COMMAND ----------


