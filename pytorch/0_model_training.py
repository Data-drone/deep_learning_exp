# Databricks notebook source
# MAGIC %md
# MAGIC # Scaling up Deep Learning
# This notebook can be run to create all the training runs we need

# COMMAND ----------

#dbutils.fs.mkdirs('/dbfs/user/brian.law/lightning_fashion_mnist/checkpoints')

# COMMAND ----------

import horovod.torch as hvd
from fashion_mnist_main import main_hvd, main_train
import os

# COMMAND ----------

# MAGIC %md
# MAGIC # Setup Config

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLFlow setup

# COMMAND ----------

# MLFlow parameters
# Worker nodes do not have access to the host and token details when running python subprocesses
databricks_host = dbutils.secrets.get(scope="scaling_dl", key="host_workspace")
databricks_token = dbutils.secrets.get(scope="scaling_dl", key="host_token")
os.environ['DATABRICKS_HOST'] = databricks_host
os.environ['DATABRICKS_TOKEN'] = databricks_token

# We precreate the experiment so that we can log different notebooks together
experiment_id = 4388967990215332

# COMMAND ----------

# MAGIC %md
# MAGIC ## Logging and Saving Dirs

# COMMAND ----------

## TO DO verify these and make sure we log to right folder
fashion_data_path = '/dbfs/user/brian.law/data/fashionMNIST'
cifar_data_path = '/dbfs/user/brian.law/data/cifar10'
voc_data_path = '/dbfs/user/brian.law/data/VOC'
log_dir = '/dbfs/user/brian.law/pl_logs'

# COMMAND ----------

# MAGIC %md
# MAGIC ## Experiment Parameters

# COMMAND ----------

root_dir = '/dbfs/user/brian.law/lightning_fashion_mnist/checkpoints'
#data_path = '/dbfs/user/brian.law/data/fashionMNIST'
#experiment_log_dir = '/dbfs/user/brian.law/tboard_test/logs'
#run_name = 'basic_fashionMNIST'


# COMMAND ----------

# MAGIC %md
# MAGIC # Experiments
# We will run a series of mnist experiments

# COMMAND ----------

from pl_bolts.datamodules import FashionMNISTDataModule, CIFAR10DataModule
from models.resnet_basic import ResnetClassification

# COMMAND ----------

# MAGIC %md
# MAGIC ## Single Node Single GPU

# COMMAND ----------

RUN_NAME = 'pl_test'
batch_size = 512
epochs=50
num_dataloader_workers = 4

data_module = FashionMNISTDataModule(data_dir=fashion_data_path, 
                                        num_workers=num_dataloader_workers, 
                                        batch_size=batch_size)

model = ResnetClassification(*data_module.size(), data_module.num_classes, pretrain=False)

trainer = main_train(
    data_module=data_module, 
    model=model, 
    num_gpus=1, 
    experiment_log_dir=log_dir, 
    epoch=epochs
)

# COMMAND ----------
