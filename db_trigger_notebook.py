# Databricks notebook source
# MAGIC %md
# MAGIC # Scaling up Deep Learning

# COMMAND ----------

#dbutils.fs.mkdirs('/dbfs/user/brian.law/lightning_fashion_mnist/checkpoints')

# COMMAND ----------

from sparkdl import HorovodRunner
import horovod.torch as hvd
from fashion_mnist_main import main_hvd
import os

# COMMAND ----------

# MLFlow workaround
#DEMO_SCOPE_TOKEN_NAME = "TOKEN"
databricks_host = dbutils.secrets.get(scope="scaling_dl", key="host_workspace")
databricks_token = dbutils.secrets.get(scope="scaling_dl", key="host_token")
os.environ['DATABRICKS_HOST'] = databricks_host
os.environ['DATABRICKS_TOKEN'] = databricks_token

# setup env
fashion_data_path = '/dbfs/user/brian.law/data/fashionMNIST'
cifar_data_path = '/dbfs/user/brian.law/data/cifar10'
voc_data_path = '/dbfs/user/brian.law/data/VOC'
log_dir = '/dbfs/user/brian.law/pl_logs'

batch_size = 512
epochs=50

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tensorboard Initialisation

# COMMAND ----------

%load_ext tensorboard
experiment_log_dir = log_dir

# COMMAND ----------

%tensorboard --logdir $experiment_log_dir

# COMMAND ----------

# MAGIC %md
# MAGIC ## Image Classification

# COMMAND ----------

# setup the experiment
from pl_bolts.datamodules import FashionMNISTDataModule, CIFAR10DataModule
from models.fashion_mnist_basic import LitFashionMNIST
from models.resnet_basic import ResnetClassification
from models.timm_model import TimmEfficientNetClassification

# greater than one worker can result in things hanging 0 or 1 works
data_module = FashionMNISTDataModule(data_dir=fashion_data_path, num_workers=4, batch_size=batch_size)
data_module = CIFAR10DataModule(data_dir=cifar_data_path, num_workers=4, batch_size=batch_size)

# initialize model
model = LitFashionMNIST(*data_module.size(), data_module.num_classes)
model = ResnetClassification(*data_module.size(), data_module.num_classes, pretrain=True)
model = TimmEfficientNetClassification(*data_module.size(), data_module.num_classes, pretrain=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Object Detection

# COMMAND ----------
from pl_bolts.datamodules import VOCDetectionDataModule
from pl_bolts.models import FasterRCNN

data_module = VOCDetectionDataModule(data_dir=voc_data_path, num_workers=4, batch_size=batch_size)

# initialise model
model = FasterRCNN(*data_module.size(), data_module.num_classes)

# COMMAND ----------

# set np to number of gpus
# Horovod Runner doesn't work when trying to distribute the local files too in repo mode
#hr = HorovodRunner(np=1, driver_log_verbosity='all')
#model = hr.run(main_hvd, mlflow_db_host=databricks_host, mlflow_db_token=databricks_token)

import horovod.spark 

# set to the number of workers * ?num gpu per worker?
num_processes = 2

model = horovod.spark.run(main_hvd, args=(databricks_host, databricks_token, data_module, model, log_dir, epochs), 
                            num_proc=num_processes, verbose=2)

# COMMAND ----------

