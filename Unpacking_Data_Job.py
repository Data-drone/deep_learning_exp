# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Loading Data via Python Module

# COMMAND ----------

local_disk_tmp_dir = '/local_disk0/tmp_data'
username = "brian.law@databricks.com"
dataset_name = 'cifar10'

# COMMAND ----------

torchvision_loader = """

import torchvision

CIFAR10 = torchvision.datasets.CIFAR10(
          root='{0}',
          
          download=True)

""".format(local_disk_tmp_dir)

# COMMAND ----------

load_path = "/Users/{0}/test_script/preload_{1}.py".format(username, dataset_name)

dbutils.fs.put(load_path, torchvision_loader, True)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Init Script to run on nodes

# COMMAND ----------

init_script = """

#!/bin/bash

/databricks/python/bin/python /dbfs{1}



""".format(username, load_path)

init_script

# COMMAND ----------

init_script_path = "dbfs:/Users/{0}/init/preload_{1}.sh".format(username, dataset_name)

dbutils.fs.put(init_script_path, init_script, True)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Triggering the job along with the copy notebook

# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC databricks jobs create --help

# COMMAND ----------

HOST = "https://e2-demo-tokyo.cloud.databricks.com/"
TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

from databricks_cli import sdk
from databricks_cli.jobs.api import JobsApi
import json

client = sdk.ApiClient(host=HOST, token=TOKEN)

jobs = JobsApi(api_client = client)

# COMMAND ----------

target_path = "/Repos/brian.law@databricks.com/scaling_deep_learning/CopyFiles_Notebook"

# COMMAND ----------

job_config ={
        "name": "CopyFiles_DeepLearning",
        "timeout_seconds": 0,
        "max_concurrent_runs": 1,
        "tasks": [
            {
                "task_key": "CopyFiles_DeepLearning",
                "notebook_task": {
                    "notebook_path": target_path
                },
                "job_cluster_key": "copy_files_temp",
                "timeout_seconds": 0,
                "email_notifications": {}
            }
        ],
        "job_clusters": [
            {
                "job_cluster_key": "copy_files_temp",
                "new_cluster": {
                    "spark_version": "10.4.x-cpu-ml-scala2.12",
                    "aws_attributes": {
                        "zone_id": "ap-northeast-1a",
                        "first_on_demand": 4,
                        "availability": "SPOT_WITH_FALLBACK",
                        "spot_bid_price_percent": 100,
                        "ebs_volume_type": "GENERAL_PURPOSE_SSD",
                        "ebs_volume_count": 1,
                        "ebs_volume_size": 100
                    },
                    "node_type_id": "m5.8xlarge",
                    "init_scripts": [{
                      "dbfs": {
                        "destination": init_script_path
                        }
                      }],
                    "enable_elastic_disk": False,
                    "runtime_engine": "STANDARD",
                    "autoscale": {
                        "min_workers": 2,
                        "max_workers": 20
                    }
                }
            }
        ]
}

# COMMAND ----------

job_create_id = jobs.create_job(json=job_config)
job_create_id

# COMMAND ----------

jobs.list_jobs()

# COMMAND ----------

notebook_params={'source_path': 'file:'+local_disk_tmp_dir, 
                 'destination': 'dbfs:/Users/{0}/data/'.format(username), 
                 'checkpoint': 'dbfs:/Users/{0}/tmp_checkpoint'.format(username)}

# COMMAND ----------

jobs.run_now(job_id=job_create_id['job_id'], jar_params=None, notebook_params=notebook_params, 
             python_params=None, spark_submit_params=None)

# COMMAND ----------

cleanup = False

if cleanup:
  jobs.delete_job(job_id=job_create_id['job_id'])
