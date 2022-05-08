# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC GraphViz which is needed to visualise the tf network charts requires graphviz to be installed
# MAGIC This is OS level so init script is one way to easily handle this.
# MAGIC 
# MAGIC See: https://www.tensorflow.org/api_docs/python/tf/keras/utils/plot_model
# MAGIC See: https://graphviz.org/download/

# COMMAND ----------

graph_viz_init = """
#!/bin/bash

apt install -y graphviz

# install ray
/databricks/python/bin/pip install pydot

""" 

# Change ‘username’ to your Databricks username in DBFS
# Example: username = “stephen.offer@databricks.com”
username = “”
dbutils.fs.put("dbfs:/Users/{0}/init/graphviz.sh".format(username), graph_viz_init, True)
