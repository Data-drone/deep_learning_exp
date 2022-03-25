# Databricks notebook source
# MAGIC %md
# MAGIC # Testing Running notebooks

# COMMAND ----------

dbutils.fs.mkdirs('/dbfs/user/brian.law/lightning_fashion_mnist/checkpoints')

# COMMAND ----------

# MAGIC %sh 
# MAGIC 
# MAGIC python fashion_mnist_test.py

# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC $DB_DRIVER_IP

# COMMAND ----------


