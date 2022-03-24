import torch
import pytorch_lightning as pl
from pl_bolts.datamodules import FashionMNISTDataModule
from model_test import LitFashionMNIST

# do tensorboard profiling
import torch.profiler


BATCH_SIZE = 256
AVAIL_GPUS = max(1, torch.cuda.device_count())
AVAIL_GPUS

# Adding mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

data_path = '/dbfs/user/brian.law/data/fashionMNIST'
experiment_log_dir = '/dbfs/user/brian.law/tboard_test/logs'
RUN_NAME = 'pl_test'


dbutils.fs.mkdirs(data_path)

# greater than one worker results in things hanging 0 or 1 works
data_module = FashionMNISTDataModule(data_dir=data_path, num_workers=4)

# initialize model
model = LitFashionMNIST(*data_module.size(), data_module.num_classes)

# start mlflow
mlflow.pytorch.autolog()

# Loggers
loggers = []
tb_logger = pl.loggers.tensorboard.TensorBoardLogger(save_dir=experiment_log_dir, name=RUN_NAME,log_graph=True)

loggers.append(tb_logger)

# Callbacks
callbacks = []


# Profilers
from pytorch_lightning.profiler import PyTorchProfiler
import os

profiler = PyTorchProfiler(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(experiment_log_dir,RUN_NAME), worker_name='worker0'),
    record_shapes=True,
    profile_memory=True,  # This will take 1 to 2 minutes. Setting it to False could greatly speedup.
    with_stack=True)

trainer = pl.Trainer(
    max_epochs=3,
    log_every_n_steps=1,
    gpus=AVAIL_GPUS,
    callbacks=callbacks,
    logger=loggers,
    strategy='ddp'
    #profiler=profiler # for tensorboard profiler
)
# Pass the datamodule as arg to trainer.fit to override model hooks :)
with mlflow.start_run(run_name='basic_fashionMNIST') as run:
  trainer.fit(model, data_module)

