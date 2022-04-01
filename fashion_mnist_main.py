#####
#
# This main loop is intended to be run inside of databricks
# ddp mode isn't supported in interactive mode so is run via %sh
# running via %sh requires that a terminal session is started and databricks cli configured
# running via %sh won't log the code against the notebook

import torch
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule
from typing import Type

# do tensorboard profiling
import torch.profiler

# Adding mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

# pytorch profiling
from pytorch_lightning.profiler import PyTorchProfiler
import os

# spark horovod
from sparkdl import HorovodRunner
import horovod.torch as hvd

### Set config flags
EPOCHS = 5
BATCH_SIZE = 256
# this will just count the driver I believe
AVAIL_GPUS = max(1, torch.cuda.device_count()) 

root_dir = '/dbfs/user/brian.law/lightning_fashion_mnist/checkpoints'
#data_path = '/dbfs/user/brian.law/data/fashionMNIST'
experiment_log_dir = '/dbfs/user/brian.law/tboard_test/logs'
RUN_NAME = 'pl_test'
experiment_id = 4388967990215332
run_name = 'basic_fashionMNIST'


def main_hvd(mlflow_db_host:str, mlflow_db_token:str, 
            data_module:Type[LightningDataModule], model:Type[LightningModule]):

    """
    
    Args:
        mlflow_db_host: 
        mlflow_db_token: 
    
    """

    hvd.init()

    # mlflow workaround
    mlflow.set_tracking_uri("databricks")
    os.environ['DATABRICKS_HOST'] = mlflow_db_host
    os.environ['DATABRICKS_TOKEN'] = mlflow_db_token

    return main_train(data_module=data_module, model=model, strat='horovod', num_gpus=1, node_id=hvd.rank())


def main_train(data_module:Type[LightningDataModule], model:Type[LightningModule], 
                num_gpus:int, strat:str='ddp', node_id:int=0):

    """
    Main training Loop

    Args:
        data_dir: data module to fit in
        strat: training strategy for parallel
        num_gpus: number of gpus to train on
        node_id: the number of the node
    
    """

    # start mlflow
    ## manually trigger log models later as there seems to be a pickling area with logging the model
    if node_id==0:
        mlflow.pytorch.autolog(log_models=False)

    # Loggers
    loggers = []
    tb_logger = pl.loggers.tensorboard.TensorBoardLogger(save_dir=experiment_log_dir, name=RUN_NAME,log_graph=True)

    loggers.append(tb_logger)

    # Callbacks
    callbacks = []


    # Profilers

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

    # main pytorch lightning trainer
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        log_every_n_steps=100,
        gpus=num_gpus,
        callbacks=callbacks,
        logger=loggers,
        strategy=strat,
        default_root_dir=root_dir #otherwise pytorch lightning will write to local
        #profiler=profiler # for tensorboard profiler
    )
    # Pass the datamodule as arg to trainer.fit to override model hooks :)
    if node_id == 0:
        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
            
            mlflow.log_param("model", model.model_tag)

            trainer.fit(model, data_module)

            # log model
            mlflow.pytorch.log_model(model, "models")

    else:
        trainer.fit(model, data_module)

    

if __name__ == '__main__':

    #main_train(data_path, AVAIL_GPUS)

    hr = HorovodRunner(np=2, driver_log_verbosity='all')

    model = hr.run(main_hvd())