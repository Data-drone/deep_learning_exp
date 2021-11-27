# Scaling Deep Learning

A learning repo on scaling deep learning.

## Goals

- To be able to measure perf of Dataloaders and Algo
- To show how to parallelise across nodes and GPUs
- To look at how to build useful backtesting primatives for:
  - Change in Dataset
  - New Model 

## A First Notebook

0_pytorch_lightning_tutorial.ipynb

Simple basic pytorch lightning:
- Edits need to Trainer
- gpus set correctly
- add accelerator

Limitations:
- can't use the "nicer" accelerators.
- Seems like ddp is the only one with no caveats and that requires script?

## Things we found so far

Dataset structure is important to make sure we make optimal use of GPUs.
- Pytorch doesn't play nice by default with objectstores
- See: webdataset and the videos on this channel for details:
  - https://www.youtube.com/channel/UCzP3yTgqIiwHC_WN6at7KVQ
- webdataset is a designed to batch up the dataset into tarfiles, one batch per tarfile
  - How do we make it adaptable with to different batchsizes
- tar format may not make for good exploration of data with other tools so we need to have both raw data and batched data for training?  
