#!/bin/bash

# 1. Downloading dataset as a zip file:
curl -L -o dataset.zip https://www.kaggle.com/api/v1/datasets/download/manishkr1754/capgemini-employee-reviews-dataset

# 2. Extracting the csv from the zip file:
unzip dataset.zip