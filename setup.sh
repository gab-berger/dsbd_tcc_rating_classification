#!/bin/bash
# Downloading dataset:
download_url="https://www.kaggle.com/api/v1/datasets/download/manishkr1754/capgemini-employee-reviews-dataset"
curl -L -o download.zip $download_url
unzip -o download.zip
rm -f download.zip