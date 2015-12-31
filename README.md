# kaggle-Walmart_Trip_Type
# 15th solution for the Walmart Recruiting: Trip Type Classification

## Classifier algorithms 
* NN1: Neural Networks with 2 hidden layers[6k+, 60, 100, 38] * 50
* NN2: Neural Networks with 2 hidden layers[6k+, 70, 90, 38] * 50
* XGB: XGBoost * 50

## Feature extraction methods  

## Feature selection methods
* NN1, NN2: xgboost

## Ensemble
* .6 * (NN1 + NN2)/2 + .4 * XGB

## Software
* Ubuntu 14.04 LTS
* xgboost-0.3
* Cuda 6.5
* python 2.7
* numpy 1.8.2
* scipy 0.13.3
* pandas 0.16.0
* scikit-learn 0.16.1 
* theano 0.7
* lasagne
* nolearn

## Usage
* python xgb.py
    * Output files: pr002_xgb_test.npy, pr002_xgb_train.npy
* python nn.py
    * Output files: pr_nn002_h1_60.npy, pr_nn002_h1_70.npy
* python make_submission.py
    * Output file: pred002.csv

