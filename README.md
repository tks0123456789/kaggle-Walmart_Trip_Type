## kaggle-Walmart_Trip_Type
## 15th solution for the Walmart Recruiting: Trip Type Classification

### Classifier algorithms 
* NN1: Neural Networks with 2 hidden layers[6k+, 60, 100, 38] * 50
* NN2: Neural Networks with 2 hidden layers[6k+, 70, 90, 38] * 50
* XGB: XGBoost * 50

### Feature extraction methods  

### Feature selection methods
* NN1, NN2: xgboost

### Ensemble
* .6 * (NN1 + NN2)/2 + .4 * XGB

### Software
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

### Usage
* Change data_path in utility_common.py to your data location
* Submission
  1. python xgb.py
    * Output files: pr002_xgb_test.npy, pr002_xgb_train.npy
  2. python nn.py
    * Output files: pr_nn002_h1_60.npy, pr_nn002_h1_70.npy
  3. python make_submission.py
    * Output file: pred002.csv
* Parameter tuning expriments
  1. python params_tune_xgb.py
    * Parameter tuning expriment for xgboost
    * Target parameters: max_depth, num_round
    * Output files: logs/r087.csv, pr_xgb087.pkl(6.4G)
  2. python params_tune_nn.py, pr_nn096.pkl(1.5G)
    * Parameter tuning expriment for NN
    * Target parameters: max_epochs
    * Output files: logs/r096.csv, logs/r096_summary.csv
  3. python params_tune_ensemble.py
    * Parameter tuning expriment for Ensemble
    * Target parameters: max_epochs
    * Output files: logs/r104.csv, logs/exp_ens_h1_60.png, logs/exp_ens_h1_70.png

