## 15th solution for the Walmart Recruiting: Trip Type Classification

### Classifier algorithms 
* NN1: Neural Networks with 2 hidden layers[6k+, 60, 100, 38]
* NN2: Neural Networks with 2 hidden layers[6k+, 70, 90, 38]
* XGB: XGBoost
* \*_avg: Averaged 50 model predictions of \*

### Feature extraction methods  

### Feature selection methods
* NN1, NN2: xgboost

### Ensemble
* .6 * (NN1_avg + NN2_avg)/2 + .4 * XGB_avg

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
  * python xgb.py
    * Output files: pr002_xgb_test.npy, pr002_xgb_train.npy
  * python nn.py
    * Output files: pr_nn002_h1_60.npy, pr_nn002_h1_70.npy
  * python make_submission.py
    * Output file: pred002.csv
  * Public: 0.52832, Private: 0.52625

* Parameter tuning expriments
  1. python params_tune_xgb.py
    * Parameter tuning expriment for xgboost
    * Target parameters: max_depth, num_round
    * Output files: logs/r087.csv, pr_xgb087.pkl(6.4G)

  2. python params_tune_nn.py
    * Parameter tuning expriment for NN
    * Target parameters: max_epochs
    * Output files: logs/r096.csv, logs/r096_summary.csv, pr_nn096.pkl(1.5G)

  3. python params_tune_ensemble.py
    * Parameter tuning expriment for Ensemble
    * Target parameters: max_epochs
    * Output files: logs/r104.csv, logs/exp_ens_h1_60.png, logs/exp_ens_h1_70.png

