"""
XGB, useUpc:False, Averagd 50 models
Averaged 50 models each
Parameter tuning: params_tune_xgb.py
Time: 7h22m
The running time on Ubuntu 14.04 LTS[i7 4790k, 32G MEM, GTX660]
"""
import numpy as np
import scipy as sp
import pandas as pd
from datetime import datetime
import xgboost as xgb
from sklearn.metrics import log_loss

from utility_common import feature_extraction, data_path

# XGB
# 2015/12/25 7h22m
nModels = 50

# X1.shape[1]: 138610
X1, target, v_train, v_test = feature_extraction(useUpc=True)
y = pd.get_dummies(target).values.argmax(1)
N = X1.shape[0]

# Parameters
# r087

num_round = 550
xgb_params = {'objective':'multi:softprob', 'num_class':38,
              'eta':.2, 'max_depth':5, 'colsample_bytree':.4, 'subsample':.8,
              'silent':1, 'nthread':8}

dtrain = xgb.DMatrix(X1[v_train-1], label = y)
dtest = xgb.DMatrix(X1[v_test-1])

pr_xgb_test = np.zeros((N/2, 38))
pr_xgb_train = np.zeros((N/2, 38))

scores = []
t0 = datetime.now()
for j in range(nModels):
    seed = 13189*j + 471987
    xgb_params['seed'] = seed
    bst = xgb.train(xgb_params, dtrain, num_round)
    pr_xgb_test += bst.predict(dtest)
    pr = bst.predict(dtrain)
    pr_xgb_train += pr
    scores.append({'seed':seed, 'nModel':j+1,
                   'train_each':log_loss(y, pr),
                   'train_avg':log_loss(y, pr_xgb_train/(j+1))})
    print scores[-1], datetime.now() - t0

pr002 = pd.DataFrame(scores)
#pr002.to_csv(path_log+'pr002.csv')

pr002.tail(1)
#     nModel     seed  train_avg  train_each
# 49      50  1118248   0.272757    0.275803

log_loss(y, pr_xgb_train/nModels)
# 0.27275720210758203

pr_xgb_test /= nModels
np.save(data_path + 'pr002_xgb_test.npy', pr_xgb_test)

pr_xgb_train /= nModels
np.save(data_path + 'pr002_xgb_train.npy', pr_xgb_train)

