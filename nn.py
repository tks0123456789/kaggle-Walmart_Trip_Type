"""
Feature selection by xgb + NN(2 hidden layers), useUpc:False
Averaged 50 models each
Parameter tuning: params_tune_ensemble.py
Time: 21h
The running time on Ubuntu 14.04 LTS[i7 4790k, 32G MEM, GTX660]
"""
import numpy as np
import scipy as sp
import pandas as pd
import xgboost as xgb
from datetime import datetime
from sklearn.metrics import log_loss

from lasagne.layers import InputLayer, DropoutLayer, DenseLayer

from utility_common import feature_extraction, data_path
from utility_nn import build_net_sparse_input
from utility_xgb import feature_selection

# NN [6000+, 60, 100, 38], [6000+, 70, 90, 38]
# 2015/12/25-26 21h 

# X2.shape[1]:13916
X2, target, v_train, v_test = feature_extraction(useUpc=False)
y = pd.get_dummies(target).values.argmax(1)
N = X2.shape[0]

# X2[v_train-1]: training
# X2[v_test-1]: test

# Parameters
# r096, r104
nModels = 50
lr = .02
mm = .2
p = .1
bs = 256
params_lst = [{'h1':60, 'h2':100, 'max_epochs':390}, {'h1':70, 'h2':90, 'max_epochs':410}]

# XGB
num_round = 400
xgb_params = {'objective':'multi:softprob', 'num_class':38,
              'eta':.2, 'max_depth':6, 'colsample_bytree':.4, 'subsample':.8,
              'silent':1, 'nthread':8}

pr_nn_test_dict = {60:np.zeros((N/2, 38)), 70:np.zeros((N/2, 38))}
pr_nn_train_dict = {60:np.zeros((N/2, 38)), 70:np.zeros((N/2, 38))}

scores = []
t0 = datetime.now()
for j in range(nModels):
    seed = 9317*j + 3173
    xgb_params['seed'] = seed
    cols = feature_selection(X2[v_train-1], y, xgb_params, num_round)
    X = X2.tocsc()[:,cols].tocsr().astype(np.float32)
    for params in params_lst:
        h1 = params['h1']
        h2 = params['h2']
        max_epochs = params['max_epochs']
        pr_nn_train = pr_nn_train_dict[h1]
        pr_nn_test = pr_nn_test_dict[h1]
        np.random.seed(seed)
        net1 = build_net_sparse_input(h1, h2, p, mm, bs=bs, max_epochs=max_epochs, num_in=X.shape[1],
                                      shuffle=True, eval_size=False)
        net1.fit(X[v_train-1], y)
        pr = net1.predict_proba(X[v_train-1])
        # net1.save_params_to(data_path+'model/nn002_h1_'+str(h1)+'_'+str(j)+'.pkl')
        pr_nn_train += pr
        pr_nn_test += net1.predict_proba(X[v_test-1])
        pms = params.copy()
        pms.update({'seed':seed, 'nModel':j+1, 'ncol':X.shape[1],
                    'loss_each':log_loss(y, pr),
                    'loss_avg':log_loss(y, pr_nn_train/(j+1)),
                    # 'dist_test':np.sqrt(np.linalg.norm(pr_nn_test/(j+1)-pr_xgb_test)**2/(N/2)),
                    # 'dist_train':np.sqrt(np.linalg.norm(pr_nn_train/(j+1)-pr_xgb_train)**2/(N/2)),
                    # 'dist_train_each':np.sqrt(np.linalg.norm(pr-pr_xgb_train)**2/(N/2))
                })
        scores.append(pms)
        print scores[-1], datetime.now() - t0

pr002_nn = pd.DataFrame(scores)
#pr002_nn.to_csv(path_log+'pr002_nn.csv')

pr002_nn.ncol.mean()
# 6335.4200000000001

pr002_nn.tail(2).iloc[:,:-4]
#     dist_test  dist_train  dist_train_each  h1   h2  loss_avg  loss_each
# 98   0.308569    0.263576         0.276422  60  100  0.095248   0.104613
# 99   0.311610    0.274198         0.285436  70   90  0.080642   0.088053

# Pr002
pr002_nn.groupby('h1').loss_each.mean()
# h1
# 60    0.103445
# 70    0.087404

for h1 in [60, 70]:
    pr_nn_test_dict[h1] /= nModels
    np.save(data_path + 'pr_nn002_h1_' + str(h1) + '.npy', pr_nn_test_dict[h1])
