"""
Target parameters: max_epochs
Feature selection by xgb + NN(2 hidden layers), useUpc:False
Time: 26h
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

# r096
# 2015/12/23-24 1 day, 1:47:32.535909
# (h1,h2):(60, 100), (70, 90)
# CV pred
# Feature selection by XGB
# Shuffle data, No scaling, No nomalizing
# NN 2 hidden layers
# params: epochs (=max_epochs)

# ncol:13916
X4, target, v_train, v_test = feature_extraction(training, test, useUpc=False)
N = X4.shape[0]
X4 = X4[v_train-1]

# params for xgb
nt = 400
tc = 6
sh = .2
cs = .4
bf = .8
xgb_params = {'eta':sh, 'silent':1, 'objective':'multi:softprob', 'num_class':38,
              'max_depth':tc, 'colsample_bytree':cs, 'subsample':bf,
              'eval_metric':'mlogloss', 'nthread':8}

# params for nn
nModels = 20
epochs = 10
nIter = 42
# total epochs=epochs*nIter
lr = .02
mm = .2
p = .1
bs = 256


h1h2_lst = [{'h1':60, 'h2':100}, {'h1':70, 'h2':90}]
pr_nn_dict = {(params['h1'], params['h2']):[np.zeros((N/2, 38)) for _ in range(nIter)] \
              for params in h1h2_lst}
scores = []
t0 = datetime.now()
for fold, idx in enumerate(kf):
    train_idx, valid_idx = idx
    params_dict = {'fold':fold}
    for j in range(nModels):
        seed = 11281*j + 9108
        xgb_params['seed'] = seed
        cols = feature_selection(X4[train_idx], y[train_idx], xgb_params, nt)
        X = X4.tocsc()[:,cols].tocsr().astype(np.float32)
        for params in h1h2_lst:
            h1 = params['h1']
            h2 = params['h2']
            params_dict.update(params)
            pr_avg = pr_nn_dict[(h1, h2)]
            np.random.seed(seed)
            net1 = build_net_sparse_input(h1, h2, p, mm, max_epochs=epochs, num_in=X.shape[1],
                                    shuffle=True, eval_size=False)
            for k in range(nIter):
                net1.fit(X[train_idx], target[train_idx])
                pr = net1.predict_proba(X[valid_idx])
                pr_avg[k][valid_idx] += pr
                sc = params_dict.copy()
                sc.update({'epochs':epochs*(k+1), 'nModels':j+1,
                           'each':log_loss(target[valid_idx], pr),
                           'avg':log_loss(target[valid_idx], pr_avg[k][valid_idx]/(j+1))})
                scores.append(sc)
                print scores[-1], datetime.now() - t0

pr_nn_dict = {key:[pr/nModels for pr in val] for key, val in pr_nn_dict.iteritems()}

output = open(data_path + 'pr_nn096.pkl', 'wb')
pickle.dump(pr_nn_dict, output)
output.close()

r096 = pd.DataFrame(scores)
r096.to_csv('logs/r096.csv')

params = ['epochs', 'h1']
tmp = r096[r096.nModels==nModels].groupby(params).avg.mean().unstack()
r096_summary = tmp.join(r096.groupby(params).each.mean().unstack(), lsuffix='avg')
r096_summary.columns = pd.MultiIndex(levels=[['NN_avg', 'NN'], [60, 70]],
                                     labels=[[0, 0, 1, 1], [0, 1, 0, 1]],
                                     names=[u'h1', u'model'])

r096_summary.to_csv('logs/r096_summary.csv')

print pd.DataFrame({'loss':r096_summary.min(0), 'epochs':r096_summary.idxmin(0)})
#               epochs      loss
# h1     model                  
# NN_avg 60        230  0.615458
#        70        210  0.616497
# NN     60        150  0.692270
#        70        160  0.691234
